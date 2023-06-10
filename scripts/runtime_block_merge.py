import copy
import itertools
import os
import sys
import traceback
from ui import adui

import modules.scripts as scripts

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import sd_models, shared, devices
from scripts.mbw_util.preset_weights import PresetWeights
import torch
from natsort import natsorted
from modules.script_callbacks import on_before_ui

presetWeights = PresetWeights()

shared.UNetBManager = None
shared.UNBMSettingsInjector = None

def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))

def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)

def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tap = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tap}_RuntimeBlockMerge_{item_id}{suf}"

class SettingsInjector():
    def __init__(self):
        self.enabled = False
        self.gui_weights = [0] * 27
        self.modelB = None

known_block_prefixes = [
    'input_blocks.0.',
    'input_blocks.1.',
    'input_blocks.2.',
    'input_blocks.3.',
    'input_blocks.4.',
    'input_blocks.5.',
    'input_blocks.6.',
    'input_blocks.7.',
    'input_blocks.8.',
    'input_blocks.9.',
    'input_blocks.10.',
    'input_blocks.11.',
    'middle_block.',
    'out.',
    'output_blocks.0.',
    'output_blocks.1.',
    'output_blocks.2.',
    'output_blocks.3.',
    'output_blocks.4.',
    'output_blocks.5.',
    'output_blocks.6.',
    'output_blocks.7.',
    'output_blocks.8.',
    'output_blocks.9.',
    'output_blocks.10.',
    'output_blocks.11.',
    'time_embed.'
]

class UNetStateManager(object):
    def __init__(self, org_unet: UNetModel = None):
        super().__init__()
        self.modelB_state_dict_by_blocks = []
        self.torch_unet = org_unet
        self.modelA_state_dict = None
        self.dtype = devices.dtype
        self.modelA_state_dict_by_blocks = []
        self.modelB_state_dict = None
        self.unet_block_module_list = [*self.torch_unet.input_blocks, self.torch_unet.middle_block, self.torch_unet.out,
                                       *self.torch_unet.output_blocks, self.torch_unet.time_embed]
        self.applied_weights = [0] * 27
        self.enabled = False
        self.modelA_path = shared.sd_model.sd_model_checkpoint
        self.modelB_path = ''
        self.force_cpu = False
        self.modelA_dtype = None
        self.modelB_dtype = None
        self.device = devices.get_cuda_device_string() if (torch.cuda.is_available() and not shared.cmd_opts.lowvram) else "cpu"


    def reload_modelA(self):
        if not self.enabled:
            return

        if self.modelA_path == shared.sd_model.sd_model_checkpoint and self.modelA_state_dict is not None:
            return
        self.modelA_path = shared.sd_model.sd_model_checkpoint

        del self.modelA_state_dict_by_blocks
        self.modelA_state_dict_by_blocks = []

        del self.modelA_state_dict
        torch.cuda.empty_cache()
        if self.force_cpu:
            self.modelA_state_dict = self.filter_unet_state_dict(
                sd_models.read_state_dict(self.modelA_path, map_location="cpu"))
            self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
            self.modelA_dtype = itertools.islice(self.modelA_state_dict.items(), 1).__next__()[1].dtype
        else:
            self.modelA_state_dict = copy.deepcopy(self.torch_unet.state_dict())
            self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)

        self.model_state_apply(self.applied_weights)
        print('model A reloaded')

    def load_modelB(self, modelB_path, force_cpu_checkbox, current_weights):
        self.force_cpu = force_cpu_checkbox
        self.device = devices.get_cuda_device_string() if (torch.cuda.is_available() and not shared.cmd_opts.lowvram) else "cpu"
        if self.force_cpu:
            self.device = "cpu"
        model_info = sd_models.get_closet_checkpoint_match(modelB_path)
        checkpoint_file = model_info.filename
        self.modelB_path = checkpoint_file


        if self.modelA_path == checkpoint_file:
            if not self.modelB_state_dict:
                self.enabled = False
            return False

        # move initialization of model A to here
        if not self.modelA_state_dict:
            if self.force_cpu:
                self.modelA_path = shared.sd_model.sd_model_checkpoint
                self.modelA_state_dict = self.filter_unet_state_dict(
                    sd_models.read_state_dict(self.modelA_path, map_location="cpu"))
                self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)

            else:
                self.modelA_state_dict = copy.deepcopy(self.torch_unet.state_dict())
                self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
            self.modelA_dtype = itertools.islice(self.modelA_state_dict.items(), 1).__next__()[1].dtype

        if self.modelB_state_dict:
            del self.modelB_state_dict_by_blocks
            del self.modelB_state_dict
            torch.cuda.empty_cache()
        self.modelB_state_dict_by_blocks = []
        self.modelB_state_dict = self.filter_unet_state_dict(
            sd_models.read_state_dict(checkpoint_file, map_location=self.device))
        self.modelB_dtype = itertools.islice(self.modelB_state_dict.items(), 1).__next__()[1].dtype
        if len(self.modelA_state_dict) != len(self.modelB_state_dict):
            print('modelA and modelB state dict have different length, aborting')
            return False
        self.map_blocks(self.modelB_state_dict, self.modelB_state_dict_by_blocks)
        # verify self.modelA_state_dict and self.modelB_state_dict have same structure
        self.model_state_apply(current_weights)

        print('Model ', os.path.split(modelB_path)[-1], " loaded")
        self.enabled = True
        return True

    def model_state_apply(self, current_weights):
        operation_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        for i in range(27):
            cur_block_state_dict = {}
            for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                if operation_dtype == torch.float32:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 current_weights[i]).to(self.dtype)
                else:
                    if self.force_cpu:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     current_weights[i]).to(self.dtype)
                    else:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])
                if str(shared.device) != self.device:
                    curlayer_tensor = curlayer_tensor.to(shared.device)
                cur_block_state_dict[cur_layer_key] = curlayer_tensor
            self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights

    def model_state_construct(self, current_weights):
        precision_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        result_state_dict = {}
        for i in range(27):
            for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                if precision_dtype == torch.float32:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 current_weights[i])
                else:
                    if self.force_cpu:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     current_weights[i]).to(torch.float16)
                    else:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])

                result_state_dict[known_block_prefixes[i] + cur_layer_key] = curlayer_tensor
        return result_state_dict



    def model_state_apply_modified_blocks(self, current_weights, current_model_B):
        if not self.enabled:
            return
        modelB_info = sd_models.get_closet_checkpoint_match(current_model_B)
        checkpoint_file_B = modelB_info.filename
        if checkpoint_file_B != self.modelB_path:
            print('model B changed, shouldn\'t happen')
            self.load_modelB(current_model_B, current_weights)
            return
        if self.applied_weights == current_weights:
            return
        operation_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    if operation_dtype == torch.float32:
                        curlayer_tensor = torch.lerp(
                            self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                            self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                            current_weights[i]).to(self.dtype)
                    else:
                        if self.force_cpu:
                            curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                         self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                         current_weights[i]).to(torch.float16)
                        else:
                            curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                         self.modelB_state_dict_by_blocks[i][cur_layer_key],
                                                         current_weights[i])
                    if str(shared.device) != self.device:
                        curlayer_tensor = curlayer_tensor.to(shared.device)
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights

    def model_state_apply_block(self, current_weights):
        if not self.enabled:
            return self.applied_weights
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights
        return self.applied_weights

    def filter_unet_state_dict(self, input_dict):
        filtered_dict = {}
        for key, value in input_dict.items():

            if key.startswith('model.diffusion_model'):
                filtered_dict[key[22:]] = value
        filtered_dict_keys = natsorted(filtered_dict.keys())
        filtered_dict = {k: filtered_dict[k] for k in filtered_dict_keys}

        return filtered_dict

    def map_blocks(self, model_state_dict_input, model_state_dict_by_blocks):
        if model_state_dict_by_blocks:
            print('mapping to non empty list')
            return
        model_state_dict_sorted_keys = natsorted(model_state_dict_input.keys())
        model_state_dict = {k: model_state_dict_input[k] for k in model_state_dict_sorted_keys}


        current_block_index = 0
        processing_block_dict = {}
        for key in model_state_dict:
            if not key.startswith(known_block_prefixes[current_block_index]):
                if not key.startswith(known_block_prefixes[current_block_index + 1]):
                    print(
                        f"unknown key {key} in statedict after block {known_block_prefixes[current_block_index]}, possible UNet structure deviation"
                    )
                    continue
                else:
                    model_state_dict_by_blocks.append(processing_block_dict)
                    processing_block_dict = {}
                    current_block_index += 1
            block_local_key = key[len(known_block_prefixes[current_block_index]):]
            processing_block_dict[block_local_key] = model_state_dict[key]

        model_state_dict_by_blocks.append(processing_block_dict)
        return

    def restore_original_unet(self):
        self.torch_unet.load_state_dict(self.modelA_state_dict)

    def unload_all(self):
        
        self.modelA_path = ''
        self.modelB_path = ''
        self.applied_weights = [0.0] * 27
        self.modelA_state_dict = None
        self.modelA_state_dict_by_blocks = []
        self.modelB_state_dict = None
        self.modelB_state_dict_by_blocks = []
        self.enabled = False


class Script(scripts.Script):
    def __init__(self):
        super().__init__()

        if shared.UNetBManager is None:
            try:
                shared.UNetBManager = UNetStateManager(shared.sd_model.model.diffusion_model)
            except AttributeError:
                shared.UNetBManager = None
            from modules.call_queue import wrap_queued_call

            def reload_modelA_checkpoint():
                if shared.opts.sd_model_checkpoint == shared.sd_model.sd_checkpoint_info.title:
                    return
                sd_models.reload_model_weights()
                shared.UNetBManager.reload_modelA()

            shared.opts.onchange("sd_model_checkpoint",
                                 wrap_queued_call(reload_modelA_checkpoint), call=False)

        if shared.UNBMSettingsInjector is None:
            shared.UNBMSettingsInjector = SettingsInjector()

    def ui(self, is_img2img):
        return adui(presetWeights)
        
    def title(self):
        return "Runtime block merging for UNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def extra_params(weights, model_b):
        weights = ','.join([str(s) for s in weights])
        return {
            'BlockMerge Model A': os.path.split(shared.sd_model.sd_model_checkpoint)[-1],
            'BlockMerge Model B': os.path.split(model_b)[-1],
            'BlockMerge Recipe': weights
        }

    @staticmethod
    def on_before_ui():
        def set_weights_value(field):
            def set_value_(p, x, xs):
                # convert weight string to tuple
                if field == "xyz_presets":
                    shared.UNBMSettingsInjector.weights = tuple(map(float, presetWeights.find_weight_by_name(x).split(",")))
                if field == "xyz_weights":
                    shared.UNBMSettingsInjector.weights = tuple(map(float,x.split(",")))
            return set_value_
        
        try:
            # add xyz grid options
            xyz_grid = None
            for script in scripts.scripts_data:
                if script.script_class.__module__ == "xyz_grid.py":
                    xyz_grid = script.module
                    break

            if xyz_grid is None:
                return

            preset_list = ["none"] + list(presetWeights.get_presets())

            axis = [
                xyz_grid.AxisOption(
                    "[RuntimeBlockMerge] Presets",
                    str,
                    set_weights_value("xyz_presets"),
                    choices=lambda: preset_list,
                ),
                xyz_grid.AxisOption(
                    "[RuntimeBlockMerge] Weights",
                    str,
                    set_weights_value("xyz_weights")
                ),
            ]

            if not any(x.label.startswith("[RuntimeBlockMerge]") for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(axis)
                
        except Exception:
            error = traceback.format_exc()
            print(
                f"[-] RuntimeBlockMerge: xyz_grid error:\n{error}",
                file=sys.stderr,
            )

    def process(self, p, *args):
        if shared.UNetBManager.enabled:
            # Check if injector has weights for xyz grid
            if hasattr(shared.UNBMSettingsInjector, "weights"):
                weight_list = list(shared.UNBMSettingsInjector.weights)
                
                while len(weight_list) < 25:
                    weight_list.append(0)
                if len(weight_list) > 27:
                    weight_list = weight_list[:27]
                    
                if len(weight_list) < 27:
                    if len(weight_list) < 26:
                        weight_list.append(args[25]) # add time_embed layer if missing
                    weight_list.append(args[26]) # add out layer if missing
                    
                shared.UNBMSettingsInjector.weights = tuple(weight_list)

            gui_weights = shared.UNBMSettingsInjector.weights if hasattr(shared.UNBMSettingsInjector, "weights") else args[:27]
            print("Merge block weights: ", gui_weights)
            model_b =  args[27]
            
            if hasattr(shared.UNBMSettingsInjector, "weights"):
                del shared.UNBMSettingsInjector.weights

            if not shared.UNetBManager:
                shared.UNetBManager = UNetStateManager(shared.sd_model.model.diffusion_model)

            # Add recipe to exif
            blockMergeExif = self.extra_params(gui_weights, args[27])
            p.extra_generation_params.update(blockMergeExif)
            shared.UNetBManager.model_state_apply_modified_blocks(gui_weights, model_b)

on_before_ui(Script.on_before_ui)