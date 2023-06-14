import os
from csv import DictReader, DictWriter

from modules import scripts


CSV_FILE_PATH = "csv/preset.tsv"
MYPRESET_PATH = "csv/custom_preset.tsv"
HEADER = ["preset_name", "preset_weights"]
path_root = scripts.basedir()

class PresetWeights():
    def __init__(self):
        self.presets = {}
        self.custom_presets = {}
        self.load_presets()

    def load_presets(self):
        self.presets = {}
        self.custom_presets = {}
        if os.path.exists(os.path.join(path_root, MYPRESET_PATH)):
            with open(os.path.join(path_root, MYPRESET_PATH), "r") as f:
                reader = DictReader(f, delimiter=":")
                lines_dict = [row for row in reader]
                for line_dict in lines_dict:
                    _w = ",".join([f"{x.strip()}" for x in line_dict["preset_weights"].split(",")])
                    self.presets.update({line_dict["preset_name"]: _w})
                    self.custom_presets.update({line_dict["preset_name"]: _w})

        with open(os.path.join(path_root, CSV_FILE_PATH), "r") as f:
            reader = DictReader(f, delimiter=":")
            lines_dict = [row for row in reader]
            for line_dict in lines_dict:
                _w = ",".join([f"{x.strip()}" for x in line_dict["preset_weights"].split(",")])
                self.presets.update({line_dict["preset_name"]: _w})

    def save_custom_presets(self, custom_presets):
        # Split the string into lines
        lines = custom_presets.strip().split('\n')

        # Create a list of dictionaries
        data = []
        
        for line in lines:
            values = line.split(':')
            if len(values) == 2:
                data.append({HEADER[0]: values[0], HEADER[1]: values[1]})
                
        with open(os.path.join(path_root, MYPRESET_PATH), "w", newline='') as f:
            writer = DictWriter(f, fieldnames=HEADER, delimiter=":")
            writer.writeheader()
            writer.writerows(data)
        self.load_presets()
    
    def get_preset_name_list(self):
        return [k for k in self.presets.keys()]

    def get_presets(self):
        return self.presets
    
    def get_custom_presets(self):
        return self.custom_presets

    def find_weight_by_name(self, preset_name=""):
        if preset_name and preset_name != "" and preset_name in self.presets.keys():
            return self.presets.get(preset_name, ",".join(["0.5" for _ in range(25)]))
        else:
            return ""

    def find_names_by_weight(self, weights=""):
        if weights and weights != "":
            if weights in self.presets.values():
                return [k for k, v in self.presets.items() if v == weights]
            else:
                _val = ",".join([f"{x.strip()}" for x in weights.split(",")])
                if _val in self.presets.values():
                    return [k for k, v in self.presets.items() if v == _val]
                else:
                    return []
        else:
            return []
