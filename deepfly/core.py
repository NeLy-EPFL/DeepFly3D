import os.path

from deepfly.utils_ramdya_lab import find_default_camera_ordering
from deepfly.GUI.Config import config
from deepfly.GUI.util.os_util import write_camera_order, read_camera_order

class Core:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_folder = os.path.join(self.input_folder, 'df3d/')
        
        self.cidread2cid = None
        self.cid2cidread = None
        self.setup_camera_ordering()


    @property
    def input_folder(self): 
        return self._input_folder


    @input_folder.setter 
    def input_folder(self, value): 
        value = os.path.abspath(value)
        value = value.rstrip('/')
        assert os.path.isdir(value), f'Not a directory {value}'
        self._input_folder = value 


    @property
    def output_folder(self): 
        return self._output_folder


    @output_folder.setter 
    def output_folder(self, value): 
        os.makedirs(value, exist_ok=True)
        value = os.path.abspath(value)
        value = value.rstrip('/')
        assert os.path.isdir(value), f'Not a directory {value}'
        self._output_folder = value 



    def setup_camera_ordering(self):
        default = find_default_camera_ordering(self.input_folder)
        if default:
            self.update_camera_ordering(default)
        else:
            self.cidread2cid, self.cid2cidread = read_camera_order(self.output_folder)


    def update_camera_ordering(self, cidread2cid):
        if len(cidread2cid) != config["num_cameras"]:
            print(f"Cannot rename images as there are no {config['num_cameras']} values")
            return False

        print("Camera order {}".format(cidread2cid))
        write_camera_order(self.output_folder, cidread2cid)
        self.cidread2cid, self.cid2cidread = read_camera_order(self.output_folder)
        return True

    