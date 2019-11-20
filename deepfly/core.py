
from deepfly.utils_ramdya_lab import find_default_camera_ordering
from deepfly.GUI.Config import config
from deepfly.GUI.util.os_util import write_camera_order, read_camera_order

class Core:
    def __init__(self, input_folder, output_folder):
        self.folder = input_folder
        self.folder_output = output_folder
        self.cidread2cid = None
        self.cid2cidread = None

    def setup_camera_ordering(self):
        default = find_default_camera_ordering(self.folder)
        if default:
            self.update_camera_ordering(default)
        else:
            self.cidread2cid, self.cid2cidread = read_camera_order(self.folder_output)

    def update_camera_ordering(self, cidread2cid):
        if len(cidread2cid) != config["num_cameras"]:
            print(f"Cannot rename images as there are no {config['num_cameras']} values")
            return False

        print("Camera order {}".format(cidread2cid))
        write_camera_order(self.folder_output, cidread2cid)
        self.cidread2cid, self.cid2cidread = read_camera_order(self.folder_output)
        return True

    