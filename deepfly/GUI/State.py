from enum import Enum
from .DB import PoseDB
from .util.os_util import get_max_img_id

class State:
    def __init__(self, folder, max_num_images, folder_output):
        self.folder = folder
        self.mode = Mode.IMAGE
        self.img_id = 0
        self.heatmap_joint_id = -1  # -1 corresponds to all joints
        self.correction_skip = True  # Correction Skip
    

class Mode(Enum):
    IMAGE = 1
    HEATMAP = 2
    POSE = 3
    CORRECTION = 4