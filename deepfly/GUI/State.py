from enum import Enum
from .DB import PoseDB
from .util.os_util import get_max_img_id

class State:
    def __init__(self, folder, max_num_images, folder_output):
        self.folder = folder
        self.mode = Mode.IMAGE
        self.view = View.Left
        self.img_id = 0
        self.hm_joint_id = -1  # -1 corresponds to all joints
        self.db = PoseDB(folder_output)
        self.bone_param = None
        self.solve_bp = True  # Automatic correction
        self.already_corrected = False
        self.correction_skip = True  # Correction Skip
        

class Mode(Enum):
    IMAGE = 1
    HEATMAP = 2
    POSE = 3
    CORRECTION = 4


class View(Enum):
    Left = 0
    Right = 1