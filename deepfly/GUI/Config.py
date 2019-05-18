from . import skeleton
from enum import Enum


class Config:
    def __init__(self, folder):
        self.folder = folder
        self.mode = Mode.IMAGE
        self.view = View.Left
        self.img_id = 0
        self.hm_joint_id = -1  # -1 corresponds to all joints
        self.db = None
        self.camNet = None
        self.bone_param = None

        self.solve_bp = True  # Automatic correction
        self.already_corrected = False
        self.correction_skip = True  # Correction Skip

        self.num_cameras = 7
        self.num_images = None
        self.num_joints = skeleton.num_joints

        self.image_shape = [960, 480]
        self.heatmap_shape = [64, 128]

        self.max_num_images = None

        self.reproj_thr = {
            0: 30,
            1: 30,
            2: 30,
            3: 30,
            4: 30,
            5: 30,
            6: 30,
            7: 30,
            8: 30,
            9: 30,
            10: 30,
            11: 30,
            12: 30,
            13: 30,
            14: 30,
            15: 30,
            16: 30,
            17: 30,
            18: 30,
        }

        assert len(self.reproj_thr) == (skeleton.num_joints // 2)


class Mode(Enum):
    IMAGE = 1
    HEATMAP = 2
    POSE = 3
    CORRECTION = 4


class View(Enum):
    Left = 0
    Right = 1
