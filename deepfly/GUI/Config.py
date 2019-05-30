from . import skeleton
from enum import Enum

#image_shape = [960, 480]
#heatmap_shape = [64, 128]
image_shape = [950,950]
heatmap_shape = [128,128]
thickness = 6
r = 3


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


        self.num_images = None
        self.num_joints = skeleton.num_joints

        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

        self.max_num_images = None

        self.num_cameras = 7
        #self.initial_angles = []
        self.reproj_thr = {v:30 for v in range(skeleton.num_joints)}
        '''
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
        '''

        #assert(len(self.initial_angles)==self.num_cameras)
        assert len(self.reproj_thr) == (skeleton.num_joints)


class Mode(Enum):
    IMAGE = 1
    HEATMAP = 2
    POSE = 3
    CORRECTION = 4


class View(Enum):
    Left = 0
    Right = 1
