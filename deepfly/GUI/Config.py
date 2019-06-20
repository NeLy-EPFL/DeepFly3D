import os

from .skeleton import skeleton_fly

config_fly = {
    "num_cameras": 7,
    "image_shape": [960, 480],
    "heatmap_shape": [64, 128],

    # skeleton
    "skeleton": skeleton_fly,
    "bones": skeleton_fly.bones,
    "bone_param": skeleton_fly.bone_param,
    "num_joints": skeleton_fly.num_joints,

    # plotting
    "line_thickness": 5,
    "scatter_r": 5,

    # pose estimation
    "resume": os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/sh8_deepfly.tar"),
    "num_stacks": 4,
    "batch_size": 64,

    # 3d pose
    "reproj_thr": {v: 30 for v in range(skeleton_fly.num_joints)},

    # calibration
    "calib_rough":
        {
            0: 0 / 57.2,
            1: -30 / 57.2,
            2: -70 / 57.2,
            3: -125 / 57.2,
            6: +110 / 57.2,
            5: +150 / 57.2,
            4: +179 / 57.2
        },
    "calib_fine": os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               "../../data/test/calib__home_user_Desktop_DeepFly3D_data_test.pkl"),

    # belief propagation
    "num_peak": 10,
    "upper_bound": 200,

    "alpha_reproj": 30,
    "alpha_heatmap": 600,
    "alpha_bone": 10
}

config_h36m = {
    "image_shape": [950, 950],
    "heatmap_shape": [128, 128],
    "checkpoint": None
}

config = config_fly
