import os

from deepfly import skeleton_h36m, skeleton_fly

default = {
    "flip_cameras": [],
    # belief propagation
    "num_peak": 10,
    "upper_bound": 200,
    "alpha_reproj": 30,
    "alpha_heatmap": 600,
    "alpha_bone": 10,
}

config_fly = {
    "name": "fly",
    "num_cameras": 7,
    "image_shape": [960, 480],
    "heatmap_shape": [64, 128],
    "left_cameras": [0, 1, 2],
    "right_cameras": [6, 5, 4],
    # skeleton
    "skeleton": skeleton_fly,
    "bones": skeleton_fly.bones,
    "bone_param": skeleton_fly.bone_param,
    "num_joints": skeleton_fly.num_joints,
    # plotting
    "line_thickness": 3,
    "scatter_r": 6,
    # pose estimation
    "resume": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../weights/sh8_deepfly.tar",
    ),
    "resume_front": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../weights/sh8_deepfly_front.tar"
    ),
    "num_stacks": 2,
    "batch_size": 12,
    "flip_cameras": [4, 5, 6],
    "num_predict": skeleton_fly.num_joints // 2,
    "mean": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../weights/mean.pth.tar"
    ),
    # 3d pose
    "reproj_thr": {v: 40 for v in range(skeleton_fly.num_joints)},
    # calibration
    "calib_rough": {
        0: 0 / 57.2,
        1: -30 / 57.2,
        2: -70 / 57.2,
        3: -125 / 57.2,
        6: +110 / 57.2,
        5: +150 / 57.2,
        4: +179 / 57.2,
    },
    "calib_fine": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../data/template/"
    ),
    # belief propagation
    "num_peak": 10,
    "upper_bound": 200,
    "alpha_reproj": 30,
    "alpha_heatmap": 600,
    "alpha_bone": 10,
    # procrustes
    "procrustes_apply": True,
    "procrustes_template": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../data/template/"
    ),
}

config_h36m = {
    "name": "h36m",
    "num_cameras": 4,
    "image_shape": [500, 500],
    "heatmap_shape": [128, 128],
    "checkpoint": None,
    "left_cameras": [0, 1, 2, 3],
    "right_cameras": [],
    # skeleton
    "skeleton": skeleton_h36m,
    "bones": skeleton_h36m.bones,
    "bone_param": skeleton_h36m.bone_param,
    "num_joints": skeleton_h36m.num_joints,
    # pose 2d
    "resume": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../weights/sh8_mpii.tar"
    ),
    "num_stacks": 8,
    "batch_size": 64,
    "num_predict": skeleton_h36m.num_joints,
    "mean": os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../weights/mean_mpii.pth.tar"
    ),
    # plotting
    "line_thickness": 2,
    "scatter_r": 2,
    # calibration
    "calib_rough": {
        0: 0 / 57.2,
        1: -30 / 57.2,
        2: -70 / 57.2,
        3: -125 / 57.2,
        6: +110 / 57.2,
        5: +150 / 57.2,
        4: +179 / 57.2,
    },
    # belief propagation
    "num_peak": 5,
    "upper_bound": 100,
    "alpha_reproj": 30,
    "alpha_heatmap": 600,
    "alpha_bone": 0,
}

"""
# setting defaults if they are missing
config_fly = dict(list(default.items()) + list(config_fly.items()))
config_h36m = dict(list(default.items()) + list(config_h36m.items()))
"""

config = config_fly
