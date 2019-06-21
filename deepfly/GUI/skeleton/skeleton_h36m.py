from enum import Enum

import numpy as np


class Tracked(Enum):
    ANKLE = 0
    HEAD_TOP = 1
    WRIST = 3
    KNEE = 4
    PELVIS = 5
    THORAX = 6,
    UPPER_NECK = 7
    SHOULDER = 8
    ELBOW = 9
    HIP = 10


tracked_points = [
    Tracked.ANKLE,
    Tracked.KNEE,
    Tracked.HIP,
    Tracked.HIP,
    Tracked.KNEE,
    Tracked.ANKLE,
    Tracked.PELVIS,
    Tracked.THORAX,
    Tracked.UPPER_NECK,
    Tracked.HEAD_TOP,
    Tracked.WRIST,
    Tracked.ELBOW,
    Tracked.SHOULDER,
    Tracked.SHOULDER,
    Tracked.ELBOW,
    Tracked.WRIST
]

limb_id = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]

bones = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15]
]

bones3d = []

colors = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (150, 100, 150),
    (255, 165, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (150, 200, 200),
    (255, 165, 0),
]

num_joints = len(tracked_points)
num_limbs = len(set(limb_id))


def is_tracked_point(joint_id, tracked_point):
    return tracked_points[joint_id] == tracked_point


def get_limb_id(joint_id):
    return limb_id[joint_id]


def is_joint_visible_left(joint_id):
    return True


def is_joint_visible_right(joint_id):
    return True


def is_limb_visible_left(limb_id):
    return True


def is_limb_visible_right(limb_id):
    return True


def is_limb_visible_mid(limb_id):
    return True


bone_param = np.ones((num_joints, 2), dtype=float)
bone_param[:, 0] = 1
bone_param[:, 1] = 99999

pictorial_joint_list = [j for j in range(num_joints)]

ignore_joint_id = []
zorder = np.arange(num_limbs)
zorder = [zorder[get_limb_id(j)] for j in range(num_joints)]


def get_zorder(cam_id):
    return zorder


def camera_see_joint(camera_id, joint_id):
    return True
