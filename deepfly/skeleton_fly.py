from enum import Enum

import numpy as np


class Tracked(Enum):
    BODY_COXA = 0
    COXA_FEMUR = 1
    FEMUR_TIBIA = 2
    TIBIA_TARSUS = 3
    TARSUS_TIP = 4
    ANTENNA = 5
    STRIPE = 6


tracked_points = [
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.ANTENNA,
    Tracked.STRIPE,
    Tracked.STRIPE,
    Tracked.STRIPE,
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.BODY_COXA,
    Tracked.COXA_FEMUR,
    Tracked.FEMUR_TIBIA,
    Tracked.TIBIA_TARSUS,
    Tracked.TARSUS_TIP,
    Tracked.ANTENNA,
    Tracked.STRIPE,
    Tracked.STRIPE,
    Tracked.STRIPE,
]
limb_id = [
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    8,
    9,
    9,
    9,
]

__limb_visible_left = [
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
]

__limb_visible_right = [
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
]

__limb_visible_mid = [
    True,
    True,
    False,
    True,
    False,
    True,
    True,
    False,
    True,
    False,
]

bones = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [16, 17],
    [17, 18],
    [19, 20],
    [20, 21],
    [21, 22],
    [22, 23],
    [24, 25],
    [25, 26],
    [26, 27],
    [27, 28],
    [29, 30],
    [30, 31],
    [31, 32],
    [32, 33],
    [35, 36],
    [36, 37],
]

bones3d = [[15, 34]]

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
    return __limb_visible_left[get_limb_id(joint_id)]


def is_joint_visible_right(joint_id):
    return __limb_visible_right[get_limb_id(joint_id)]


def is_limb_visible_left(limb_id):
    return __limb_visible_left[limb_id]


def is_limb_visible_right(limb_id):
    return __limb_visible_right[limb_id]


def is_limb_visible_mid(limb_id):
    return __limb_visible_mid[limb_id]


def camera_see_limb(camera_id, limb_id):
    if camera_id < 3:
        return is_limb_visible_left(limb_id)
    elif camera_id == 3:
        return is_limb_visible_mid(limb_id)
    elif camera_id > 3:
        return is_limb_visible_right(limb_id)
    else:
        raise NotImplementedError


def camera_see_joint(camera_id, joint_id):
    if camera_id == 7:
        camera_id = 3
    if camera_id in [2, 4]:  # they cannot see the stripes
        return camera_see_limb(camera_id, limb_id[joint_id]) and not (
                tracked_points[joint_id] == Tracked.STRIPE
        )
    elif camera_id == 3:
        return (
                camera_see_limb(camera_id, limb_id[joint_id])
                and tracked_points[joint_id] != Tracked.BODY_COXA
                and tracked_points[joint_id] != Tracked.COXA_FEMUR
        )
    elif camera_id in [0, 1, 5, 6]:
        return camera_see_limb(camera_id, limb_id[joint_id])
    else:
        raise NotImplementedError


bone_param = np.ones((num_joints, 2), dtype=float)
bone_param[:, 0] = 0.9
bone_param[:, 1] = 0.3
for joint_id in range(num_joints):
    if is_tracked_point(joint_id, Tracked.BODY_COXA) or is_tracked_point(joint_id, Tracked.STRIPE) or is_tracked_point(
            joint_id, Tracked.ANTENNA):
        bone_param[joint_id, 1] = 10000  # no bone

# joints to be ignored during calibration
ignore_joint_id = [
    joint_id
    for joint_id in range(num_joints)
    if
    is_tracked_point(joint_id, Tracked.BODY_COXA) or is_tracked_point(joint_id, Tracked.COXA_FEMUR) or is_tracked_point(
        joint_id, Tracked.ANTENNA)
]

# joints to be ignored during calibration
ignore_joint_id_wo_stripe = [
    joint_id
    for joint_id in range(num_joints)
    if
    is_tracked_point(joint_id, Tracked.BODY_COXA) or is_tracked_point(joint_id, Tracked.COXA_FEMUR) or is_tracked_point(
        joint_id, Tracked.ANTENNA)
]
pictorial_joint_list = [j for j in range(num_joints)]

zorder_left = [7, 8, 6, 9, 5, 1, 0, 2, 3, 4]
zorder_right = [1, 0, 2, 3, 4, 7, 8, 6, 9, 5]
zorder_mid = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

zorder_left = [zorder_left[get_limb_id(j)] for j in range(num_joints)]
zorder_right = [zorder_right[get_limb_id(j)] for j in range(num_joints)]
zorder_mid = [zorder_mid[get_limb_id(j)] for j in range(num_joints)]


def get_zorder(cam_id):
    if cam_id < 3:
        zorder = zorder_right
    elif cam_id == 3:
        zorder = zorder_mid
    elif cam_id > 3:
        zorder = zorder_left
    else:
        raise NotImplementedError
    zorder = np.max(zorder) - zorder
    return zorder
