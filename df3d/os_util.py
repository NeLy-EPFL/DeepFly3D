import glob
import os
import re

import numpy as np
from df3d.config import config


import df3d.logger as logger


def get_max_img_id(path):
    print(path)
    bound_low = 0
    bound_high = 100000

    curr = (bound_high + bound_low) // 2
    while bound_high - bound_low > 1:
        if image_exists_img_id(path, curr):
            bound_low = curr
        else:
            bound_high = curr
        curr = (bound_low + bound_high) // 2

    if not image_exists_img_id(path, curr):
        logger.debug("Cannot find image at {} with img_id {}".format(path, curr))
        raise FileNotFoundError("No image found.")

    return curr


def image_exists_img_id(path, img_id):
    return os.path.isfile(
        os.path.join(path, constr_img_name(0, img_id, False)) + ".jpg"
    ) or os.path.isfile(os.path.join(path, constr_img_name(0, img_id, True)) + ".jpg")


def constr_img_name(cid, pid, pad=True):
    if pad:
        return "camera_{}_img_{:06d}".format(cid, pid)
    else:
        return "camera_{}_img_{}".format(cid, pid)


def read_camera_order(folder):
    assert os.path.isdir(
        folder
    ), "Trying to call read_camera_order on {}, which is not a folder".format(folder)

    path = os.path.join(folder, "./cam_order.npy")
    if os.path.isfile(path):
        order = np.load(file=path, allow_pickle=True)
    else:
        order = np.arange(config["num_cameras"])
        write_camera_order(folder, order)
        logger.debug(
            "Could not find camera order under {}. Writing the default ordering {}.".format(
                folder, order
            )
        )

    cidread2cid = order.copy()
    cid2cidread = np.zeros(cidread2cid.size, dtype=int)
    for cidread, cid in enumerate(cidread2cid):
        cid2cidread[cid] = cidread

    return cidread2cid, cid2cidread


def write_camera_order(folder, cidread2cid):
    assert os.path.isdir(
        folder
    ), "Trying to write_camera_order into {}, which is not a folder".format(folder)

    path = os.path.join(folder, "cam_order")
    logger.debug(
        "Writing the camera ordering {} into folder {}".format(cidread2cid, folder)
    )
    # print("Saving camera order {}: {}".format(path, cidread2cid))
    np.save(path, cidread2cid)


def parse_img_name(name):
    match = re.match("camera_(\d+)_img_(\d+)", name.replace(".jpg", ""))
    return int(match[1]), int(match[2])

def parse_vid_name(name):
    match = re.match("camera_(\d+)", name.replace(".mp4", ""))
    #print(match, name)
    return int(match[1])
