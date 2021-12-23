import os
import re

import df3d.logger as logger


def get_max_img_id(path):
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


def parse_img_name(name):
    match = re.match("camera_(\d+)_img_(\d+)", name.replace(".jpg", ""))
    return int(match[1]), int(match[2])


def parse_vid_name(name):
    match = re.match("camera_(\d+)", name.replace(".mp4", ""))
    return int(match[1])
