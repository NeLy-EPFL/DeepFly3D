import glob
import os

import numpy as np
from pathlib import Path
from ..Config import config
import re

image_name_re = re.compile(r'camera_0_img_([0-9]+).jpg')


def get_max_img_id(path):
    files = os.listdir(path)
    matches = (image_name_re.search(f) for f in files)
    ids = [int(m.groups()[0]) for m in matches if m and m.groups()] 
    m = max(ids) if ids else 0
    print('Max id is {}'.format(m))
    return m


def image_exists_img_id(path, img_id):
    return os.path.isfile(os.path.join(path, constr_img_name(0, img_id, False)) + '.jpg') or os.path.isfile(
        os.path.join(path, constr_img_name(0, img_id, True)) + '.jpg')


def constr_img_name(cid, pid, pad=True):
    if pad:
        return "camera_{}_img_{:06d}".format(cid, pid)
    else:
        return "camera_{}_img_{}".format(cid, pid)


def read_camera_order(folder):
    assert(folder.endswith('df3d/') or folder.endswith('df3d'))

    path = os.path.join(folder, "cam_order.npy")
    if os.path.isfile(path):
        order = np.load(file=path, allow_pickle=True)
    else:
        order = np.arange(config["num_cameras"])
        write_camera_order(folder, order)

    cidread2cid = order.copy()
    cid2cidread = np.zeros(cidread2cid.size, dtype=int)
    for cidread, cid in enumerate(cidread2cid):
        cid2cidread[cid] = cidread

    return cidread2cid, cid2cidread


def write_camera_order(folder, cidread2cid):
    assert (folder.endswith('df3d/') or folder.endswith('df3d'))
    path = os.path.join(folder, "cam_order")
    # print("Saving camera order {}: {}".format(path, cidread2cid))
    try:
        np.save(path, cidread2cid)
    except:
        return


def read_calib(folder):
    calibration_path = glob.glob(os.path.join(folder, "calib*.pkl"))
    if len(calibration_path) > 0:
        calibration_path = calibration_path[0]
        calib = np.load(file=calibration_path, allow_pickle=True)
    else:
        # print("Cannot read calibration file from the folder {}".format(folder))
        calibration_path = None
        calib = None

    return calib


def parse_img_name(name):
    #print('Parsing name: {}'.format(name))
    return int(name.split("_")[1]), int(name.split("_")[3].replace(".jpg", ""))


def constr_img_name(cid, pid, pad=True):
    if pad:
        return "camera_{}_img_{:06d}".format(cid, pid)
    else:
        return "camera_{}_img_{}".format(cid, pid)
