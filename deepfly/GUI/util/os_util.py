import os

import numpy as np

import glob

from ..Config import config


def get_max_img_id(path):
    img_path_list = os.listdir(path)
    img_id_list = [
        int(os.path.basename(p).split("_")[-1].replace(".jpg", ""))
        for p in img_path_list
        if p.endswith(".jpg")
    ]
    return max(img_id_list)


def read_camera_order(folder):
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
    return int(name.split("_")[1]), int(name.split("_")[3].replace(".jpg", ""))


def constr_img_name(cid, pid, pad=True):
    if pad:
        return "camera_{}_img_{:06d}".format(cid, pid)
    else:
        return "camera_{}_img_{}".format(cid, pid)
