import ast
import numpy as np
from PyQt5.QtWidgets import *
from deepfly.GUI.Config import config
from deepfly.GUI.util.os_util import read_calib
    

def button_set_width(btn, text=" ", margin=20):
    width = btn.fontMetrics().boundingRect(text).width() + 7 + margin
    btn.setMaximumWidth(width)


def calibrate_calc(drosophAnnot, min_img_id, max_img_id):
    calib = read_calib(config["calib_fine"])
    assert(calib is not None)
    drosophAnnot.core.camNetAll.load_network(calib)

    # take a copy of the current points2d
    pts2d = np.zeros(
        (7, drosophAnnot.core.num_images, config["skeleton"].num_joints, 2),
        dtype=float,
    )
    for cam_id in range(config["num_cameras"]):
        pts2d[cam_id, :] = drosophAnnot.core.camNetAll[cam_id].points2d.copy()

    # ugly hack to temporarly incorporate manual corrections to calibration
    c = 0
    for cam_id in range(config["num_cameras"]):
        for img_id in range(drosophAnnot.core.num_images):
            if drosophAnnot.state.db.has_key(cam_id, img_id):
                pt = drosophAnnot.state.db.read(cam_id, img_id) * config["image_shape"]
                drosophAnnot.core.camNetAll[cam_id].points2d[img_id, :] = pt
                c += 1
    print("Calibration: replaced {} points from manuall correction".format(c))

    # keep the pts only in the range
    for cam in drosophAnnot.core.camNetAll:
        cam.points2d = cam.points2d[min_img_id:max_img_id, :]

    drosophAnnot.core.camNetLeft.triangulate()
    drosophAnnot.core.camNetLeft.bundle_adjust(cam_id_list=(0,1,2), unique=False, prior=True)
    drosophAnnot.core.camNetRight.triangulate()
    drosophAnnot.core.camNetRight.bundle_adjust(cam_id_list=(0,1,2), unique=False, prior=True)
    
    # put old values back
    for cam_id in range(config["num_cameras"]):
        drosophAnnot.core.camNetAll[cam_id].points2d = pts2d[cam_id, :].copy()

    drosophAnnot.save_calibration()
