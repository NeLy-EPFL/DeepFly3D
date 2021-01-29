# pylint: disable=unsubscriptable-object
import glob
import os
import pickle

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import deepfly.logger as logger
import deepfly.skeleton_fly as skeleton
from deepfly.Camera import Camera
from deepfly.Config import config
from deepfly.cv_util import triangulate_linear
from deepfly.os_util import read_camera_order
from deepfly.os_util import parse_img_name
import json
from itertools import product
import numpy as np

"""
def load_pred_from_json(path_json, folder_name, num_images):
    json_data = json.load(open(path_json, "r"))
    pred = np.zeros(
        (config["num_cameras"] + 1, num_images, skeleton.num_joints // 2, 2)
    )

    for session_id in json_data.keys():
        if folder_name in json_data[session_id]["data"]:
            for image_name in json_data[session_id]["data"][folder_name].keys():

                cid, img_id = parse_img_name(image_name)
                anot = json_data[session_id]["data"][folder_name][image_name][
                    "position"
                ]

                pred[cid, img_id, :15] = anot[:15]

    return pred
"""


def json2points2d(json, image_shape, num_images):
    """convert json df3d annotate format to df3d format
    """
    pred = np.zeros((7, num_images, skeleton.num_joints, 2))
    for k, v in json.items():
        v = json[k]
        p = np.array(json[k]["position"]) * image_shape
        img_id = int(k.split("_")[-1])
        cam_id = int(k.split("_")[1])
        p2 = np.zeros((38, 2))
        p2[:15] = p[:15]
        p2[19 : 19 + 15] = p[15:30]
        p2[18] = p[30]  # antenna
        p2[19 + 18] = p[35]  # antenna
        pred[cam_id, img_id] = p2

    return pred


def pred2points2d(pred, cam_id, cam_id_read, image_shape):
    """convert sh output to df3d format
    """

    if cam_id == 3:
        return pred_front2points2d(pred, cam_id, cam_id_read, image_shape)
    else:
        pred_cam = np.zeros(shape=(pred.shape[1], skeleton.num_joints, 2), dtype=float)
        num_joints = skeleton.num_joints
        num_images = pred.shape[1]

        if cam_id > 3:
            pred_cam[:num_images, num_joints // 2 :, :] = (
                pred[cam_id_read, :num_images] * image_shape
            )
        elif cam_id < 3:
            pred_cam[:num_images, : num_joints // 2, :] = (
                pred[cam_id_read, :num_images] * image_shape
            )

        return pred_cam


def pred_front2points2d(pred_front, cam_id, cam_id_read, image_shape):
    """convert sh output to df3d format
    """
    pred_cam = np.zeros(
        shape=(pred_front.shape[1], skeleton.num_joints, 2), dtype=float
    )

    assert cam_id == 3
    l = np.array([0, 1, 2, 3, 4, 7, 8, 9])
    m = l.shape[0]

    pred_cam[:, l] = pred_front[3][:, np.arange(m)]
    pred_cam[:, 19 + l] = pred_front[3][:, m + np.arange(m)]
    pred_cam[:, 18] = pred_front[3][:, -2]
    pred_cam[:, 19 + 18] = pred_front[3][:, -1]
    pred_cam *= image_shape
    return pred_cam


def find_pred_path(path_folder):
    pred_path_list = glob.glob(os.path.join(path_folder, "pred*.pkl"))
    pred_path_list.sort(key=os.path.getmtime)
    pred_path_list = pred_path_list[::-1]

    return pred_path_list[0] if len(pred_path_list) else None


def find_pred_front_path(path_folder):
    pred_path_list = glob.glob(os.path.join(path_folder, "front_pred*.pkl"))
    pred_path_list.sort(key=os.path.getmtime)
    pred_path_list = pred_path_list[::-1]

    return pred_path_list[0] if len(pred_path_list) else None


def find_calib_path(path_folder):
    pred_path_list = glob.glob(os.path.join(path_folder, "calib*"))
    pred_path_list.sort(key=os.path.getmtime)
    pred_path_list = pred_path_list[::-1]

    return pred_path_list[0] if len(pred_path_list) else None


def find_hm_path(path_folder):
    heatmap_path_list = glob.glob(os.path.join(path_folder, "heatmap*.pkl"))
    heatmap_path_list.sort(key=os.path.getmtime)
    heatmap_path_list = heatmap_path_list[::-1]

    return heatmap_path_list[0] if len(heatmap_path_list) else None


def find_pose_result_path(path_folder):
    heatmap_path_list = glob.glob(os.path.join(path_folder, "pose_result*.pkl"))
    heatmap_path_list.sort(key=os.path.getmtime)
    heatmap_path_list = heatmap_path_list[::-1]

    return heatmap_path_list[0] if len(heatmap_path_list) else None


def load_heatmap(hm_path, shape):
    logger.debug("Heatmap shape: {}".format(shape))
    heatmap = np.memmap(filename=hm_path, mode="r", shape=shape, dtype="float32")

    return heatmap


class CameraNetwork:
    def __init__(
        self,
        image_folder,
        output_folder,
        num_images=900,
        cam_list=None,
        cam_id_list=range(config["num_cameras"]),
        cid2cidread=None,
        pred=None,
    ):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.points3d = None
        self.num_images = num_images
        self.num_cameras = len(cam_id_list)

        self.cid2cidread = (
            cid2cidread
            if cid2cidread is not None
            else read_camera_order(self.output_folder)[0]
        )

        if cam_list:
            logger.debug("Camera list is already given, skipping loading.")
            self.cam_list = cam_list
            return

        self.cam_list = list()
        pred_path = find_pred_path(self.output_folder)
        pred_front_path = find_pred_front_path(self.output_folder)
        if pred_path is not None:
            pred = np.load(file=pred_path, mmap_mode="r", allow_pickle=True)
            pred = pred[:, : self.num_images]

            pred_front = np.load(file=pred_front_path, mmap_mode="r", allow_pickle=True)
            pred_front = pred_front[:, : self.num_images]
        else:
            pred = None
            pred_front = None

        for cam_id in cam_id_list:
            cam_id_read = self.cid2cidread[cam_id]
            if pred is not None:
                pred_cam = pred2points2d(
                    pred if cam_id != 3 else pred_front,
                    cam_id,
                    cam_id_read,
                    config["image_shape"],
                )
            else:
                pred_cam = None
            self.cam_list.append(
                Camera(
                    cid=cam_id,
                    cid_read=cam_id_read,
                    image_folder=image_folder,
                    points2d=pred_cam,
                    hm=None,
                )
            )

        calibration_path = find_calib_path(self.output_folder)
        if calibration_path is not None:
            calibration = np.load(file=calibration_path, allow_pickle=True)
            logger.debug("Reading calibration from {}".format(self.output_folder))
            _ = self.load_network(calibration)

    def set_default_camera_parameters(self):
        c = np.load(find_calib_path(config["calib_fine"]), allow_pickle=True)
        self.load_network(c)

    def __getitem__(self, item):
        return self.cam_list[item]

    def has_calibration(self):
        return np.all([c.P is not None for c in self.cam_list])

    def has_pose(self):
        return self.cam_list[0].points2d is not None

    def has_heatmap(self):
        return self.cam_list[0].hm is not None

    def triangulate(self):
        assert self.cam_list

        s = self.cam_list[0].points2d.shape
        self.points3d = np.zeros(shape=(s[0], s[1], 3), dtype=np.float)

        for img_id, j_id in product(range(s[0]), range(s[1])):
            cam_list_iter = list()
            points2d_iter = list()
            for cam in self.cam_list:
                if not (
                    np.any(cam[img_id, j_id, :] == 0)
                    or not config["skeleton"].camera_see_joint(cam.cam_id, j_id)
                ):
                    cam_list_iter.append(cam)
                    points2d_iter.append(cam[img_id, j_id, :])

            if len(cam_list_iter) >= 2:
                self.points3d[img_id, j_id, :] = triangulate_linear(
                    cam_list_iter, points2d_iter
                )

    def reprojection_error(self):
        ignore_joint_list = config["skeleton"].ignore_joint_id
        s = self.points3d.shape
        err = np.zeros((len(self.cam_list), s[0], s[1]))
        for (img_id, j_id, cam_idx) in product(
            range(s[0]), range(s[1]), range(len(self.cam_list))
        ):
            p3d = self.points3d[img_id, j_id].reshape(1, 3)
            if (
                config["skeleton"].camera_see_joint(self[cam_idx].cam_id, j_id)
                and not j_id in ignore_joint_list
                and not np.any(self.points3d[img_id, j_id, :] == 0)
                and not np.any(self[cam_idx][img_id, j_id, :] == 0)
            ):
                err[cam_idx, img_id, j_id] = np.sum(
                    np.abs(self[cam_idx].project(p3d) - self[cam_idx][img_id, j_id])
                )

        err_mean = np.mean(np.abs(err))
        logger.debug("Ignore_list {}:  {:.4f}".format(ignore_joint_list, err_mean))
        return err

    def prepare_bundle_adjust_param(self, camera_id_list=None, max_num_images=1000):
        ignore_joint_list = config["skeleton"].ignore_joint_id
        if camera_id_list is None:
            camera_id_list = list(range(self.num_cameras))
        camera_params = np.zeros(shape=(len(self.cam_list), 13), dtype=float)
        cam_list = self.cam_list
        for cid in range(len(self.cam_list)):
            camera_params[cid, 0:3] = np.squeeze(cam_list[cid].rvec)
            camera_params[cid, 3:6] = np.squeeze(cam_list[cid].tvec)
            camera_params[cid, 6] = cam_list[cid].focal_length_x
            camera_params[cid, 7] = cam_list[cid].focal_length_y
            camera_params[cid, 8:13] = np.squeeze(cam_list[cid].distort)

        point_indices = []
        camera_indices = []
        points2d_ba = []
        points3d_ba = []
        s = self.points3d.shape

        img_id_list = np.arange(s[0] - 1)
        if s[0] > max_num_images:
            logger.debug(
                "There are too many ({}) images for calibration. Selecting {} randomly.".format(
                    s[0], max_num_images
                )
            )
            img_id_list = np.random.randint(0, high=s[0] - 1, size=(max_num_images))

        d = dict()
        for img_id, j_id in product(img_id_list, range(s[1])):
            is_stripe = config["skeleton"].is_tracked_point(
                j_id, config["skeleton"].Tracked.STRIPE
            )
            points3d_ba.append(self.points3d[img_id, j_id])
            d[(img_id, j_id)] = len(points3d_ba) - 1
            for cam_idx, cam in enumerate(cam_list):
                if (
                    j_id not in ignore_joint_list
                    and not np.any(self.points3d[img_id, j_id] == 0)
                    and not np.any(cam[img_id, j_id] == 0)
                    and config["skeleton"].camera_see_joint(cam.cam_id, j_id)
                    and cam_idx in camera_id_list
                ):
                    points2d_ba.extend(cam[img_id, j_id])
                    point_indices.append(
                        d[(img_id, j_id)] if not is_stripe else d[(img_id, j_id % 19)]
                    )
                    camera_indices.append(cam_idx)

        points3d_ba = np.squeeze(np.array(points3d_ba))
        points2d_ba = np.squeeze(np.array(points2d_ba))
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)

        n_cameras = camera_params.shape[0]
        n_points = points3d_ba.shape[0]

        x0 = np.hstack((camera_params.ravel(), points3d_ba.ravel()))

        return (
            x0.copy(),
            points2d_ba.copy(),
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
        )

    def calibrate(self, cam_id_list=None):
        assert self.cam_list
        if cam_id_list is None:
            cam_id_list = range(self.num_cameras)

        self.reprojection_error()
        (
            x0,
            points_2d,
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
        ) = self.prepare_bundle_adjust_param(cam_id_list)
        logger.debug(f"Number of points for calibration: {n_points}")
        A = bundle_adjustment_sparsity(
            n_cameras, n_points, camera_indices, point_indices
        )
        res = least_squares(
            residuals,
            x0,
            jac_sparsity=A,
            verbose=2 if logger.debug_enabled() else 0,
            x_scale="jac",
            ftol=1e-4,
            method="trf",
            args=(
                self.cam_list,
                n_cameras,
                n_points,
                camera_indices,
                point_indices,
                points_2d,
            ),
            max_nfev=1000,
        )

        logger.debug(
            "Bundle adjustment, Average reprojection error: {}".format(
                np.mean(np.abs(res.fun))
            )
        )

        self.triangulate()
        return res

    def save_network(self, path, meta=None):
        if path is not None and os.path.exists(path):  # to prevent overwriting
            d = pickle.load(open(path, "rb"))
        else:
            d = {cam_id: dict() for cam_id in np.arange(0, 7)}
            d["meta"] = meta

        for cam in self.cam_list:
            d[cam.cam_id]["R"] = cam.R
            d[cam.cam_id]["tvec"] = cam.tvec
            d[cam.cam_id]["intr"] = cam.intr
            d[cam.cam_id]["distort"] = cam.distort

        if path is not None:
            pickle.dump(d, open(path, "wb"))

        return d

    def load_network(self, calib):
        d = calib
        if calib is None:
            return None
        for cam in self.cam_list:
            if cam.cam_id in d and d[cam.cam_id]:
                cam.set_R(d[cam.cam_id]["R"])
                cam.set_tvec(d[cam.cam_id]["tvec"])
                cam.set_intrinsic(d[cam.cam_id]["intr"])
                cam.set_distort(d[cam.cam_id]["distort"])
            else:
                logger.debug(
                    "Camera {} is not on the calibration file".format(cam.cam_id)
                )

        return d["meta"]

    def get_points2d_matrix(self):
        pts2d = np.zeros((7, self.num_images, config["num_joints"], 2), dtype=float)

        for cam in self.cam_list:
            pts2d[cam.cam_id, :] = cam.points2d.copy()

        return pts2d

    def set_points2d_matrix(self, pts2d):
        for cam in self.cam_list:
            cam.points2d[:] = pts2d[cam.cam_id]

    """
    STATIC
    """

    @staticmethod
    def calc_essential_matrix(points2d_1, points2d_2, intr):
        E, mask = cv2.findEssentialMat(
            points1=points2d_1,
            points2=points2d_2,
            cameraMatrix=intr,
            method=cv2.RANSAC,
            prob=0.9999,
            threshold=5,
        )
        logger.debug(
            "Essential matrix inlier ratio: {}".format(np.sum(mask) / mask.shape[0])
        )
        return E, mask

    @staticmethod
    def calc_Rt_from_essential(E, points1, points2, intr):
        _, R, t, mask, _ = cv2.recoverPose(
            E, points1=points1, points2=points2, cameraMatrix=intr, distanceThresh=100
        )
        return R, t, mask


def residuals(
    params,
    cam_list,
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    points_2d,
    residual_mask=None,
):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    assert point_indices.shape[0] == points_2d.shape[0]
    assert camera_indices.shape[0] == points_2d.shape[0]

    camera_params = params[: n_cameras * 13].reshape((n_cameras, 13))
    points3d = params[n_cameras * 13 :].reshape((n_points, 3))
    cam_indices_list = list(set(camera_indices))

    points_proj = np.zeros(shape=(point_indices.shape[0], 2), dtype=np.float)
    for cam_id in cam_indices_list:
        cam_list[cam_id].set_rvec(camera_params[cam_id][0:3])
        cam_list[cam_id].set_tvec(camera_params[cam_id][3:6])

        points2d_mask = camera_indices == cam_id
        points3d_where = point_indices[points2d_mask]
        points_proj[points2d_mask] = cam_list[cam_id].project(points3d[points3d_where])

    res = points_proj - points_2d
    res = res.ravel()

    return res


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    assert camera_indices.shape[0] == point_indices.shape[0]
    n_camera_params = 13
    m = camera_indices.size * 2
    # all the parameters, 13 camera parameters and x,y,z values for n_points
    n = n_cameras * n_camera_params + n_points * 3
    A = lil_matrix((m, n), dtype=int)  # sparse matrix
    i = np.arange(camera_indices.size)

    for s in range(n_camera_params):
        # assign camera parameters to points residuals (reprojection error)
        A[2 * i, camera_indices * n_camera_params + s] = 1
        A[2 * i + 1, camera_indices * n_camera_params + s] = 1

    for s in range(3):
        # assign 3d points to residuals (reprojection error)
        A[2 * i, n_cameras * n_camera_params + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * n_camera_params + point_indices * 3 + s] = 1

    return A
