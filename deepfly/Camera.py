import math
import os

import cv2
import pims
import numpy as np
import skimage
import skimage.feature

from deepfly.Config import config
from deepfly.plot_util import plot_drosophila_2d, plot_drosophila_heatmap


class Camera:
    def __init__(self, cid, image_folder, hm=None, points2d=None, cid_read=None,
                 from_video=False):
        self.cam_id = cid
        self.cam_id_read = cid_read if cid_read is not None else cid
        self.image_folder = image_folder
        self.points2d = points2d  # pixel coordinates, not normalized
        self.hm = hm

        self.R = None
        self.rvec = None
        self.tvec = None
        self.image_shape = (480, 960)
        self.focal_length_x = 16041.0
        self.focal_length_y = 15971.7
        self.cx = 240
        self.cy = 480
        self.intr = np.array(
            [
                [self.focal_length_x, 0, self.cx],
                [0, self.focal_length_y, self.cy],
                [0, 0, 1],
            ],
            dtype=float,
        )
        self.distort = np.zeros(5, dtype=np.float)
        self.P = None
        self.from_video = from_video
        if from_video:
            video_path = os.path.join(image_folder,
                                      'camera_%d.mp4' % self.cam_id_read)
            self.video_obj = pims.Video(video_path)

    def set_intrinsic(self, intrinsic):
        self.intr = intrinsic
        self.focal_length_x = intrinsic[0, 0]
        self.focal_length_y = intrinsic[1, 1]
        self.P = Camera.calc_projection_matrix(self.R, self.tvec, self.intr)

    def set_R(self, R, set_rvec=True):
        self.R = R
        if set_rvec:
            self.set_rvec(cv2.Rodrigues(R)[0], set_R=False)
        self.P = Camera.calc_projection_matrix(self.R, self.tvec, self.intr)

    def set_rvec(self, rvec, set_R=True):
        self.rvec = np.squeeze(rvec)
        if set_R:
            self.set_R(cv2.Rodrigues(rvec)[0], set_rvec=False)
        self.P = Camera.calc_projection_matrix(self.R, self.tvec, self.intr)

    def set_tvec(self, tvec):
        self.tvec = tvec.astype(np.float)
        self.P = Camera.calc_projection_matrix(self.R, self.tvec, self.intr)

    def set_distort(self, distort):
        self.distort = np.squeeze(distort)

    def set_focal_length(self, fx, fy):
        self.focal_length_x = fx
        self.focal_length_y = fy
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy
        self.P = Camera.calc_projection_matrix(self.R, self.tvec, self.intr)

    def set_alpha(self, alpha, r=94):
        self.set_eulerAngles(np.array([0.0, alpha, 0.0]))
        sign_x = 1 if 0 < alpha < math.pi else -1
        sign_z = (
            1
            if (math.pi / 2 < alpha < math.pi or -math.pi < alpha < -math.pi / 2)
            else -1
        )
        tvec = np.abs(np.array([r * np.sin(alpha), 0.0, -r * np.cos(alpha)]))
        tvec *= [sign_x, 0, sign_z]
        tvec = -1 * self.R.dot(tvec)
        self.set_tvec(tvec)

    def set_eulerAngles(self, ea):
        self.set_R(self.eulerAngles_to_R(ea))

    def get_euler_angles(self):
        return Camera.R_to_eulerAngles(self.R)

    #########################
    ###########    OPERATIONS
    #########################

    def is_empty(self):
        has_points2d = np.any(self.points2d != 0)
        has_images = self.get_image(img_id=0) is not None

        return not has_points2d and not has_images

    def __getitem__(self, key):
        return self.points2d[key].reshape(-1, 2)

    def project(self, points3d):
        if points3d.shape[0] != 3:
            points3d = points3d.transpose()
        op = points3d.transpose()  # opencv wants 3xn matrix
        points2d, _ = cv2.projectPoints(
            op, self.rvec, self.tvec, self.intr, self.distort
        )
        points2d = np.squeeze(points2d)
        if points2d.ndim == 1:
            points2d = points2d[np.newaxis, :]
        if points2d.shape[1] != 2:
            points2d = points2d.transpose()
        return points2d

    def reprojection_error(self, points3d, mask):
        assert points3d.shape[0] == self[mask].shape[0]
        points2d = self[mask]
        err_list = (
            points2d.reshape(-1, 2) - self.project(points3d).reshape(-1, 2)
        ).ravel()
        return np.mean(np.abs(err_list)), np.array(err_list)

    #########################
    #################    PLOT
    #########################

    def get_image(self, img_id, flip=False):

        if self.from_video:
            img = self.video_obj[img_id]
        else:
            from deepfly.os_util import constr_img_name

            img_name, img_name_pad = (
                constr_img_name(self.cam_id_read, img_id, pad=False),
                constr_img_name(self.cam_id_read, img_id, pad=True),
            )

            image_path = os.path.join(self.image_folder, "{}.jpg".format(img_name))
            image_pad_path = os.path.join(self.image_folder, "{}.jpg".format(img_name_pad))

            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
            elif os.path.isfile(image_pad_path):
                img = cv2.imread(image_pad_path)
            else:
                raise FileNotFoundError

            if img is None:
                print("Cannot find", self.cam_id, img_id)
                raise FileNotFoundError

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # remove A
        if flip:
            img = cv2.flip(img, 1)
        return img

    def get_points2d(self, img_id):
        if self.points2d is not None:
            pts = self[img_id]
        else:
            raise NotImplementedError

        return pts

    def plot_2d(
        self,
        img_id=0,
        pts=None,
        draw_joints=None,
        flip_points=False,
        img=None,
        colors=None,
        thickness=None,
        draw_limbs=None,
        flip_image=False,
        circle_color=None,
        zorder=None,
        r_list=None,
    ):
        if img is None:
            img = self.get_image(img_id, flip=flip_image)
        if pts is None and self.points2d is not None:
            pts = self.get_points2d(img_id)
        if pts is None:
            pts = np.zeros((config["skeleton"].num_joints, 2))
        if zorder is None:
            zorder = config["skeleton"].get_zorder(self.cam_id)
        if draw_joints is None:
            draw_joints = [
                j
                for j in range(config["skeleton"].num_joints)
                if config["skeleton"].camera_see_joint(self.cam_id, j)
            ]
        pts_tmp = pts.copy()
        if flip_points:
            pts_tmp[pts_tmp > config["image_shape"][0]] = config["image_shape"][0]
            pts_tmp[:, 0] = config["image_shape"][0] - pts_tmp[:, 0]
            pts_tmp[pts_tmp == config["image_shape"][0]] = 0
        pts_tmp = pts_tmp.astype(int)

        img = plot_drosophila_2d(
            pts=pts_tmp,
            draw_joints=draw_joints,
            img=img,
            colors=colors,
            thickness=thickness,
            draw_limbs=draw_limbs,
            circle_color=circle_color,
            zorder=zorder,
            r_list=r_list,
        )
        return img

    #########################
    #########################    STATIC
    #########################

    @staticmethod
    def hm_to_pred(
        hm,
        num_pred=1,
        scale=(1, 1),
        min_distance=1,
        threshold_abs=0.1,
        threshold_rel=None,
    ):
        pred = []
        if hm.ndim == 2:
            pred = skimage.feature.peak_local_max(
                hm,
                min_distance=min_distance,
                threshold_abs=threshold_abs,
                num_peaks=num_pred,
                exclude_border=False,
                threshold_rel=threshold_rel,
            )

            if len(pred) == 0:  # in case hm is zero array
                pred = np.array([0, 0])
            if num_pred == 1:
                pred = np.squeeze(pred)
            pred = pred / [hm.shape[0], hm.shape[1]]
            pred = np.array(pred)
            if pred.ndim == 1:
                pred = pred[np.newaxis, :]
            tmp = pred[:, 1].copy()
            pred[:, 1] = pred[:, 0]
            pred[:, 0] = tmp
            pred[:, 0] *= scale[0]
            pred[:, 1] *= scale[1]
            return pred
        else:
            for hm_ in hm:
                pred.append(
                    Camera.hm_to_pred(
                        hm_, min_distance=min_distance, threshold_abs=threshold_abs
                    )
                )
        return np.squeeze(pred) * scale

    @staticmethod
    def parse_img_name(img_name):
        return (int(img_name.split("_")[1]), int(img_name.split("_")[3]))

    @staticmethod
    def eulerAngles_to_R(theta):
        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(theta[0]), -math.sin(theta[0])],
                [0, math.sin(theta[0]), math.cos(theta[0])],
            ]
        )
        R_y = np.array(
            [
                [math.cos(theta[1]), 0, math.sin(theta[1])],
                [0, 1, 0],
                [-math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )
        R_z = np.array(
            [
                [math.cos(theta[2]), -math.sin(theta[2]), 0],
                [math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    @staticmethod
    def R_to_eulerAngles(R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    @staticmethod
    def calc_extr_from_Rt(R, tvec):
        extrinsic = np.zeros(shape=(3, 4))
        extrinsic[:3, :3] = R
        extrinsic[:, 3] = np.squeeze(tvec)
        return extrinsic

    @staticmethod
    def calc_projection_matrix(R, tvec, intr):
        extr = Camera.calc_extr_from_Rt(R, tvec)
        P = np.matmul(intr, extr)
        return P
