import json
import math
import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import skimage
import skimage.feature
from scipy.spatial import KDTree

from . import skeleton
from .plot_util import plot_drosophila_heatmap, plot_drosophila_2d


class Camera:
    def __init__(self, cid, image_folder, json_path=None, hm=None, points2d=None, num_images=1000, cid_read=None):
        self.cam_id = cid
        self.cam_id_read = cid_read if cid_read is not None else cid
        self.json_path = json_path
        self.image_folder = image_folder
        self.points2d = points2d  # pixel coordinates, not normalized
        if json_path is not None:  # to read annotations from
            self.__parse_json(json_path, num_images=num_images)
        self.hm = hm

        self.R = None
        self.rvec = None
        self.tvec = None
        """
        # Calculation for the focal length of the cameras
        sensor_width_mm_x = 11.3
        sensor_width_mm_y = 7.1
        image_width_in_pixels = 1920.0
        image_height_in_pixels = 1200.0
        focal_mm = 94.0

        # converting focal mm to focal pixels
        pixel_size_mm = 5.86 * (10 ** -3)  # (10**-3)
        f_pixel = (focal_mm / pixel_size_mm)
        print("Focal length guess in pixels {:.6}".format(f_pixel))
        print("Focal length guess in pixels {:.6}".format(
            (focal_mm / sensor_width_mm_x) * image_width_in_pixels))

        self.focal_length_x = (focal_mm / sensor_width_mm_x) * image_width_in_pixels
        self.focal_length_y = (focal_mm / sensor_width_mm_y) * image_height_in_pixels
        """
        self.image_shape = (480, 960)
        self.focal_length_x = 16041.0
        self.focal_length_y = 15971.7
        self.cx = 240
        self.cy = 480
        self.intr = np.array([[self.focal_length_x, 0, self.cx],
                              [0, self.focal_length_y, self.cy],
                              [0, 0, 1]], dtype=float)
        self.distort = np.zeros(5, dtype=np.float)
        self.P = None
        self.mask_unique = None

    def __parse_json(self, json_path, num_images=1000, num_joints=40):
        # holds the key(folder_name, image_name), annotation pair
        self.annotation_dict = dict()

        # process the json
        json_data = json.load(open(json_path, "r"))
        for session_id in json_data.keys():
            if self.folder_name in json_data[session_id]["data"]:
                for image_name in json_data[session_id]["data"][self.folder_name].keys():
                    key = (self.folder_name, image_name)
                    cid, pid = Camera.parse_img_name(image_name)
                    if cid == self.cam_id and "NaN" not in image_name:
                        if key not in self.annotation_dict.keys():
                            self.annotation_dict[key] = []
                        else:
                            continue

                        anot = json_data[session_id]["data"][self.folder_name][image_name]["position"]
                        self.annotation_dict[key] = np.array(anot)

        self.points2d = np.zeros(shape=(num_images, num_joints, 2))

        for k, v in self.annotation_dict.items():
            image_name = k[1]
            cam_id, img_id = Camera.parse_img_name(image_name)
            self.points2d[img_id, :, :] = v * [960, 480]  # un-normalize the coordinates
        self.points2d = np.abs(self.points2d)  # remove the confidence signs
        self.points2d = self.points2d.astype(np.int)

        print("Camera {}: data shape: {}".format(self.cam_id, self.points2d.shape))
        print("Camera {}: Loaded ~{} annotations".format(self.cam_id, np.sum(self.points2d != 0)))

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

    def set_alpha(self, alpha, r=94, inv=False):
        import math
        self.set_eulerAngles(np.array([0., alpha, 0.]))
        sign_x = 1 if 0 < alpha < math.pi else -1
        sign_z = 1 if (math.pi / 2 < alpha < math.pi or -math.pi < alpha < -math.pi / 2) else -1
        tvec = np.abs(np.array([r * np.sin(alpha), 0., -r * np.cos(alpha)]))
        tvec *= [sign_x, 0, sign_z]
        tvec = -1 * self.R.dot(tvec)
        self.set_tvec(tvec)

    def set_eulerAngles(self, ea):
        self.set_R(self.eulerAngles_to_R(ea))

    def get_euler_angles(self):
        return Camera.R_to_eulerAngles(self.R)

    '''
    OPERATIONS
    '''

    def __getitem__(self, key):
        return self.points2d[key].reshape(-1, 2)

    def calc_mask_unique(self, thr=3):
        m = np.zeros(shape=self.points2d.shape, dtype=np.bool)
        size_list = []
        for j in range(self.points2d.shape[1]):
            mask = np.zeros(self.points2d.shape, np.bool)
            mask[:, j, :] = True
            # mask = np.logical_and(mask, self.points2d != 0)
            if np.sum(mask) == 0:
                continue

            r, c, _ = np.nonzero(mask)
            r = r[::2]
            c = c[::2]

            pts = self[mask]
            kd_tree = KDTree(pts)
            res = kd_tree.query_pairs(r=thr, p=2)
            for (p1, p2) in res:
                if mask[r[p1], c[p1], 0] and mask[r[p2], c[p2], 0]:
                    mask[r[p2], c[p2], :] = False

            size_list.append(np.sum(mask) / 2.)
            m = np.logical_or(m, mask)
        self.mask_unique = m
        print("Camera {} after pruning: {}".format(self.cam_id, size_list))
        return m

    def intersect(self, cam, ignore_joint_list=[0]):
        assert (np.array_equal(self.points2d.shape, cam.points2d.shape))
        bool_intersect = np.logical_and(self.points2d != 0, cam.points2d != 0)
        for j in ignore_joint_list:
            bool_intersect[:, np.arange(j, bool_intersect.shape[1], 5), :] = False
        return bool_intersect

    def project(self, points3d):
        if points3d.shape[0] != 3:
            points3d = points3d.transpose()
        op = points3d.transpose()  # opencv wants 3xn matrix
        points2d, _ = cv2.projectPoints(op, self.rvec, self.tvec, self.intr, self.distort)
        points2d = np.squeeze(points2d)
        if points2d.ndim == 1:
            points2d = points2d[np.newaxis, :]
        if points2d.shape[1] != 2:
            points2d = points2d.transpose()
        return points2d

    def reprojection_error(self, points3d, mask):
        assert (points3d.shape[0] == self[mask].shape[0])
        points2d = self[mask]
        err_list = (points2d.reshape(-1, 2) - self.project(points3d).reshape(-1, 2)).ravel()
        return np.mean(np.abs(err_list)), np.array(err_list)

    '''
    PLOT
    '''

    def get_heatmap(self, img_id, j_id=None):
        if j_id is None:
            if self.cam_id == 3:
                j_id = list(range(skeleton.num_joints))
            else:
                j_id = list(range(skeleton.num_joints//2 ))
        if not isinstance(j_id, list):
            j_id = [j_id]
        if self.hm is None:
            print("Trying to read nonexisting heatmap")
            return np.zeros(shape=(len(j_id), 64,128), dtype=float)
        for j in j_id:
            if not skeleton.camera_see_joint(self.cam_id, j):
                print("Trying to read heatmap from camera {} point {}".format(self.cam_id, j))
        if self.cam_id > 3:
            j_id = [(j % (skeleton.num_joints // 2)) for j in j_id]
        if self.cam_id < 3 or self.cam_id > 3:
            if j_id is not None:
                return self.hm[self.cam_id_read, img_id, j_id, :]
            else:
                return self.hm[self.cam_id_read, img_id, :]
        elif self.cam_id == 3:
            cam3_j = [j for j in j_id if j < skeleton.num_joints // 2]
            cam7_j = [j % (skeleton.num_joints // 2) for j in j_id if j >= skeleton.num_joints // 2]
            cam3_hm = self.hm[self.cam_id_read, img_id, cam3_j, :, :]
            cam7_hm = self.hm[7, img_id, cam7_j, :, :]
            if cam3_j and cam7_j:
                return np.concatenate([cam3_hm, cam7_hm])
            elif cam3_j:
                return cam3_hm
            elif cam7_j:
                return cam7_hm
            else:
                raise NotImplementedError

    def get_image(self, img_id, flip=False):
        try:
            img = mpimg.imread(
                os.path.join(self.image_folder, "camera_{}_img_{:06}.jpg".format(self.cam_id_read, img_id)))
        except FileNotFoundError:
            img = mpimg.imread(os.path.join(self.image_folder, "camera_{}_img_{}.jpg".format(self.cam_id_read, img_id)))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if flip:
            img = cv2.flip(img, 1)
        return img

    def get_points2d(self, img_id):
        if self.points2d is not None:
            # get the points from self.points2d
            pts = self[img_id]
        elif np.all(pts == 0) and self.hm is not None:
            # get the points from self.hm
            print('points2d is None, getting points from the heatmap')
            pts = np.squeeze(np.array(Camera.hm_to_pred(self.hm[img_id, :]))) * [960, 480]
        else:
            raise NotImplementedError

        return pts

    def plot_2d(self, img_id=0, pts=None, draw_joints=None, flip_points=False, img=None, colors=None, thickness=None,
                draw_limbs=None, flip_image=False, circle_color=None, zorder=None):
        if img is None:
            img = self.get_image(img_id, flip=flip_image)
        if pts is None and self.points2d is not None:
            pts = self.get_points2d(img_id)
        if pts is None:
            pts = np.zeros((skeleton.num_joints, 2))
        if draw_joints is None:
            draw_joints = [j for j in range(skeleton.num_joints) if skeleton.camera_see_joint(self.cam_id, j)]
        pts_tmp = pts.copy()
        if flip_points:
            pts_tmp[pts_tmp > 960] = 960
            pts_tmp[:, 0] = 960 - pts_tmp[:, 0]
            pts_tmp[pts_tmp == 960] = 0
        pts_tmp = pts_tmp.astype(int)

        img = plot_drosophila_2d(pts=pts_tmp, draw_joints=draw_joints, img=img, colors=colors, thickness=thickness,
                                 draw_limbs=draw_limbs, circle_color=circle_color, zorder=zorder)
        return img

    def plot_heatmap(self, img_id, hm=None, img=None, concat=False, draw_joints=None, scale=1, flip_heatmap=False,
                     flip_image=False):
        '''
        concat: Whether to return a single image or njoints images concatenated
        '''
        if img is None:
            inp = self.get_image(img_id, flip=flip_image)
        else:
            inp = img
        if hm is None and self.hm is not None:
            hm = self.get_heatmap(img_id, draw_joints)  # self.hm[img_id, :, :, :]
        hm_tmp = hm.copy()
        if flip_heatmap:
            for i in range(hm.shape[0]):
                hm_tmp[i] = cv2.flip(hm[i], 1)
        return plot_drosophila_heatmap(inp, hm_tmp, concat=concat, scale=scale)

    '''
    STATIC
    '''

    @staticmethod
    def hm_to_pred(hm, num_pred=1, scale=(1, 1), min_distance=1, threshold_abs=0.1, threshold_rel=None):
        pred = []
        if hm.ndim == 2:
            pred = skimage.feature.peak_local_max(hm, min_distance=min_distance, threshold_abs=threshold_abs,
                                                  num_peaks=num_pred,
                                                  exclude_border=False, threshold_rel=threshold_rel)

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
                pred.append(Camera.hm_to_pred(hm_, min_distance=min_distance, threshold_abs=threshold_abs))
        return np.squeeze(pred) * scale

    @staticmethod
    def parse_img_name(img_name):
        return (int(img_name.split("_")[1]), int(img_name.split("_")[3]))

    @staticmethod
    def eulerAngles_to_R(theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
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
