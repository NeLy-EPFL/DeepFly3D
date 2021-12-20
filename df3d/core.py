import glob
import math  # inf
import os.path
import pickle
import re
import subprocess
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from df2d.inference import inference_folder
from pyba.CameraNetwork import CameraNetwork
from sklearn.neighbors import NearestNeighbors

from df3d import logger
from df3d.config import config
from df3d.db import PoseDB
from df3d.os_util import get_max_img_id, parse_vid_name
from df3d.plot_util import normalize_pose_3d
from df3d.procrustes import procrustes_seperate
from df3d.signal_util import filter_batch, smooth_pose2d


def find_default_camera_ordering(input_folder: str):
    """Uses regexes to infer the correct camera ordering based on folder path.

    This is useful for Ramdya's Lab as a given data acquisition agent (say CLC)
    always uses the same camera ordering.

    Parameters:
    input_folder: the folder path on which to run the regexes.
    """

    known_users = [
        (r"/CLC/", [0, 6, 5, 4, 3, 2, 1]),
        (r"/FA/", [6, 5, 4, 3, 2, 1, 0]),
        (r"/SG/", [6, 5, 4, 3, 2, 1, 0]),
        (r"Laura", [0, 6, 5, 4, 3, 2, 1]),
        (r"AYMANNS_Florian", [6, 5, 4, 3, 2, 1, 0]),
        (r"data/test", [0, 1, 2, 3, 4, 5, 6]),
        (r"/JB/", [6, 5, 4, 3, 2, 1, 0]),
    ]
    #
    input_folder = str(input_folder)  # use `str` in case pathlib.Path instance

    def match(regex):
        return re.search(regex, input_folder)

    candidates = [order for (regex, order) in known_users if match(regex)]
    if candidates:
        order = candidates[0]
        logger.debug(f"Default camera ordering found: {order}")
        return np.array(order)
    else:
        raise NotImplementedError("Cannot find camera ordering ")


class Core:
    """Main interface to interact and use the 2d and 3d pose estimation network."""

    def __init__(
        self,
        input_folder: str,
        output_subfolder: str,
        num_images_max: int,
        camera_ordering: List[int],
    ):
        self.input_folder = input_folder
        self.output_subfolder = output_subfolder
        self.output_folder = os.path.join(input_folder, output_subfolder)

        self.expand_videos()  # turn .mp4 into .jpg
        self.num_images_max = num_images_max or math.inf
        max_img_id = get_max_img_id(self.input_folder)
        self.num_images = min(self.num_images_max, max_img_id + 1)
        self.max_img_id = self.num_images - 1

        self.db = PoseDB(self.output_folder)
        self.camera_ordering = self.setup_camera_ordering(camera_ordering)

        self.camNet = None
        self.points2d = None
        self.points3d = None
        # if already ran before, initiliaze with df3d_result file
        if os.path.exists(self.save_path):
            from pyba.config import df3d_bones, df3d_colors

            df3d_result = pickle.load(open(self.save_path, "rb"))
            image_path = image_path = os.path.join(
                self.input_folder, "camera_{cam_id}_img_{img_id}.jpg"
            )
            self.points2d = df3d_result["points2d"]
            if 'points3d' in df3d_result:
                self.points3d = df3d_result["points3d"]
            self.camNet = CameraNetwork(
                df3d_result["points2d"],
                calib=df3d_result,
                num_images=self.num_images,
                image_path=image_path,
                colors=df3d_colors,
                bones=df3d_bones,
            )

    # -------------------------------------------------------------------------
    # properties

    @property
    def input_folder(self):
        return self._input_folder

    @input_folder.setter
    def input_folder(self, value: str):
        value = os.path.abspath(value)
        value = value.rstrip("/")
        assert os.path.isdir(value), f"Not a directory {value}"
        self._input_folder = value

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        os.makedirs(value, exist_ok=True)
        value = os.path.abspath(value)
        value = value.rstrip("/")
        assert os.path.isdir(value), f"Not a directory {value}"
        self._output_folder = value

    @property
    def image_shape(self):
        return config["image_shape"]

    @property
    def number_of_joints(self):
        return config["skeleton"].num_joints

    @property
    def has_pose(self):
        return self.camNetLeft.has_pose() and self.camNetRight.has_pose()

    @property
    def has_heatmap(self):
        return self.camNetLeft.has_heatmap() and self.camNetRight.has_heatmap()

    @property
    def has_calibration(self):
        return self.camNetLeft.has_calibration() and self.camNetRight.has_calibration()

    # -------------------------------------------------------------------------
    # public methods

    def pose2d_estimation(self, overwrite=True):
        """Runs the pose2d estimation on self.input_folder.

        Parameters:
        overwrite: whether to overwrite existing pose estimation results (default: True)
        """

        # to make sure we rotate the necessary cameras
        class load_f:
            def __init__(self, cam_order):
                self.cam_order = cam_order.tolist()

            def parse_img_path(self, name: str) -> Tuple[int, int]:
                """returns cid and img_id """
                name = os.path.basename(name)
                match = re.match(r"camera_(\d+)_img_(\d+)", name.replace(".jpg", ""))
                return int(match[1]), int(match[2])

            def __call__(self, x):
                img = plt.imread(x)
                cam_id, _ = self.parse_img_path(x)
                if self.cam_order.index(cam_id) > 3:
                    img = img[:, ::-1]
                return img

        self.points2d = inference_folder(
            folder=self.input_folder,
            load_f=load_f(self.camera_ordering),
            return_heatmap=False,
            max_img_id=self.max_img_id,
        )

        # fmt: off
        # 2d pose estimation outputs 19 points, which is what a single camera sees,
        #     however there are 38 joints in total
        points2d_cp = np.zeros((self.points2d.shape[0], self.points2d.shape[1], self.points2d.shape[2]*2, 2))
        points2d_cp[self.camera_ordering[:3], :, :19] = self.points2d[self.camera_ordering[:3]]
        points2d_cp[self.camera_ordering[4:], :, 19:] = self.points2d[self.camera_ordering[4:]]

        # (x,y) are switched in pyba coordinate system, so we need to swap last two axes
        tmp = np.copy(points2d_cp[..., 0])
        points2d_cp[..., 0] = points2d_cp[..., 1]
        points2d_cp[..., 1] = tmp

        # cameras 2 and 4 cannot see the stripes
        points2d_cp[self.camera_ordering[2], :, 15:] = 0 
        points2d_cp[self.camera_ordering[4], :, 19+15:] = 0

        # flip lr back left-hand-side cameras
        for cidx in [4,5,6]:
            points2d_cp[self.camera_ordering[cidx], ..., 0] = 1 - points2d_cp[self.camera_ordering[cidx], ..., 0]
        #fmt:off
        self.points2d = points2d_cp

    def next_error(self, img_id):
        """Finds the next image with an error in prediction after img_id.

        Parameters:
        img_id: a valid image id after which to search for an error.

        Returns:
        int: None or the id of an image with an error in prediction.
        """

        return self.next_error_in_range(range(img_id + 1, self.max_img_id + 1))

    def prev_error(self, img_id):
        """Finds the previous image with an error in prediction before img_id.

        Parameters:
        img_id: a valid image id before which to search for an error.

        Returns:
        int: None or the id of an image with an error in prediction.
        """

        return self.next_error_in_range(range(img_id - 1, -1, -1))

    def calibrate_calc(self, min_img_id, max_img_id):
        """Calibrates and saves the results in the output folder.

        Uses the images between min_img_id and max_img_id for the calibration.
        """
        calib_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "../data/calib.pkl"
        )

        calib = pickle.load(open(calib_path, "rb"))
        calib_reordered = {
            cidx: calib[idx] for (idx, cidx) in enumerate(self.camera_ordering)
        }

        image_path = os.path.join(self.input_folder, "camera_{cam_id}_img_{img_id}.jpg")
        self.camNet = CameraNetwork(
            self.points2d * [960, 480], calib=calib_reordered, image_path=image_path
        )
        self.camNet.bundle_adjust(update_intrinsic=False, update_distort=False)

    def nearest_joint(self, cam_id, img_id, x, y):
        """Finds the joint nearest to (x,y) coordinates on the img_id of cam_id.

        Parameters:
        cam_id: the id of the camera from which the image is taken
        img_id: the id of an image on which to look for a joint
        x: abscissa of the point from which we want the nearest joint
        y: coordinate of the point from which we want the nearest joint

        Returns:
        (x,y): the coordinates of the joint nearest to (x,y)
        """

        joints = range(config["skeleton"].num_joints)
        visible = lambda j_id: config["skeleton"].camera_see_joint(cam_id, j_id)
        unvisible_joints = [j_id for j_id in joints if not visible(j_id)]

        pts = self.corrected_points2d(cam_id, img_id)
        pts[unvisible_joints] = [9999, 9999]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(pts)
        _, indices = nbrs.kneighbors(np.array([[x, y]]))
        return indices[0][0]

    def move_joint(self, cam_id, img_id, joint_id, x, y):
        """Moves the joint specified by joint_id to position (x,y)."""

        modified_joints = self.db.read_modified_joints(cam_id, img_id)
        modified_joints = list(sorted(set(modified_joints + [joint_id])))
        points = self.corrected_points2d(cam_id, img_id)
        points[joint_id] = np.array([x, y])
        self.write_corrections(cam_id, img_id, modified_joints, points)

    def smooth_points2d(self, cam_id, private_cache=dict()):
        """Gets the smoothened points2d of cam_id.

        Parameters:
        cam_id: the camera id from which to get the points2d
        private_cache: private argument used as a singleton instance to store a cache.
        """
        if cam_id not in private_cache:
            cam = self.camNet.cam_list[cam_id]
            private_cache[cam_id] = smooth_pose2d(cam.points2d)
        return private_cache[cam_id]

    def plot_2d(self, cam_id, img_id, with_corrections=False, smooth=False, joints=[]):
        """Plots the 2d pose estimation results.

        Parameters:
        cam_id: id of the camera from which to take the image
        img_id: id of the image to plot
        with_corrections: whether to plot manually corrected joints positions (default: False)
        smooth: whether to smoothen the joints positions for nicer videos (default: False)
        joints: ids of the joints to plot, use empty list for all joints (default: [])

        Returns:
        an image as an np.array with the plot.
        """

        cam = self.camNet[cam_id]

        from pyba.config import df3d_bones, df3d_colors

        return cam.plot_2d(img_id, bones=df3d_bones, colors=df3d_colors)

    def get_image(self, cam_id, img_id):
        """Returns the img_id image from cam_id camera."""
        return self.camNet.cam_list[cam_id].get_image(img_id)

    @property
    def save_path(self):
        return os.path.join(
            self.output_folder,
            "df3d_result_{}.pkl".format(self.input_folder.replace("/", "_")),
        )

    def get_points3d(self):
        """Returns a numpy array with 3d positions of the joints.

        Indexing is as follows:
        array[image_id][joint_id] = (x, y, z)
        """
        camNetL = self.camNetLeft
        camNetR = self.camNetRight

        camNetL.triangulate()
        camNetL.calibrate(cam_id_list=(0, 1, 2))

        camNetR.triangulate()
        camNetR.calibrate(cam_id_list=(0, 1, 2))

        self.camNet.triangulate()
        points3d = np.copy(self.camNet.points3d)
        points3d = procrustes_seperate(points3d)
        points3d = normalize_pose_3d(points3d, rotate=True)
        points3d = filter_batch(points3d)
        return points3d

    def save_corrections(self):
        """Writes the manual corrections to a file in the output folder."""
        self.db.dump()

    '''
    def post_process(self, points2d_matrix):
        """Runs some hardcoded post-processing on the pose estimation results."""
        pts2d = points2d_matrix
        if "fly" in config["name"]:
            # some post-processing for body-coxa
            for cam_id in range(pts2d.shape[0]):
                for j in range(config["skeleton"].num_joints):
                    if config["skeleton"].camera_see_joint(cam_id, j) and config[
                        "skeleton"
                    ].is_tracked_point(j, config["skeleton"].Tracked.BODY_COXA):
                        pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                        pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])
    '''

    def save(self):
        """Saves the pose estimation results to a file in the output folder."""
        dict_merge = dict()
        dict_merge["points2d"] = np.copy(self.points2d)

        # temporarily incorporate corrected values
        # points2d = np.copy(self.points2d)
        # self.post_process(points2d)
        # dict_merge["points2d_processed"] = points2d

        if self.camNet is not None and self.camNet.has_calibration():
            self.camNet.triangulate()
            pts3d = self.camNet.points3d
            dict_merge["points3d_wo_procrustes"] = pts3d
            pts3d = procrustes_seperate(pts3d)
            dict_merge["points3d"] = pts3d
            dict_merge = {**self.camNet.summarize(), **dict_merge}
        else:
            logger.debug("Triangulation skipped.")

        dict_merge["camera_ordering"] = self.camera_ordering

        pickle.dump(dict_merge, open(self.save_path, "wb"))
        print(f"Saved results at: {self.save_path}")

    # -------------------------------------------------------------------------
    # private helper methods

    def corrected_points2d(self, cam_id, img_id):
        """Gets the estimated or manually corrected 2d position of the joints.

        Returns:
        An array with the position of the joints on img_id from cam_id.
        """

        points2d = self.camNetAll.cam_list[cam_id].get_points2d(img_id).copy()
        manual_corrections = self.db.manual_corrections()
        if img_id in manual_corrections.get(cam_id, {}):
            points2d[:] = manual_corrections[cam_id][img_id]
        return points2d

    def corrected_points2d_matrix(self):
        """Gets the estimated or manually corrected 2d positions of the joints.

        Returns:
        An array with the positions of the joints for each cam_id, img_id.
        Indexing is as follows: results[cam_id][img_id][joint_id] = (x,y)
        """

        manual_corrections = self.db.manual_corrections()
        pts2d = self.camNetAll.get_points2d_matrix()
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.num_images):
                if img_id in manual_corrections.get(cam_id, {}):
                    pts2d[cam_id, img_id, :] = manual_corrections[cam_id][img_id]
        return pts2d

    def setup_camera_ordering(self, camera_ordering) -> np.ndarray:
        """Reads camera ordering from file or attempts to use a default ordering instead."""

        # if camera ordering preference is not given, then check the default matching
        camera_ordering = (
            find_default_camera_ordering(self.input_folder)
            if camera_ordering is None
            else camera_ordering
        )

        # self.cidread2cid, self.cid2cidread = read_camera_order(self.output_folder)
        return np.array(camera_ordering)

    def expand_videos(self):
        """ expands video camera_x.mp4 into set of images camera_x_img_y.jpg"""
        for vid in glob.glob(os.path.join(self.input_folder, "camera_*.mp4")):
            cam_id = parse_vid_name(os.path.basename(vid))
            if not (
                os.path.exists(
                    os.path.join(self.input_folder, f"camera_{cam_id}_img_0.jpg")
                )
                or os.path.exists(
                    os.path.join(self.input_folder, f"camera_{cam_id}_img_000000.jpg")
                )
            ):
                command = f"ffmpeg -i {vid} -qscale:v 2 -start_number 0 {self.input_folder}/camera_{cam_id}_img_%d.jpg  < /dev/null"
                subprocess.call(command, shell=True)

    def check_cameras(self):
        cam_missing = [cam.cam_id for cam in self.camNetAll.cam_list if cam.is_empty()]
        assert not cam_missing, "Some cameras are missing: {}".format(cam_missing)

    def next_error_in_range(self, range_of_ids):
        """Finds the first image in range_of_ids on which there is an estimation error.

        Returns:
        An image id with a suspected pose estimation error or None if none found.
        """
        all_joints = range(config["skeleton"].num_joints)
        pictorial = config["skeleton"].pictorial_joint_list
        joints = [j for j in all_joints if j in pictorial]
        for img_id in range_of_ids:
            for joint_id in joints:
                if self.joint_has_error(img_id, joint_id):
                    return img_id
        return None

    def joint_has_error(self, img_id, joint_id):
        """Indicates whether joint_id was estimated with error or not.

        Returns:
        boolean: whether there is a suspected error for joint_id on img_id.
        """

        get_error = self.get_joint_reprojection_error
        err_left = get_error(img_id, joint_id, self.camNetLeft)
        err_right = get_error(img_id, joint_id, self.camNetRight)
        err = max(err_left, err_right)
        return err > config["reproj_thr"][joint_id]

    def write_corrections(self, cam_id, img_id, modified_joints, points2d):
        """Saves the provided manual corrections to a file in the output_folder.

        Only the corrections which differ sufficiently from the original
        pose estimation results are saved.

        Parameters:
        cam_id: id of the camera from which to take the image
        img_id: id of the image on which the corrections are made
        modified_joints: list of joints that have been corrected
        points2d: array of the (x,y) location of *all* the joints on img_id.
        """

        l1_threshold = 30
        original_points2d = self.camNetAll.cam_list[cam_id].get_points2d(img_id)
        l1_error = np.abs(original_points2d - points2d)
        joints_to_check = [
            j
            for j in range(config["num_joints"])
            if (j not in config["skeleton"].ignore_joint_id)
            and config["skeleton"].camera_see_joint(cam_id, j)
        ]
        unseen_joints = [
            j
            for j in range(config["skeleton"].num_joints)
            if not config["skeleton"].camera_see_joint(cam_id, j)
        ]
        if np.any(l1_error[joints_to_check] > l1_threshold):
            points2d = points2d.copy()
            points2d[unseen_joints, :] = 0.0
            points2d = points2d / self.image_shape
            self.db.write(points2d, cam_id, img_id, True, modified_joints)
        else:
            # the corrections are too similar to original predicted points,
            # erase previous corrections
            self.db.remove_corrections(cam_id, img_id)
