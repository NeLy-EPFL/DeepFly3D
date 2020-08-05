import math  # inf
import os.path
import pickle
import re

import numpy as np
from sklearn.neighbors import NearestNeighbors

from deepfly import logger
from deepfly.belief_propagation import solve_belief_propagation
from deepfly.CameraNetwork import CameraNetwork
from deepfly.Config import config
from deepfly.DB import PoseDB
from deepfly.optim_util import energy_drosoph
from deepfly.os_util import (
    get_max_img_id,
    read_calib,
    read_camera_order,
    write_camera_order,
)
from deepfly.plot_util import normalize_pose_3d
from deepfly.pose2d import ArgParse
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.procrustes import procrustes_seperate
from deepfly.signal_util import filter_batch, smooth_pose2d


def find_default_camera_ordering(input_folder):
    """Uses regexes to infer the correct camera ordering based on folder path.

    This is useful for Ramdya's Lab as a given data acquisition agent (say CLC)
    always uses the same camera ordering.

    Parameters:
    input_folder: the folder path on which to run the regexes.
    """

    known_users = [
        (r"/CLC/", [0, 6, 5, 4, 3, 2, 1]),
        (r"data/test", [0, 1, 2, 3, 4, 5, 6]),
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


class Core:
    """Main interface to interact and use the 2d and 3d pose estimation network."""

    def __init__(self, input_folder, output_subfolder, num_images_max):
        self.input_folder = input_folder
        self.output_subfolder = output_subfolder
        self.output_folder = os.path.join(input_folder, output_subfolder)

        self.num_images_max = num_images_max or math.inf
        max_img_id = get_max_img_id(self.input_folder)
        self.num_images = min(self.num_images_max, max_img_id + 1)
        self.max_img_id = self.num_images - 1

        self.db = PoseDB(self.output_folder)
        self.setup_camera_ordering()
        self.set_cameras()
        self.check_cameras()

    # -------------------------------------------------------------------------
    # properties

    @property
    def input_folder(self):
        return self._input_folder

    @input_folder.setter
    def input_folder(self, value):
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

    def update_camera_ordering(self, cidread2cid):
        """Writes the new ordering inside the output folder and uses it.

        Parameters:
        cidread2cid: the new camera ordering to write and use.

        Returns:
        boolean: whether the ordering was successfuly updated.
        """

        if cidread2cid is None:
            return False

        if len(cidread2cid) != config["num_cameras"]:
            print(
                f"Cannot rename images as there are no {config['num_cameras']} values"
            )
            return False

        print("Camera order {}".format(cidread2cid))
        write_camera_order(self.output_folder, cidread2cid)
        self.cidread2cid, self.cid2cidread = read_camera_order(self.output_folder)
        self.camNetAll.set_cid2cidread(self.cid2cidread)
        return True

    def pose2d_estimation(self, overwrite=True):
        """Runs the pose2d estimation on self.input_folder.

        Parameters:
        overwrite: whether to overwrite existing pose estimation results (default: True)
        """

        parser = ArgParse.create_parser()
        args, _ = parser.parse_known_args()
        args.checkpoint = False
        args.unlabeled = self.input_folder
        args.output_folder = self.output_subfolder
        args.resume = config["resume"]
        args.stacks = config["num_stacks"]
        args.test_batch = config["batch_size"]
        args.img_res = [config["heatmap_shape"][0] * 4, config["heatmap_shape"][1] * 4]
        args.hm_res = config["heatmap_shape"]
        args.num_classes = config["num_predict"]
        args.max_img_id = self.max_img_id
        args.overwrite = overwrite

        pose2d_main(args)  # will write output files in output directory
        self.set_cameras()  # makes sure cameras use the latest heatmaps and predictions

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

        print(f"Calibration considering frames between {min_img_id}:{max_img_id}")
        calib = read_calib(config["calib_fine"])
        assert calib is not None
        self.camNetAll.load_network(calib)

        # print(self.camNetAll.cam_list[0].points2d.shape, "awawd")

        # take a copy of the current points2d
        pts2d = np.zeros(
            (config["num_cameras"], self.num_images, config["skeleton"].num_joints, 2),
            dtype=float,
        )
        for cam_id in range(config["num_cameras"]):
            pts2d[cam_id, :] = self.camNetAll.cam_list[cam_id].points2d.copy()

        # ugly hack to temporarly incorporate manual corrections to calibration
        c = 0
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.num_images):
                if self.db.has_key(cam_id, img_id):
                    pt = self.corrected_points2d(cam_id, img_id)
                    self.camNetAll.cam_list[cam_id].points2d[img_id, :] = pt
                    c += 1
        print(f"Calibration: replaced {c} with manual corrections")

        # keep the pts only in the range
        for cam in self.camNetAll.cam_list:
            cam.points2d = cam.points2d[min_img_id:max_img_id, :]

        self.camNetLeft.triangulate()
        self.camNetLeft.bundle_adjust(cam_id_list=(0, 1, 2))
        self.camNetRight.triangulate()
        self.camNetRight.bundle_adjust(cam_id_list=(0, 1, 2))

        # put old values back
        for cam_id in range(config["num_cameras"]):
            self.camNetAll.cam_list[cam_id].points2d = pts2d[cam_id, :].copy()

        self.save_calibration()
        self.set_cameras()

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

    def solve_bp(self, img_id):
        """Solves the belief propagation on image specified by img_id."""
        if self.has_calibration and self.has_pose:
            self.solve_bp_for_camnet(img_id, self.camNetLeft)
            self.solve_bp_for_camnet(img_id, self.camNetRight)

    def smooth_points2d(self, cam_id, private_cache=dict()):
        """Gets the smoothened points2d of cam_id.

        Parameters:
        cam_id: the camera id from which to get the points2d
        private_cache: private argument used as a singleton instance to store a cache.
        """
        if cam_id not in private_cache:
            cam = self.camNetAll.cam_list[cam_id]
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

        cam = self.camNetAll.cam_list[cam_id]
        joints = joints if joints else range(config["skeleton"].num_joints)
        visible = lambda j_id: config["skeleton"].camera_see_joint(cam_id, j_id)
        visible_joints = [j_id for j_id in joints if visible(j_id)]
        zorder = config["skeleton"].get_zorder(cam_id)
        corrected = self.db.has_key(cam_id, img_id)

        def compute_r_list(img_id):
            r_list = [config["scatter_r"]] * config["num_joints"]
            for joint_id in range(config["skeleton"].num_joints):
                if joint_id not in config["skeleton"].pictorial_joint_list:
                    continue
                if self.joint_has_error(img_id, joint_id):
                    r_list[joint_id] = config["scatter_r"] * 2
            return r_list

        if with_corrections:
            pts2d = self.corrected_points2d(cam_id, img_id)
            r_list = compute_r_list(img_id)
            circle_color = (0, 255, 0) if corrected else (0, 0, 255)
        else:
            pts2d = self.smooth_points2d(cam_id) if smooth else cam.points2d
            pts2d = pts2d[img_id]
            r_list = None
            circle_color = None

        return cam.plot_2d(
            img_id,
            pts2d,
            circle_color=circle_color,
            draw_joints=visible_joints,
            zorder=zorder,
            r_list=r_list,
        )

    def plot_heatmap(self, cam_id, img_id, joints=[]):
        """Plots the probability map for joint positions.

        Parameters:
        cam_id: id of the camera from which to take the image
        img_id: id of the image to plot
        joints: ids of the joints to plot, use empty list for all joints (default: [])

        Returns:
        an image as an np.array with the probability map represented as heatmap.
        """
        cam = self.camNetAll.cam_list[cam_id]
        joints = joints if joints else range(config["skeleton"].num_joints)
        visible = lambda j_id: config["skeleton"].camera_see_joint(cam_id, j_id)
        visible_joints = [j_id for j_id in joints if visible(j_id)]
        return cam.plot_heatmap(
            img_id, concat=False, scale=2, draw_joints=visible_joints
        )

    def get_image(self, cam_id, img_id):
        """Returns the img_id image from cam_id camera."""
        return self.camNetAll.cam_list[cam_id].get_image(img_id)

    def get_points3d(self):
        """Returns a numpy array with 3d positions of the joints.

        Indexing is as follows:
        array[image_id][joint_id] = (x, y, z)
        """
        camNetL = self.camNetLeft
        camNetR = self.camNetRight

        camNetL.triangulate()
        camNetL.bundle_adjust(cam_id_list=(0, 1, 2))

        camNetR.triangulate()
        camNetR.bundle_adjust(cam_id_list=(0, 1, 2))

        self.camNetAll.triangulate()
        points3d_m = self.camNetAll.points3d_m.copy()
        points3d_m = procrustes_seperate(points3d_m)
        points3d_m = normalize_pose_3d(points3d_m, rotate=True)
        points3d_m = filter_batch(points3d_m)
        return points3d_m

    def save_corrections(self):
        """Writes the manual corrections to a file in the output folder."""
        self.db.dump()

    def post_process(self, points2d_matrix):
        """Runs some hardcoded post-processing on the pose estimation results."""
        pts2d = points2d_matrix
        if "fly" in config["name"]:
            # some post-processing for body-coxa
            for cam_id in range(len(self.camNetAll.cam_list)):
                for j in range(config["skeleton"].num_joints):
                    if config["skeleton"].camera_see_joint(cam_id, j) and config[
                        "skeleton"
                    ].is_tracked_point(j, config["skeleton"].Tracked.BODY_COXA):
                        pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                        pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])

    def save_pose(self):
        """Saves the pose estimation results to a file in the output folder."""
        pts2d = self.corrected_points2d_matrix()
        dict_merge = self.camNetAll.save_network(path=None)
        pts2d_orig = self.camNetAll.get_points2d_matrix()

        # temporarily incorporate corrected values
        self.camNetAll.set_points2d_matrix(pts2d)
        self.post_process(pts2d)
        dict_merge["points2d"] = pts2d

        if self.camNetLeft.has_calibration() and self.camNetLeft.has_pose():
            self.camNetAll.triangulate()
            pts3d = self.camNetAll.points3d_m
            if config["procrustes_apply"]:
                print("Applying Procrustes on 3D Points")
                pts3d = procrustes_seperate(pts3d)
            dict_merge["points3d"] = pts3d
        else:
            logger.debug("Triangulation skipped.")

        # put uncorrected values back
        self.camNetAll.set_points2d_matrix(pts2d_orig)

        save_path = os.path.join(
            self.output_folder,
            "pose_result_{}.pkl".format(self.input_folder.replace("/", "_")),
        )
        pickle.dump(dict_merge, open(save_path, "wb"))
        print(f"Saved the pose at: {save_path}")

    # -------------------------------------------------------------------------
    # private helper methods

    def save_calibration(self):
        """Saves the calibration results to a file in the output folder."""
        filename = "calib.pkl"  # f"calib_{self.input_folder.replace('/', '_')}.pkl"
        calib_path = os.path.join(self.output_folder, filename)
        print("Saving calibration {}".format(calib_path))
        self.camNetAll.save_network(calib_path)

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

    def setup_camera_ordering(self):
        """Reads camera ordering from file or attempts to use a default ordering instead."""
        default = find_default_camera_ordering(self.input_folder)
        if default is not None:  # np.arrays don't evaluate to bool
            write_camera_order(self.output_folder, default)
        self.cidread2cid, self.cid2cidread = read_camera_order(self.output_folder)

    def set_cameras(self):
        """Creates the camera network instances using the latest calibration files."""
        calib = read_calib(self.output_folder)
        self.camNetAll = CameraNetwork(
            image_folder=self.input_folder,
            output_folder=self.output_folder,
            cam_id_list=range(config["num_cameras"]),
            cid2cidread=self.cid2cidread,
            num_images=self.num_images,
            calibration=calib,
        )
        self.camNetLeft = CameraNetwork(
            image_folder=self.input_folder,
            output_folder=self.output_folder,
            cam_id_list=config["left_cameras"],
            num_images=self.num_images,
            calibration=calib,
            cid2cidread=[self.cid2cidread[cid] for cid in config["left_cameras"]],
            cam_list=[
                cam
                for cam in self.camNetAll.cam_list
                if cam.cam_id in config["left_cameras"]
            ],
        )
        self.camNetRight = CameraNetwork(
            image_folder=self.input_folder,
            output_folder=self.output_folder,
            cam_id_list=config["right_cameras"],
            num_images=self.num_images,
            calibration=calib,
            cid2cidread=[self.cid2cidread[cid] for cid in config["right_cameras"]],
            cam_list=[
                self.camNetAll.cam_list[cam_id] for cam_id in config["right_cameras"]
            ],
        )

        self.camNetLeft.bone_param = config["bone_param"]
        self.camNetRight.bone_param = config["bone_param"]
        calib = read_calib(config["calib_fine"])
        self.camNetAll.load_network(calib)

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

    def get_joint_reprojection_error(self, img_id, joint_id, camNet):
        """Computes the joint_reprojection_error for joint_id on img_id"""

        visible_cameras = [
            cam
            for cam in camNet.cam_list
            if config["skeleton"].camera_see_joint(cam.cam_id, joint_id)
        ]
        if len(visible_cameras) < 2:
            err_proj = 0
        else:
            pts = np.array(
                [cam.points2d[img_id, joint_id, :] for cam in visible_cameras]
            )
            _, err_proj, _, _ = energy_drosoph(
                visible_cameras, img_id, joint_id, pts / [960, 480]
            )
        return err_proj

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

    def solve_bp_for_camnet(self, img_id, camNet):
        """Solves the belief propagation on the provided camera network."""
        # Compute prior
        prior = []
        manual_corrections = self.db.manual_corrections()
        for cam in camNet.cam_list:
            if img_id in manual_corrections.get(cam.cam_id, {}):
                for joint_id in range(manual_corrections[cam.cam_id][img_id].shape[0]):
                    pt2d = manual_corrections[cam.cam_id][img_id][joint_id]
                    prior.append((cam.cam_id, joint_id, pt2d / config["image_shape"]))

        pts_bp = solve_belief_propagation(
            camNet.cam_list, img_id, config["bone_param"], prior=prior
        )
        pts_bp = np.array(pts_bp)

        for idx, cam in enumerate(camNet.cam_list):
            # set points which are not estimated by bp
            pts_bp_rep = self.db.read(cam.cam_id, img_id)
            if pts_bp_rep is not None:
                pts_bp_rep *= self.image_shape
            else:
                pts_bp_rep = cam.points2d[img_id, :]

            pts_bp_ip = pts_bp[idx] * self.image_shape
            pts_bp_ip[pts_bp_ip == 0] = pts_bp_rep[pts_bp_ip == 0]

            # save corrections
            modified_joints = self.db.read_modified_joints(cam.cam_id, img_id)
            self.write_corrections(cam.cam_id, img_id, modified_joints, pts_bp_ip)

        print("Finished Belief Propagation")

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
