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
from df3d.body_align import align_to_body_axes
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
        (r"sample/test", [0, 1, 2, 3, 4, 5, 6]),
        (r"/JB/", [6, 5, 4, 3, 2, 1, 0]),
    ]

    input_folder = str(input_folder)  # use `str` in case pathlib.Path instance

    def match(regex):
        return re.search(regex, input_folder)

    candidates = [order for (regex, order) in known_users if match(regex)]
    if candidates:
        order = candidates[0]
        logger.debug(f"Default camera ordering found: {order}")
        return np.array(order)
    else:
        raise NotImplementedError(
            f"Cannot find camera ordering for folder {input_folder}. Please"
            " set your camera ordering using the --order flag. Example usage"
            " is df3d-cli /your/path/images/ --order 0 1 2 3 4 5 6"
        )


class Core:
    """Main interface to interact and use the 2d and 3d pose estimation network."""

    def __init__(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
        num_images_max: Optional[int] = None,
        camera_ordering: List[int] = [0, 1, 2, 3, 4, 5, 6],
        start_image_idx: int = 0,
    ):
        self.input_folder = input_folder
        if output_folder is None:
            self.output_folder = self.input_folder + "_df3d"
        else:
            self.output_folder = output_folder

        self.expand_videos()  # turn .mp4/.avi into .jpg
        self.fps = self.get_fps()
        self.num_images_max = num_images_max if num_images_max is not None else 0
        self.max_img_id = get_max_img_id(self.input_folder)
        self.start_image_idx = start_image_idx
        if self.num_images_max > 0:
            self.num_images = min(self.num_images_max, self.max_img_id + 1 - self.start_image_idx)
            self.max_img_id = self.start_image_idx + self.num_images - 1
        else:
            self.num_images = self.max_img_id + 1 - self.start_image_idx
        image_path = os.path.join(self.input_folder, "camera_{cam_id}_img_{img_id}.jpg")
        image0_path = image_path.format(cam_id=0, img_id=0)
        if "image_shape" in config:
            self.image_shape = config["image_shape"]
        if os.path.exists(image0_path):
            image0 = plt.imread(image0_path)
            image0_shape = list(image0.shape[:2][::-1])
            if "image_shape" in config and image0_shape != self.image_shape:
                raise ValueError(f"Actual image shape {image0_shape} does not match"
                                 f" config.py image shape {self.image_shape}")
            self.image_shape = config["image_shape"] = image0_shape
        if not hasattr(self, "image_shape"):
            raise ValueError("Image shape not specified in df3d.config and could"
                             f" not be read from {image0_path}")

        self.db = PoseDB(self.output_folder)
        self.camera_ordering = self.setup_camera_ordering(camera_ordering)

        self.camNet = None
        self.points2d = None
        self.points3d = None
        # if already ran before, initiliaze with df3d_result file
        if os.path.exists(self.save_path):
            from pyba.config import df3d_bones, df3d_colors

            with open(self.save_path, "rb") as f:
                df3d_result = pickle.load(f)
            self.points2d = df3d_result["points2d"]
            self.conf = df3d_result["heatmap_confidence"]

            if "points3d" in df3d_result:
                self.points3d = df3d_result["points3d"]

            self.camNet = CameraNetwork(
                df3d_result["points2d"] * self.image_shape[::-1],
                calib=df3d_result,
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
    def number_of_joints(self):
        return config["skeleton"].num_joints

    @property
    def has_pose(self):
        return True
        # return self.camNet.has_pose()

    @property
    def has_calibration(self):
        return self.camNet.has_calibration()

    # -------------------------------------------------------------------------
    # public methods

    def pose2d_estimation(self, batch_size: int = 8, disable_pin_memory: bool = False,
                          save_top_k_peaks: bool = False,
                          keep_heatmaps: bool = False):
        """Runs the pose2d estimation on self.input_folder.

        Parameters:
        batch_size: Batch size to use when running inference on the images (default: 8)
        disable_pin_memory: Whether to disable the `pin_memory` option for the dataloader (default: False)
        save_top_k_peaks: If True, also extract the top-K local-maximum peaks per
            heatmap and store them as `self.top_k_peaks` (shape
            [n_cameras, n_frames, 38, K, 3]) for later use by the pictorial-
            structures / belief-propagation pose-correction step. K, the local-
            max search radius, and the relative threshold come from config keys
            `num_peak`, plus internal defaults. Default: False.
        keep_heatmaps: If True, retain the full per-joint heatmaps from the 2D
            network in `self._heatmaps` (shape [n_cameras, n_frames, 38, H, W],
            ~8.7 GB / 1000 frames) so that a subsequent call to
            `run_belief_propagation` can score arbitrary 2D candidates against
            them. Heatmaps are not written to the result pkl. Default: False.
        """
        inference_kwargs = dict(
            folder=self.input_folder,
            camera_ids_to_flip=[camera_id for index, camera_id in enumerate(self.camera_ordering) if index > 3], # flip the last 3 cameras so all images face to the right
            return_heatmap=keep_heatmaps,
            return_confidence=True,
            return_peaks=save_top_k_peaks,
            max_img_id=self.max_img_id,
            batch_size=batch_size,
            disable_pin_memory=disable_pin_memory,
        )
        if save_top_k_peaks:
            inference_kwargs['num_peaks'] = config['num_peak']

        result = inference_folder(**inference_kwargs)
        result = list(result) if isinstance(result, tuple) else [result]
        self.points2d = result.pop(0)
        self.conf = result.pop(0)
        heatmaps = result.pop(0) if keep_heatmaps else None
        peaks = result.pop(0) if save_top_k_peaks else None

        # 2d pose estimation outputs 19 points, which is what a single camera sees,
        #     however there are 38 joints in total
        points2d_cp = np.zeros((self.points2d.shape[0], self.points2d.shape[1], self.points2d.shape[2]*2, 2))
        points2d_cp[self.camera_ordering[:3], :, :19] = self.points2d[self.camera_ordering[:3]]
        points2d_cp[self.camera_ordering[4:], :, 19:] = self.points2d[self.camera_ordering[4:]]

        # antennae not visible from hind corner cameras
        points2d_cp[self.camera_ordering[0], :, 15] = 0
        points2d_cp[self.camera_ordering[6], :, 19+15] = 0
        # stripes not visible from front corner cameras
        points2d_cp[self.camera_ordering[2], :, 16:19] = 0
        points2d_cp[self.camera_ordering[4], :, 19+16:19+19] = 0

        # flip lr back left-hand-side cameras
        for cidx in [4,5,6]:
            points2d_cp[self.camera_ordering[cidx], ..., 1] = 1 - points2d_cp[self.camera_ordering[cidx], ..., 1]
            # points2d_cp[points2d_cp==1] == 0 # ugly hack

        # fmt:on
        self.points2d = points2d_cp

        if peaks is not None:
            # Mirror the 19-->38 joint expansion above. peaks has shape
            # [n_cameras, n_frames, 19, K, 3] with last axis (y, x, score).
            n_cameras, n_frames, _, K, _ = peaks.shape
            peaks_cp = np.zeros((n_cameras, n_frames, 38, K, 3), dtype=np.float32)
            peaks_cp[self.camera_ordering[:3], :, :19] = peaks[self.camera_ordering[:3]]
            peaks_cp[self.camera_ordering[4:], :, 19:] = peaks[self.camera_ordering[4:]]
            # cameras 0 and 6 cannot see the stripes and antenna
            peaks_cp[self.camera_ordering[2], :, 15:] = 0
            peaks_cp[self.camera_ordering[4], :, 19+15:] = 0
            # Flip lr for cams 4,5,6 to match points2d, but only on real peaks
            # (score > 0); padded slots stay (0, 0, 0).
            for cidx in [4, 5, 6]:
                cam = self.camera_ordering[cidx]
                real = peaks_cp[cam, ..., 2] > 0
                peaks_cp[cam, ..., 1] = np.where(
                    real, 1 - peaks_cp[cam, ..., 1], peaks_cp[cam, ..., 1]
                )
            self.top_k_peaks = peaks_cp
        else:
            self.top_k_peaks = None

        if heatmaps is not None:
            # Heatmaps shape [n_cameras, n_frames, 19, H, W]. Mirror the
            # 19->38 joint expansion done for points2d above, including the
            # horizontal flip of the heatmap's width axis (column) for the
            # right-side cameras whose images were flipped before inference.
            n_cameras, n_frames, _, H, W = heatmaps.shape
            heatmaps_cp = np.zeros((n_cameras, n_frames, 38, H, W), dtype=np.float32)
            heatmaps_cp[self.camera_ordering[:3], :, :19] = heatmaps[self.camera_ordering[:3]]
            heatmaps_cp[self.camera_ordering[4:], :, 19:] = heatmaps[self.camera_ordering[4:]]
            heatmaps_cp[self.camera_ordering[2], :, 15:] = 0
            heatmaps_cp[self.camera_ordering[4], :, 19+15:] = 0
            for cidx in [4, 5, 6]:
                cam = self.camera_ordering[cidx]
                heatmaps_cp[cam] = heatmaps_cp[cam, :, :, :, ::-1]
            self._heatmaps = heatmaps_cp
        else:
            self._heatmaps = None

    def next_error(self, img_id):
        """Finds the next image with an error in prediction after img_id.

        Parameters:
        img_id: a valid image id after which to search for an error.

        Returns:
        int: None or the id of an image with an error in prediction.,
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

        with open(calib_path, "rb") as f:
            calib = pickle.load(f)
        calib_reordered = {
            cidx: calib[idx] for (idx, cidx) in enumerate(self.camera_ordering)
        }

        image_path = os.path.join(self.input_folder, "camera_{cam_id}_img_{img_id}.jpg")

        self.camNet = CameraNetwork(
            self.points2d * self.image_shape[::-1], calib=calib_reordered, image_path=image_path
        )
        self.camNet.bundle_adjust(update_intrinsic=False, update_distort=False)
        # camNet built from fresh points2d -> reprojection-error cache stale.
        self._reproj_err_norms_cache = None
        print(f"Reprojection error is {self.camNet.reprojection_error()}")

    def run_belief_propagation(self):
        """Apply pictorial-structures pose correction (Fig. 10 of the 2019 eLife paper).

        Requires `pose2d_estimation(keep_heatmaps=True)` to have been called
        and `calibrate_calc` to have produced a calibrated `self.camNet`.

        For each frame, BP scores cross-camera triangulations of the network's
        per-camera top-K heatmap peaks against bone-length priors and per-view
        heatmap probabilities, picks the MAP configuration per leg, and writes
        the corrected (per-camera, per-joint) 2D points back into both
        `self.points2d` and the cameras' stored 2D points so that the next
        `save()` produces a triangulation from the corrected detections.
        Heatmaps and any cached reprojection-error tensor are released after
        the run.
        """
        from tqdm import tqdm
        from df3d.belief_propagation import solve_belief_propagation

        if self._heatmaps is None:
            raise RuntimeError(
                'Heatmaps not retained from inference; '
                'call pose2d_estimation(keep_heatmaps=True) first.'
            )
        if self.camNet is None or not self.camNet.has_calibration():
            raise RuntimeError(
                'BP needs a calibrated camNet; run calibrate_calc first.'
            )
        # BP addresses cameras and heatmaps by integer cam_id; a None cam_id
        # (a Camera built outside pyba.CameraNetwork) would silently misindex.
        cams_lacking_id = [c for c in self.camNet.cam_list
                           if not isinstance(c.cam_id, (int, np.integer))]
        if cams_lacking_id:
            raise RuntimeError(
                'Belief propagation indexes cameras by integer cam_id, but '
                f'{len(cams_lacking_id)} camera(s) have non-integer cam_id '
                f'{[c.cam_id for c in cams_lacking_id]}. Build the camera network '
                'via pyba.CameraNetwork (which assigns cam_id automatically).'
            )

        # Attach this camera's heatmaps for the duration of the BP run.
        for cam in self.camNet.cam_list:
            cam.heatmaps = self._heatmaps[cam.cam_id]

        # The middle camera (camera_ordering[3]) has no per-joint heatmaps
        # populated by the current pipeline; skip it to keep BP's candidate-
        # product non-empty.
        mid_cam_id = int(self.camera_ordering[3])
        bp_cams = [c for c in self.camNet.cam_list if c.cam_id != mid_cam_id]

        bone_param = config['bone_param']
        num_peak = config['num_peak']

        # solve_belief_propagation returns a list (len(bp_cams)) of (38, 2)
        # arrays in (x_norm, y_norm) form. Build corrected per-frame outputs.
        n_cams = self.camNet.get_ncams()
        n_frames = self.num_images
        # Start from a copy of the current points2d so cams not touched by BP
        # (the middle one) and joints not visible to a given camera keep
        # their network-predicted values.
        corrected = np.copy(self.points2d)

        for img_id in tqdm(range(self.start_image_idx,
                                 self.start_image_idx + n_frames),
                           desc='Belief Propagation'):
            bp_pts = solve_belief_propagation(
                cam_list=bp_cams, img_id=img_id,
                bone_param=bone_param, num_peak=num_peak,
            )
            # bp_pts[i] is a (38, 2) array in (x_norm, y_norm). Write it back
            # into corrected[cam.cam_id], swapping to (y_norm, x_norm) to
            # match the Convention A used by the rest of df3d.
            for bp_idx, cam in enumerate(bp_cams):
                pts_xy = bp_pts[bp_idx]
                pts_yx = pts_xy[:, ::-1]
                # only overwrite joints visible from this camera so we don't
                # zero out joints BP left untouched.
                vis = np.array([
                    config['skeleton'].camera_see_joint(cam.cam_id, j_id)
                    for j_id in range(38)
                ])
                # also keep zeros where BP returned zero (no candidate)
                nonzero = np.any(pts_yx != 0, axis=-1)
                mask = vis & nonzero
                corrected[cam.cam_id, img_id, mask] = pts_yx[mask]

        self.points2d = corrected
        # Refresh each camera's stored points2d (in pyba (x_pix, y_pix) form,
        # which is image_shape * (x_norm, y_norm) = image_shape * points2d_yx[..., ::-1]).
        scaled_xy = self.points2d[..., ::-1] * self.image_shape
        for cam in self.camNet.cam_list:
            cam.points2d = scaled_xy[cam.cam_id]
            cam.heatmaps = None  # release heatmap reference

        self._heatmaps = None
        self._reproj_err_norms_cache = None

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

    def plot_2d(self, cam_id, img_id, with_corrections=False,
                smooth=False, joints=[], reprojection=False):
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
        from pyba.config import df3d_bones, df3d_colors

        if with_corrections and reprojection:
            raise ValueError("'with_corrections' and 'reprojection' "
                             "cannot both be set to True")

        cam = self.camNet[cam_id]
        if reprojection:
            return cam.plot_reprojections(img_id, self.camNet.points3d,
                                          bones=df3d_bones, colors=df3d_colors)
        pts2d = self.corrected_points2d(cam_id, img_id) if with_corrections else None
        return cam.plot_2d(img_id, points2d=pts2d,
                           bones=df3d_bones, colors=df3d_colors)

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

        When ``config["align_body_axes"]`` is True (the default) the points are
        rotated into the fly body frame, so the axes are anatomically
        meaningful: x = anterior-posterior (+x anterior), y = medial-lateral
        (+y the fly's left), z = dorsal-ventral (+z dorsal / leg lift). See
        ``df3d.body_align``. Set the config key to False to keep the raw
        procrustes-template frame.
        """

        points3d = np.copy(self.camNet.points3d)
        points3d = procrustes_seperate(points3d)
        # This used to be normalize_pose_3d(..., rotate=True), which applies
        # plot_util.rotate_points3d(): it swaps the y and z axes and negates
        # both, a transform whose determinant is -1. That is a reflection, not
        # a rotation, and it existed only to make the arbitrary
        # procrustes-template frame display upright. The template is now itself
        # body-aligned, so the points arrive upright and the reflection would
        # merely mirror the fly -- swapping its left and right, and disagreeing
        # with the frame save() writes. Dropped.
        points3d = normalize_pose_3d(points3d)
        points3d = filter_batch(points3d)
        if config.get("align_body_axes", True):
            points3d = align_to_body_axes(points3d)
        return points3d

    def save_corrections(self):
        """Writes the manual corrections to a file in the output folder."""
        self.db.dump()

    def save(self):
        """Saves the pose estimation results to a file in the output folder."""
        dict_merge = dict()
        dict_merge["points2d"] = np.copy(self.points2d)

        if self.camNet is not None and self.camNet.has_calibration():
            self.camNet.triangulate()
            pts3d = self.camNet.points3d
            dict_merge["points3d_wo_procrustes"] = pts3d
            pts3d = procrustes_seperate(pts3d)
            if config.get("align_body_axes", True):
                # Rotate into the fly body frame: x = anterior-posterior,
                # y = medial-lateral (+y = fly's left), z = dorsal-ventral
                # (+z = dorsal / leg lift). See df3d.body_align. This is a rigid
                # rotation, so it leaves joint angles and all relative geometry
                # unchanged; points3d_wo_procrustes preserves the raw frame.
                pts3d = align_to_body_axes(pts3d)
            dict_merge["points3d"] = pts3d
            # Record whether points3d is in the body frame, so downstream
            # loaders can align legacy (raw-frame) results without re-rotating
            # already-aligned ones.
            dict_merge["body_axis_aligned"] = bool(
                config.get("align_body_axes", True))
            dict_merge = {**self.camNet.summarize(), **dict_merge}
        else:
            logger.debug("Triangulation skipped.")

        dict_merge["camera_ordering"] = self.camera_ordering
        dict_merge["heatmap_confidence"] = self.conf
        dict_merge["image_shape"] = self.image_shape
        if getattr(self, 'top_k_peaks', None) is not None:
            dict_merge["top_k_peaks"] = self.top_k_peaks

        with open(self.save_path, "wb") as f:
            pickle.dump(dict_merge, f)
        print(f"Saved results at: {self.save_path}")

    # -------------------------------------------------------------------------
    # private helper methods

    def corrected_points2d(self, cam_id, img_id):
        """Gets the estimated or manually corrected 2d position of the joints.

        Returns:
        An array with the position of the joints on img_id from cam_id.
        """

        points2d = self.camNet.cam_list[cam_id][img_id].copy()
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
        pts2d = self.camNet.points2d
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.start_image_idx, self.start_image_idx + self.num_images):
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

    def get_fps(self):
        rates = []
        for vid in (glob.glob(os.path.join(self.input_folder, "camera_?.mp4"))
                    + glob.glob(os.path.join(self.input_folder, "camera_?.avi"))):
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=avg_frame_rate", "-of",
                   "default=noprint_wrappers=1:nokey=1", vid]
            try:
                rates.append(subprocess.check_output(cmd, text=True))
            except:
                logger.warning(f"Command failed: {' '.join(cmd)}")
                break
        if len(rates) == 0:
            return None
        if any(rate != rates[0] for rate in rates):
            logger.warning("Framerates of input videos differ from one another,"
                           f" using the first one: {rates}")
        rate = rates[0]
        try:
            return float(rate)
        except ValueError:
            pass
        try:
            numerator, denominator = map(int, rate.split('/'))
            return numerator / denominator if denominator != 0 else None
        except ValueError:
            pass
        logger.warning(f'Could not parse framerate from string "{rate}" returned'
                       ' by ffprobe command, so setting fps to None.')
        return None

    def expand_videos(self):
        """expands video camera_x.mp4 or camera_x.avi into set of images camera_x_img_y.jpg"""
        for vid in (glob.glob(os.path.join(self.input_folder, "camera_?.mp4"))
                    + glob.glob(os.path.join(self.input_folder, "camera_?.avi"))):
            cam_id = parse_vid_name(os.path.basename(vid))
            if not (
                os.path.exists(
                    os.path.join(self.input_folder, f"camera_{cam_id}_img_0.jpg")
                )
                or os.path.exists(
                    os.path.join(self.input_folder, f"camera_{cam_id}_img_000000.jpg")
                )
            ):
                command = f"ffmpeg -nostats -loglevel error -i {vid} -qscale:v 2 -start_number 0 {self.input_folder}/camera_{cam_id}_img_%d.jpg  < /dev/null"
                subprocess.call(command, shell=True)

    def delete_images(self):
        """Delete images under self.input_folder.

        Deletes the images with signature {self.input_folder}/camera_{cam_id}_img_{img_id}.jpg for all img_id,
        Images are deleted only given {self.input_folder}/camera_{cam_id}.mp4 or .avi exists.

        Returns:
        Nothing.
        """
        for vid in (glob.glob(os.path.join(self.input_folder, "camera_[0-9].mp4"))
                    + glob.glob(os.path.join(self.input_folder, "camera_[0-9].avi"))):
            cam_id = parse_vid_name(os.path.basename(vid))
            pattern = os.path.join(self.input_folder, f'camera_{cam_id}_img_*.jpg')
            command = f"rm {pattern}"
            logger.debug(f"Deleting images for camera {cam_id}.")
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

        Compares the reprojection error against the per-joint threshold in
        config["reproj_thr"]. Pre-refactor, the error was computed separately
        against `camNetLeft` and `camNetRight` (two halves of the camera ring)
        and the max was taken; with the current single-camNet pipeline we
        instead take the max over all cameras that see this joint.

        Returns:
        boolean: whether there is a suspected error for joint_id on img_id.
        """
        err_per_cam = self._reprojection_error_norms()[:, img_id, joint_id]
        return float(np.max(err_per_cam)) > config["reproj_thr"][joint_id]

    def _reprojection_error_norms(self):
        """Per-camera per-frame per-joint reprojection error magnitudes (pixels).

        Cached on first call. Joints not visible from a given camera have a
        zero residual (per `pyba.Camera.can_see_mask`). Note: this is computed
        from `camNet.points2d`, which reflects network predictions but not
        manual corrections stored in `self.db` — same behaviour as the
        pre-refactor `get_joint_reprojection_error` helper.
        """
        if getattr(self, '_reproj_err_norms_cache', None) is None:
            if self.camNet is None or not self.camNet.has_calibration():
                raise RuntimeError(
                    'Cannot compute reprojection errors before calibration; '
                    'run Core.calibrate_calc() first.'
                )
            self.camNet.triangulate()
            residuals = self.camNet.reprojection_error(reduce=False)
            self._reproj_err_norms_cache = np.linalg.norm(residuals, axis=-1)
        return self._reproj_err_norms_cache

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
        original_points2d = self.camNet.cam_list[cam_id][img_id]
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
