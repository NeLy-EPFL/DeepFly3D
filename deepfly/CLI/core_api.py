"""core_api.py

This file is the API exposed from to core to the CLI. For decoupling purposes. 

To be clear, this file acts as an interface for the functionalities of DeepFly 
that should have been implemented in the core, rather than in the CLI.
It exposes functions that will be called by the CLI.
"""

import math
import argparse, os.path
import logging
import re
from pathlib import Path
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly import pose2d
from ..GUI.Config import config
from ..GUI.util.os_util import get_max_img_id, write_camera_order, read_calib, read_camera_order
from ..GUI.util.plot_util import normalize_pose_3d
from ..GUI.util.signal_util import *
from ..GUI.CameraNetwork import CameraNetwork
from deepfly.pose2d.utils.osutils import find_leaf_recursive
from ..GUI.util.os_util import *
import cv2
from tqdm import tqdm
import time
import itertools
from deepfly.GUI.util.plot_util import plot_drosophila_3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from deepfly.pose3d.procrustes.procrustes import procrustes_seperate
from logging import getLogger
from deepfly.utils_ramdya_lab import find_default_camera_ordering
import pickle

img3d_dpi = 100  # this is the dpi for one image on the 3d video's grid
img3d_aspect = (2, 2)  # this is the aspect ration for one image on the 3d video's grid
img2d_aspect = (2, 1)  # this is the aspect ration for one image on the 3d video's grid
video_width = 500  # total width of the 2d and 3d videos


#=========================================================================
# Public interface


def setup(input_folder, output_folder, camera_ids, num_images_max, overwrite=False):
    args = _get_pose2d_args(input_folder, output_folder, camera_ids, num_images_max, overwrite)
    _setup_default_camera_ordering(args)
    _create_output_folder(args)
    _save_camera_ordering(args)
    return args


def pose2d_estimation(setup_data):
    pose2d_main(setup_data)

    
def pose3d_estimation(setup_data):
    _pose3d_estimation(setup_data)


def pose2d_video(setup_data):
    return _make_pose2d_video(setup_data)


def pose3d_video(setup_data):
    return _make_pose3d_video(setup_data)


#=========================================================================
# Below is private implementation


def _get_pose2d_args(input_folder, output_folder, camera_ids, num_images_max, overwrite):
    # Validate arguments
    input_folder = os.path.abspath(input_folder).rstrip('/')
    # Create a pose2d.ArgParse to get access to its default values
    p = pose2d.ArgParse.create_parser()
    args = p.parse_args([])
    # and fill it with our real command-line arguments
    args.unlabeled = input_folder
    args.input_folder = input_folder
    args.output_folder = output_folder
    args.camera_ids = camera_ids
    args.unlabeled_recursive = False
    args.num_images_max = num_images_max
    args.overwrite = overwrite
    max_img_id = get_max_img_id(args.input_folder)
    args.num_images = min(max_img_id+1, args.num_images_max)
    return _clean_args(args)


def _clean_args(args):
    args.input_folder = os.path.abspath(args.input_folder).rstrip('/')
    if args.camera_ids:
        ids = set(args.camera_ids)  # only keep unique ids
        if len(ids) != config['num_cameras']:
            raise ValueError('CAMERA-IDS argument must contain {} distinct ids, one per camera'.format(config['num_cameras']))
    return args


def _setup_default_camera_ordering(args):
    """ This is a convenience function which automatically creates a default camera ordering for
        frequent users in the neuro-engineering lab.
    """
    args.camera_ids = np.array(args.camera_ids) if args.camera_ids else find_default_camera_ordering(args.input_folder)


def _create_output_folder(args):
    path = os.path.join(args.input_folder, args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    getLogger('df3d').debug("Creating output folder {}".format(path))


def _save_camera_ordering(args):
    """ Saves the camera ordering args.camera_ids to the output_folder """
    if args.camera_ids is not None:
        path = os.path.join(args.input_folder, args.output_folder)
        write_camera_order(path, args.camera_ids)
        getLogger('df3d').debug('Camera ordering wrote to file in "{}"'.format(path))


def _pose3d_estimation(args):
    camNetAll, camNetLeft, camNetRight = _getCamNets(args)

    pts2d = np.zeros((7, args.num_images, config["num_joints"], 2), dtype=float)
    for cam in camNetAll:
        pts2d[cam.cam_id, :] = cam.points2d.copy()

    # some post-processing for body-coxa
    if "fly" in config["name"]:
        for cam_id in range(len(camNetAll.cam_list)):
            for j in range(config["skeleton"].num_joints):
                coxa_tracked = config["skeleton"].is_tracked_point(j, config["skeleton"].Tracked.BODY_COXA)
                if config["skeleton"].camera_see_joint(cam_id, j) and coxa_tracked:
                    pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                    pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])

    dict_merge = camNetAll.save_network(path=None)
    dict_merge["points2d"] = pts2d
    dict_merge["points3d"] = camNetAll.points3d_m

    path = os.path.join(args.input_folder, args.output_folder)
    save_path = os.path.join(path, "pose_result_{}.pkl".format(args.input_folder.replace("/", "_")))
    pickle.dump(dict_merge, open(save_path, "wb"))
    getLogger('df3d').info(f"Pose estimation results saved at: {save_path}")


def _make_pose2d_video(args):
    """ Creates pose2d estimation videos """
    # Here we create a generator (keyword "yield")
    def imgs_generator():
        camNet = _get_camNet(args)

        def stack(img_id):
                row1 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [0, 1, 2]])
                row2 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [4, 5, 6]])
                return np.vstack([row1, row2])

        for img_id in range(args.num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()

    _make_video(args, 'pose2d.mp4', generator)


def _get_camNet(args, cam_id_list=range(7), cam_list=None):
    """ Create and setup a CameraNetwork """

    folder = os.path.join(args.input_folder, args.output_folder)
    getLogger('df3d').debug('Looking for data in {}'.format(folder))
    calib = read_calib(config['calib_fine'])
    cid2cidread, _ = read_camera_order(folder)

    camNet = CameraNetwork(
        image_folder=args.input_folder,
        cam_id_list=cam_id_list,
        calibration=calib,
        cid2cidread=cid2cidread,
        num_images=args.num_images, output_folder=folder,
        cam_list=cam_list
    )

    return camNet


def _getCamNets(args):
    folder = os.path.join(args.input_folder, args.output_folder)
    getLogger('df3d').debug('Looking for data in {}'.format(folder))
    calib = read_calib(config['calib_fine'])
    cid2cidread, _ = read_camera_order(folder)

    camNetAll = CameraNetwork(
        image_folder=args.input_folder,
        output_folder=folder,
        cam_id_list=range(config["num_cameras"]),
        cid2cidread=cid2cidread,
        num_images=args.num_images,
        calibration=calib,
        num_joints=config["skeleton"].num_joints,
        heatmap_shape=config["heatmap_shape"],
    )
    camNetLeft = CameraNetwork(
        image_folder=args.input_folder,
        output_folder=folder,
        cam_id_list=config["left_cameras"],
        num_images=args.num_images,
        calibration=calib,
        num_joints=config["skeleton"].num_joints,
        cid2cidread=[cid2cidread[cid] for cid in config["left_cameras"]],
        heatmap_shape=config["heatmap_shape"],
        cam_list=[cam for cam in camNetAll if cam.cam_id in config["left_cameras"]],
    )
    camNetRight = CameraNetwork(
        image_folder=args.input_folder,
        output_folder=folder,
        cam_id_list=config["right_cameras"],
        num_images=args.num_images,
        calibration=calib,
        num_joints=config["skeleton"].num_joints,
        cid2cidread=[cid2cidread[cid] for cid in config["right_cameras"]],
        heatmap_shape=config["heatmap_shape"],
        cam_list=[cam for cam in camNetAll if cam.cam_id in config["right_cameras"]],
    )

    camNetLeft.bone_param = config["bone_param"]
    camNetRight.bone_param = config["bone_param"]
    camNetAll.load_network(calib)

    camNetLeft.triangulate()
    camNetLeft.bundle_adjust(cam_id_list=(0,1,2), unique=False, prior=True)

    camNetRight.triangulate()
    camNetRight.bundle_adjust(cam_id_list=(0,1,2), unique=False, prior=True)

    camNetAll.triangulate()
    camNetAll.points3d_m = procrustes_seperate(camNetAll.points3d_m)
    camNetAll.points3d_m = normalize_pose_3d(camNetAll.points3d_m, rotate=True)
    camNetAll.points3d_m = filter_batch(camNetAll.points3d_m)
    for cam in camNetAll:
        cam.points2d = smooth_pose2d(cam.points2d)

    return camNetAll, camNetLeft, camNetRight


def _make_pose3d_video(args):
    # Here we create a generator (keyword "yield")
    def imgs_generator():
        camNetAll, camNetLeft, camNetRight = _getCamNets(args)

        def stack(img_id):
            row1 = np.hstack([_compute_2d_img(camNetLeft, img_id, cam_id) for cam_id in (0, 1, 2)])
            row2 = np.hstack([_compute_2d_img(camNetRight, img_id, cam_id) for cam_id in (0, 1, 2)])
            row3 = np.hstack([_compute_3d_img(camNetAll, img_id, cam_id) for cam_id in (2, 3, 4)])
            img = np.vstack([row1, row2, row3])
            return img

        for img_id in range(args.num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()
    video_name = 'video_pose3d_' + args.input_folder.replace('/', '_') + '.mp4'
    _make_video(args, video_name, generator)


def _make_video(args, video_name, imgs):
    """ Code used to generate a video using cv2.
    - args:  the command-line arguments
    - video_name: a string ending with .mp4, for instance: "pose2d.mp4"
    - imgs: an iterable with the images to write
    """

    first_frame = next(imgs)
    imgs = itertools.chain([first_frame], imgs)

    shape = int(first_frame.shape[1]), int(first_frame.shape[0])
    video_path = os.path.join(args.input_folder, args.output_folder, video_name)
    getLogger('df3d').debug('Saving video to: ' + video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    output_shape = _resize(current_shape=shape, new_width=video_width)
    getLogger('df3d').debug('Video size is: {}'.format(output_shape))
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape)

    progress_bar = tqdm if getLogger('df3d').isEnabledFor(logging.INFO) else lambda x: x
    for img in progress_bar(imgs):
        resized = cv2.resize(img, output_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)

    video_writer.release()
    getLogger('df3d').info('Video created at {}\n'.format(video_path))


def _resize(current_shape, new_width):
    width, height = current_shape
    ratio = new_width / width;
    return (int(width * ratio), int(height * ratio))


def _setup_ax3d(ax1):
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])


def _compute_2d_img(camNet1, img_id, cam_id):
    img = camNet1[cam_id].plot_2d(img_id)
    img = cv2.resize(img, (img2d_aspect[0]*img3d_dpi, img2d_aspect[1]*img3d_dpi))
    return img


def _compute_3d_img(camNet1, img_id, cam_id):
    import numpy as np

    plt.style.use('dark_background')
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
    fig.tight_layout(pad=0)

    ax3d = Axes3D(fig)
    _setup_ax3d(ax3d)
    plot_drosophila_3d(ax3d, camNet1.points3d_m[img_id].copy(), cam_id=cam_id, lim=2, thickness=np.ones((camNet1.points3d_m.shape[1])) * 1.5)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


if __name__ == '__main__':
    main()

