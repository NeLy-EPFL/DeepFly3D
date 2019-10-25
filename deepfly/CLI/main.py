import math
import argparse, os.path
import logging
import re
from pathlib import Path
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.pose2d import ArgParse
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

logger = logging.getLogger(__name__)
img3d_dpi = 100  # this is the dpi for one image on the 3d video's grid
img3d_aspect = (2, 2)  # this is the aspect ration for one image on the 3d video's grid
img2d_aspect = (2, 1)  # this is the aspect ration for one image on the 3d video's grid
video_width = 500  # total width of the 2d and 3d videos

known_users = [  
    # TODO: Put your regexes and ordering here.
    (r'/CLC/', [0, 6, 5, 4, 3, 2, 1]),
]


def main():
    setup_logger()

    args = parse_cli_args()  # parse the CLI args using ArgParse
    clean_cli_args(args)     # clean and validate the input values got from ArgParse

    # If the flag --debug-args is set, we only output args and terminate
    if args.debug_args:
        print('---- DEBUG MODE ----')
        print(args)
        print()
        return 0

    # For each input folder (found recursively or not)
    unlabeled_folder_list = find_leaf_recursive(args.unlabeled) if args.unlabeled_recursive else [args.unlabeled]
    for unlabeled_folder in unlabeled_folder_list:

        # First, we update the args based on the folder's content
        max_img_id = get_max_img_id(unlabeled_folder)
        args.num_images = min(max_img_id+1, args.num_images_max)
        args.input_folder = unlabeled_folder
        args.unlabeled = unlabeled_folder
        args.unlabeled_recursive = False

        # Then, we check if cameras need reordering
        setup_default_camera_ordering(args)
        save_camera_ordering(args)

        # Then, we run the pose estimation (if the flag --skip-estimation is not set)
        if not args.skip_estimation:
            pose2d_main(args)

        # Then, we create 2d videos if the flag --vid2d is set
        if args.vid2d:
            make_pose2d_video(args)

        # And we create 3d videos if the flag --vid3d is set
        if args.vid3d:
            make_pose3d_video(args)
    
    return args


def setup_logger():
    logger.setLevel(logging.DEBUG)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description = "DeepFly3D pose estimation"
    )
    parser.add_argument(
        "input_folder", 
        help="Folder containing unlabeled images."
    )

    parser.add_argument(
        "-n", "--num-images-max",
        help="Maximal number of images to process.",
        default=math.inf,
        type=int
    )
    parser.add_argument(
        "-ids", "--camera-ids",
        help="Ordering of the cameras provided as an ordered list of ids. Example: 0 1 4 3 2 5 6.",
        default=None,
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--debug-args",
        help="Displays the argument list for debugging purposes",
        action='store_true'
    )
    parser.add_argument(
        "--vid2d",
        help="Generate pose2d videos",
        action='store_true'
    )
    parser.add_argument(
        "--vid3d",
        help="Generate pose3d videos",
        action='store_true'
    )
    parser.add_argument(
        "--skip-estimation",
        help="Skip pose estimation",
        action='store_true'
    )
    parser = ArgParse.add_arguments(parser)
    return parser.parse_args()


def clean_cli_args(args):
    # Cleanup input values
    args.input_folder = os.path.abspath(args.input_folder).rstrip('/')
    
    # Add custom constants
    args.unlabeled = args.input_folder
    
    # Validate the provided camera ordering
    if args.camera_ids:
        ids = set(args.camera_ids)  # only keep unique ids
        if len(ids) != config['num_cameras']:
            raise ValueError('CAMERA-IDS argument must contain {} distinct ids, one per camera'.format(config['num_cameras']))


def setup_default_camera_ordering(args):
    """ This is a convenience function which automatically creates a default camera ordering for 
        frequent users in the neuro-engineering lab.
    """
    if args.camera_ids is not None:
        return 

    for regex, ordering in known_users:
        if re.search(regex, args.input_folder):
            logger.debug('Using default ordering for current user: {}'.format(ordering))
            args.camera_ids = ordering
            return


def save_camera_ordering(args):
    """ Saves the camera ordering args.camera_ids to the output_folder """
    if args.camera_ids:
        path = os.path.join(args.input_folder, args.output_folder)
        write_camera_order(path, args.camera_ids)
        logger.debug('Camera ordering wrote to file in "{}"'.format(path))


def make_pose2d_video(args):
    """ Creates pose2d estimation videos """

    # Here we create a generator (keyword "yield")
    def imgs_generator():
        camNet = get_camNet(args)
        
        def stack(img_id):
                row1 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [0, 1, 2]])
                row2 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [4, 5, 6]])
                return np.vstack([row1, row2])
        
        for img_id in range(args.num_images):
            yield stack(img_id)
    
    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()

    make_video(args, 'pose2d.mp4', generator)


def get_camNet(args, cam_id_list=range(7), cam_list=None):
    """ Create and setup a CameraNetwork """

    folder = os.path.join(args.input_folder, args.output_folder)
    print('Looking for data in {}'.format(folder))
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



def getCamNets(args):
    folder = os.path.join(args.input_folder, args.output_folder)
    print('Looking for data in {}'.format(folder))
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
    return camNetAll, camNetLeft, camNetRight


def make_pose3d_video(args):

    # Here we create a generator (keyword "yield")
    def imgs_generator():
        camNetAll, camNetLeft, camNetRight = getCamNets(args)

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

        def stack(img_id):
            row1 = np.hstack([compute_2d_img(camNetLeft, img_id, cam_id) for cam_id in (0, 1, 2)])
            row2 = np.hstack([compute_2d_img(camNetRight, img_id, cam_id) for cam_id in (0, 1, 2)])
            row3 = np.hstack([compute_3d_img(camNetAll, img_id, cam_id) for cam_id in (2, 3, 4)])
            img = np.vstack([row1, row2, row3])
            return img
        
        for img_id in range(args.num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()

    make_video(args, 'pose3d.mp4', generator)


def make_video(args, video_name, imgs):
    """ Code used to generate a video using cv2.
    - args:  the command-line arguments
    - video_name: a string ending with .mp4, for instance: "pose2d.mp4"
    - imgs: an iterable with the images to write
    """

    first_frame = next(imgs)
    imgs = itertools.chain([first_frame], imgs)

    shape = int(first_frame.shape[1]), int(first_frame.shape[0])
    video_path = os.path.join(args.input_folder, args.output_folder, video_name)
    print('Saving video to: ' + video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    output_shape = resize(current_shape=shape, new_width=video_width)
    print('Video size is: {}'.format(output_shape))
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape)

    for img in tqdm(imgs):
        resized = cv2.resize(img, output_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)

    video_writer.release()
    print('Video created at {}\n'.format(video_path))


def resize(current_shape, new_width):
    width, height = current_shape
    ratio = new_width / width;
    return (int(width * ratio), int(height * ratio))


def setup_ax3d(ax1):
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])


def compute_2d_img(camNet1, img_id, cam_id):
    img = camNet1[cam_id].plot_2d(img_id)
    img = cv2.resize(img, (img2d_aspect[0]*img3d_dpi, img2d_aspect[1]*img3d_dpi))
    return img


def compute_3d_img(camNet1, img_id, cam_id):
    import numpy as np

    plt.style.use('dark_background')
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
    fig.tight_layout(pad=0)

    ax3d = Axes3D(fig)
    setup_ax3d(ax3d)
    plot_drosophila_3d(ax3d, camNet1.points3d_m[img_id].copy(), cam_id=cam_id, lim=2, thickness=np.ones((camNet1.points3d_m.shape[1])) * 1.5)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
    

if __name__ == '__main__':
    main()
