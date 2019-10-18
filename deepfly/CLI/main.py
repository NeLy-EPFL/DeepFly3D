import math
import argparse, os.path
import logging
import re
from pathlib import Path
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.pose2d import ArgParse
from ..GUI.Config import config
from ..GUI.util.os_util import get_max_img_id, write_camera_order, read_calib, read_camera_order
from ..GUI.CameraNetwork import CameraNetwork
from deepfly.pose2d.utils.osutils import find_leaf_recursive
from ..GUI.util.os_util import *
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

known_users = [  
    # TODO: Put your regexes and ordering here.
    (r'/CLC/', [0, 6, 5, 4, 3, 2, 1]),
]

def main():
    setup_logger()

    args = parse_cli_args()  # parse the CLI args using ArgParse
    clean_cli_args(args)     # clean and validate the input values got from ArgParse

    setup_default_camera_ordering(args)  # custom function for convenience at the lab
    save_camera_ordering(args)           # write the camera ordering to file

    print('------------------------------------------------------')
    print('POSE 2D ESTIMATION')
    pose2d_main(args)

    print()
    print('------------------------------------------------------')
    print('POSE 2D VIDEOS')
    if args.unlabeled_recursive:
        unlabeled_folder_list = find_leaf_recursive(args.unlabeled)
        unlabeled_folder_list = [path for path in unlabeled_folder_list if "images" in path]
    else:
        unlabeled_folder_list = [args.unlabeled]
    for unlabeled_folder in unlabeled_folder_list:
        max_img_id = get_max_img_id(unlabeled_folder)
        args.num_images = min(max_img_id+1, args.num_images_max)
        args.input_folder = unlabeled_folder
        args.unlabeled = unlabeled_folder
        make_pose2d_video(args)
    
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
    parser = ArgParse.add_arguments(parser)
    return parser.parse_args()


def clean_cli_args(args):
    # Cleanup input values
    args.input_folder = os.path.abspath(args.input_folder).rstrip('/')
    
    # Add custom constants
    #args.num_images = _num_images(args.input_folder, args.num_images_max)
    args.unlabeled = args.input_folder
    # Validate the provided camera ordering
    if args.camera_ids:
        ids = set(args.camera_ids)  # only keep unique ids
        if len(ids) != config['num_cameras']:
            raise ValueError('CAMERA-IDS argument must contain {} distinct ids, one per camera'.format(config['num_cameras']))


#def _num_images(input_folder, num_images_max):
#    """Compute the number of images to process based on:
#    - The number of images in the input folder (actually their maximal id), and
#    - The maximal number of images allowed to process
#    """
#    max_id = get_max_img_id(input_folder)
#    nb_in_folder = max_id + 1
#    return min(num_images_max, nb_in_folder)


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
    if args.camera_ids:
        write_camera_order(os.path.join(args.input_folder, 'df3d/'), args.camera_ids)
        logger.debug('Camera ordering wrote to file in "{}"'.format(args.input_folder))


def make_pose2d_video(args):

    folder = os.path.join(args.input_folder, args.output_folder)
    print('Looking for data in {}'.format(folder))
    calib = read_calib(folder)
    cid2cidread, cidread2cid = read_camera_order(folder)

    camNet = CameraNetwork(image_folder=args.input_folder, cam_id_list=range(7), calibration=calib, 
            cid2cidread=cid2cidread, num_images=args.num_images, output_folder=folder)
        
    def stack(img_id):
            row1 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [0, 1, 2]])
            row2 = np.hstack([camNet[cam_id].plot_2d(img_id) for cam_id in [4, 5, 6]])
            return np.vstack([row1, row2])

    first_frame = stack(0)
    shape = int(first_frame.shape[1]), int(first_frame.shape[0])

    #plt.imshow(first_frame)

    video_path = os.path.join(args.input_folder, args.output_folder, 'pose2d.mp4')
    print('Saving pose2d video to: ' + video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, shape)

    for img_id in tqdm(range(args.num_images)):
            grid = stack(img_id)
            resized = cv2.resize(grid, shape)#(int(shape[0]), int(shape[1])))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            video_writer.write(rgb)

    video_writer.release()
    print('Done generating the pose2d video at {}\n'.format(video_path))


if __name__ == '__main__':
    main()
