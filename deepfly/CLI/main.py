import math
import argparse, os.path
import logging
import re
from pathlib import Path
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.pose2d import ArgParse
from ..GUI.Config import config
from ..GUI.util.os_util import get_max_img_id, write_camera_order

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

    #import pprint
    #pprint.pprint(args)  # only used during dev
    pose2d_main(args)


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
    args.num_images = _num_images(args.input_folder, args.num_images_max)
    args.unlabeled = args.input_folder

    # Validate the provided camera ordering
    if args.camera_ids:
        ids = set(args.camera_ids)  # only keep unique ids
        if len(ids) != config['num_cameras']:
            raise ValueError('CAMERA-IDS argument must contain {} distinct ids, one per camera'.format(config['num_cameras']))


def _num_images(input_folder, num_images_max):
    """Compute the number of images to process based on:
    - The number of images in the input folder (actually their maximal id), and
    - The maximal number of images allowed to process
    """
    max_id = get_max_img_id(input_folder)
    nb_in_folder = max_id + 1
    return min(num_images_max, nb_in_folder)


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
        write_camera_order(os.path.join(args.input_folder, './df3d/'), args.camera_ids)
        logger.debug('Camera ordering wrote to file in "{}"'.format(args.input_folder))


if __name__ == '__main__':
    main()