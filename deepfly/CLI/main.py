import math
import argparse, os.path
from pathlib import Path
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.pose2d import ArgParse
from ..GUI.Config import config
from ..GUI.util.os_util import get_max_img_id

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description = "DeepFly3D pose estimation"
    )

    # Convention for faster vertical reading:
    # - first line is argument, starting with short form (e.g. -f, --folder)
    # - second line is help text (e.g. help="the input folder")
    # - then whatever else is required.
    # last line is closing parenthesis )

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
    parser.add_argument( # TODO: there is "--stacks" in ArgParse, is it the same?
        "--num-stacks",
        help="TODO",
        default=config['num_stacks'],
        type=int
    )
    
    parser = ArgParse.add_arguments(parser)

    return parser.parse_args()


def num_images(input_folder, num_images_max):
    """Compute the number of images to process based on:
    - The number of images in the input folder (actually their maximal id), and
    - The maximal number of images allowed to process
    """
    max_id = get_max_img_id(input_folder)
    nb_in_folder = max_id + 1
    return min(num_images_max, nb_in_folder)
    
    
def main():
    args = parse_cli_args()
    
    # Cleanup input values
    args.input_folder = os.path.abspath(args.input_folder).rstrip('/')
    
    # Add custom constants
    args.num_images = num_images(args.input_folder, args.num_images_max)
    args.checkpoint = False
    args.unlabeled = args.input_folder

    # Run
    pose2d_main(args)


if __name__ == '__main__':
    main()