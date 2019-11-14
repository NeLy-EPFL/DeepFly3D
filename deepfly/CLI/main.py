import argparse, math
from pathlib import Path
from colorama import Style, init as colorama_init
import logging
from logging import getLogger
from . import core_api
from . import utils


def main():
    args = parse_cli_args()
    colorama_init()

    setup_logger(args)
    
    if args.debug:
        return print_debug(args)

    if args.from_file and args.recursive:
        getLogger('df3d').error('Error: choose an input method between "from file" and "recursive" but not both.')
        return 1

    if args.recursive:
        return run_recursive(args)

    if args.from_file:
        return run_from_file(args)

    return run(args)


def setup_logger(args):
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = getLogger('df3d')
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    #
    if args.verbose:
        logger.setLevel(logging.INFO)
    #
    if args.verbose2:
        logger.setLevel(logging.DEBUG)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description = "DeepFly3D pose estimation"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable info output (such as progress bars)",
        action='store_true'
    )
    parser.add_argument(
        "-vv", "--verbose2",
        help="Enable debug output",
        action='store_true'
    )
    parser.add_argument(
        "-d", "--debug",
        help="Displays the argument list for debugging purposes",
        action='store_true'
    )
    parser.add_argument(
        "input_folder", 
        help="Without additional arguments, a folder containing unlabeled images.",
        metavar="INPUT"
    )
    parser.add_argument(
        "-r", "--recursive",
        help="INPUT is a folder. Successively use its subfolders named 'images/'",
        action='store_true'
    )
    parser.add_argument(
        "-f", "--from-file",
        help="INPUT is a text-file, where each line names a folder. Successively use the listed folders.",
        action='store_true',
    )
    parser.add_argument(
        "-n", "--num-images-max",
        help="Maximal number of images to process.",
        default=math.inf,
        type=int
    )
    parser.add_argument(
        "-i", "--camera-ids",
        help="Ordering of the cameras provided as an ordered list of ids. Example: 0 1 4 3 2 5 6.",
        default=None,
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "-2d", "--video-2d",
        help="Generate pose2d videos",
        action='store_true'
    )
    parser.add_argument(
        "-3d", "--video-3d",
        help="Generate pose3d videos",
        action='store_true'
    )
    parser.add_argument(
        "-skip", "--skip-estimation",
        help="Skip pose estimation",
        action='store_true'
    )
    return parser.parse_args()


def print_debug(args):
    print(f"Enabled logging level: {logging.getLevelName(getLogger('df3d').getEffectiveLevel())}")
    #
    print('Arguments are:')
    for key,val in vars(args).items():
        print(f'\t{key}: {val}')
    print()
    return 0


def run_from_file(args):
    getLogger('df3d').info(f'{Style.BRIGHT}Looking for folders listed in {args.input_folder}{Style.RESET_ALL}')
    try:
        with open(args.input_folder, 'r') as f:
            folders = list(line.strip() for line in f)
    except FileNotFoundError:
        getLogger('df3d').error(f'Unable to find the file {args.input_folder}')
        return 1
    except IsADirectoryError:
        getLogger('df3d').error(f'{args.input_folder} is a directory, please provide a file instead.')
        return 1

    folders = list(dict.fromkeys(folders))  # removes duplicate entries

    errors = False
    current_directory = Path('.')
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            getLogger('df3d').error(f'[Error] Not a directory or does not exist: {str(folder)}')
            errors = True
    if errors:
        return 1

    s = 's' if len(folders) > 1 else ''
    folders_str = "\n-".join(folders)
    getLogger('df3d').info(f'Folder{s} found:\n-{folders_str}')
    args.from_file = False
    run_in_folders(args, folders)


def run_recursive(args):
    subfolder_name = 'images'
    getLogger('df3d').info(f'{Style.BRIGHT}Recursively looking for subfolders named `{subfolder_name}` inside `{args.input_folder}`{Style.RESET_ALL}')
    subfolders = utils.find_subfolders(args.input_folder, 'images')
    s = 's' if len(subfolders) > 1 else ''
    folders_str = "\n-".join(subfolders)
    getLogger('df3d').info(f'Found {len(subfolders)} subfolder{s}:\n-{folders_str}')
    args.recursive = False
    run_in_folders(args, subfolders)


def run_in_folders(args, folders):
    for folder in folders:
        args.input_folder = folder
        run(args)


def run(args):
    nothing_to_do = args.skip_estimation and (not args.vid2d) and (not args.vid3d)
    
    if nothing_to_do:
        getLogger('df3d').info(f'{Style.BRIGHT}Nothing to do. Check your command-line arguments.{Style.RESET_ALL}')
        return 0
    
    getLogger('df3d').info(f'{Style.BRIGHT}Working in {args.input_folder}{Style.RESET_ALL}')
    setup_data = core_api.setup(args.input_folder, args.camera_ids, args.num_images_max)

    if not args.skip_estimation:
        core_api.pose_estimation(setup_data)

    if args.video_2d:
        core_api.pose2d_video(setup_data)

    if args.video_3d:
        core_api.pose3d_video(setup_data)

    return 0


if __name__ == '__main__':
    main()
