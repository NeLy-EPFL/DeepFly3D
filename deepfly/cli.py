import argparse
from collections import deque  # for find_subfolder
import logging
import math
import os  # for find_subfolder
from pathlib import Path
import numpy as np

from colorama import Style, init as colorama_init

from deepfly import logger
from deepfly import video
from deepfly.core import Core


def main():
    args = parse_cli_args()
    colorama_init()

    setup_logger(args)

    if args.debug:
        return print_debug(args)

    if args.from_file and args.recursive:
        msg = 'Error: choose an input method between "from file" and "recursive" but not both.'
        logger.error(msg)
        return 1

    if args.recursive:
        return run_recursive(args)

    if args.from_file:
        return run_from_file(args)

    return run(args)


def setup_logger(args):
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    l = logger.getLogger()
    l.addHandler(handler)
    l.setLevel(logging.WARNING)
    #
    if args.verbose:
        l.setLevel(logging.INFO)
    #
    if args.verbose2:
        l.setLevel(logging.DEBUG)


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
        "--output-folder",
        help="The name of subfolder where to write results",
        default='df3d',
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
        "-o", "--overwrite",
        help="Rerun pose estimation and overwrite existing pose results",
        action='store_true'
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
        help="Skip 2D and 3D pose estimation",
        action='store_true'
    )
    return parser.parse_args()


def print_debug(args):
    print(f"Enabled logging level: {logging.getLevelName(logger.getLogger().getEffectiveLevel())}")
    #
    print('Arguments are:')
    for key,val in vars(args).items():
        print(f'\t{key}: {val}')
    print()
    return 0


def run_from_file(args):
    logger.info(f'{Style.BRIGHT}Looking for folders listed in {args.input_folder}{Style.RESET_ALL}')
    try:
        with open(args.input_folder, 'r') as f:
            folders = list(line.strip() for line in f)
    except FileNotFoundError:
        logger.error(f'Unable to find the file {args.input_folder}')
        return 1
    except IsADirectoryError:
        logger.error(f'{args.input_folder} is a directory, please provide a file instead.')
        return 1

    folders = list(dict.fromkeys(folders))       # removes duplicate entries
    folders = [f for f in folders if f.strip()]  # remove blank lines
    folders = [Path(f) for f in folders]         # convert to path objects

    bad = [f for f in folders if not f.is_dir()]
    for f in bad:
        logger.error(f'[Error] Not a directory or does not exist: {str(f)}')
    if bad:
        return 1

    s = 's' if len(folders) > 1 else ''
    folders_str = "\n-".join(folders)
    logger.info(f'Folder{s} found:\n-{folders_str}')
    args.from_file = False
    run_in_folders(args, folders)


def run_recursive(args):
    subfolder_name = 'images'
    msg = f'{Style.BRIGHT}Recursively looking for subfolders named `{subfolder_name}` inside `{args.input_folder}`{Style.RESET_ALL}'
    logger.info(msg)
    subfolders = find_subfolders(args.input_folder, 'images')
    s = 's' if len(subfolders) > 1 else ''
    folders_str = "\n-".join(subfolders)
    logger.info(f'Found {len(subfolders)} subfolder{s}:\n-{folders_str}')
    args.recursive = False
    run_in_folders(args, subfolders)


def run_in_folders(args, folders):
    errors = []
    for folder in folders:
        try:
            args.input_folder = folder
            run(args)
        except KeyboardInterrupt:
            logger.warning(f'{Style.BRIGHT}Keyboard Interrupt received. Terminating...{Style.RESET_ALL}')
            break
        except Exception as e:
            errors.append((folder, e))
            logger.error(f'{Style.BRIGHT}An error occured while processing {folder}. Continuing...{Style.RESET_ALL}')
    #
    if errors:
        logger.error(f'\n{Style.BRIGHT}{len(errors)} out of {len(folders)} folders terminated with errors.{Style.RESET_ALL}')
        for (folder, exc) in errors:
            logger.error(f'\n{Style.BRIGHT}In {folder}{Style.RESET_ALL}', exc_info=exc)


def run(args):
    nothing_to_do = args.skip_estimation and (not args.video_2d) and (not args.video_3d)

    if nothing_to_do:
        logger.info(f'{Style.BRIGHT}Nothing to do. Check your command-line arguments.{Style.RESET_ALL}')
        return 0

    logger.info(f'{Style.BRIGHT}\nWorking in {args.input_folder}{Style.RESET_ALL}')
    
    core = Core(args.input_folder, args.output_folder, args.num_images_max)
    core.overwrite = args.overwrite  # monkey-patch: save it for later
    if args.camera_ids is not None:
        core.update_camera_ordering(args.camera_ids)

    if not args.skip_estimation:
        core.pose2d_estimation(core.overwrite)
        core.calibrate_calc(0, core.max_img_id)
        core.save_pose()

    if args.video_2d:
        video.make_pose2d_video(core.plot_2d, core.num_images, core.input_folder, core.output_folder)

    if args.video_3d:
        video.make_pose3d_video(core.get_points3d(), core.plot_2d, core.num_images, core.input_folder, core.output_folder)

    return 0


def find_subfolders(path, name):
    """
    Implements a Breadth First Search algorithm to find all subfolders named `name`.
    Using a BFS allows to stop as soon as we find the target subfolder, without listing its content.
    Which is a performance improvement when target subfolders contain hundreds on thousands of images.
    """
    found = []
    to_visit = deque()
    visited = set()
    
    to_visit.append(Path(path))
    while to_visit:
        current = to_visit.popleft()
        if current.is_dir() and current not in visited:
            visited.add(current)
            if current.name == name:
                found.append(str(current))
            else:
                for child in current.iterdir():
                    to_visit.append(child)
    return found


if __name__ == '__main__':
    main()
