import argparse
import logging
import math
from collections import deque  # for find_subfolder
from pathlib import Path

from colorama import Style
from colorama import init as colorama_init

import df3d.logger as logger
from df3d import video
from df3d.config import config
from df3d.core import Core


def main():
    """Main entry point to run the command-line interface."""

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
    """Configures deepfly's logger to output to console.

    The correct verbosity level is parsed from the command-line arguments.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    """

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
    """Uses ArgParse to parse the command line arguments.

    Returns:
    A simple namespace containing parsed arguments values.
    """

    parser = argparse.ArgumentParser(description="DeepFly3D pose estimation")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable info output (such as progress bars)",
        action="store_true",
    )
    parser.add_argument(
        "-vv", "--verbose2", help="Enable debug output", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Displays the argument list for debugging purposes",
        action="store_true",
    )
    parser.add_argument(
        "input_folder",
        help="Without additional arguments, a folder containing unlabeled images.",
        metavar="INPUT",
    )
    parser.add_argument(
        "--output-folder",
        help="The name of the folder where results will be written. If not specified, a folder with the same name as INPUT suffixed with '_df3d' will be created. If INPUT is a file, the output folder will be created in the same directory as INPUT.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        help="INPUT is a folder. Successively use its subfolders named 'images/'",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--from-file",
        help="INPUT is a text-file, where each line names a folder. Successively use the listed folders.",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--delete-images",
        help="Delete image files *after running df3d-cli*. Only deletes if a corresponding .mp4 or .avi file is already in the folder. Especially useful if you are expanding .mp4's/.avi's for processing.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--num-images-max",
        help="Specify the range of frames to process. Provide a single value N to process"
             " up to N frames starting from frame 0. Provide two values START END to process"
             " frames from START to END inclusive. If not provided, all frames are processed.",
        default=[0],
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--order",
        "--camera-ids",
        help="Ordering of the cameras provided as a list of ids. Example: --order 0 1 4 3 2 5 6.",
        default=[0, 1, 2, 3, 4, 5, 6],
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--video-2d", help="Generate pose2d videos", action="store_true"
    )
    parser.add_argument(
        "--video-3d", help="Generate pose3d videos", action="store_true"
    )
    parser.add_argument(
        "--only-render-legs",
        help="Render only leg keypoints/bones in output videos; hide antenna and stripe keypoints."
             f" Defaults to config['only_render_legs'] = {config['only_render_legs']}.",
        action="store_true",
        default=config["only_render_legs"],
    )
    parser.add_argument(
        "--skip-pose-estimation",
        help="Skip 2D and 3D pose estimation",
        dest="skip_estimation",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for inference - how many images are processed through the model at once",
        type=int,
        default=8
    )
    parser.add_argument(
        "--pin-memory-disabled",
        help="Whether to disable `pin_memory` in the dataloader. Keeping this enabled usually speeds up the processing, but sometimes leads to memory leaks. See https://github.com/NeLy-EPFL/DeepFly2D/issues/6",
        action="store_true",
    )
    parser.add_argument(
        "--save-top-k-peaks",
        help="Also save the top-K local-maximum heatmap peaks per joint into"
             " the df3d_result pkl under the key 'top_k_peaks'. K is taken"
             " from config['num_peak'] (default 10). These peaks are the input"
             " to the pictorial-structures / belief-propagation pose-correction"
             " step from the 2019 eLife paper; enabling this flag is the data-"
             "collection prerequisite for that correction.",
        action="store_true",
    )
    parser.add_argument(
        "--belief-propagation",
        help="Apply pictorial-structures pose correction (Fig. 10 of the 2019"
             " eLife paper) after 2D inference and calibration. BP scores"
             " cross-camera triangulations of the top-K heatmap peaks against"
             " bone-length priors and per-view heatmap probabilities, picks"
             " the MAP configuration per leg, and writes corrected 2D points"
             " back into the result pkl. Requires keeping the full heatmaps"
             " in memory during the run (~8.7 GB / 1000 frames).",
        action="store_true",
    )
    parser.add_argument(
        "--output-fps",
        help="FPS for output videos. If not specified, uses the FPS from the input "
             "videos. If specified, overrides the input video FPS.",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    args.input_folder = Path(args.input_folder).expanduser().resolve()
    if args.output_folder is None:
        args.output_folder = args.input_folder.with_name(args.input_folder.stem + "_df3d")
    else:
        args.output_folder = Path(args.output_folder).expanduser().resolve()
    args.input_folder = str(args.input_folder)
    args.output_folder = str(args.output_folder)

    n_args = args.num_images_max
    if len(n_args) == 1:
        args.start_image_idx = 0
        args.num_images_max = n_args[0]
    elif len(n_args) == 2:
        start, end = n_args[0], n_args[1]
        if start < 0 or end < start:
            parser.error(
                f"-n START END requires START >= 0 and END >= START, got {start} {end}"
            )
        args.start_image_idx = start
        args.num_images_max = end - start + 1
    else:
        parser.error("-n accepts at most 2 values: -n N or -n START END")

    return args


def print_debug(args):
    """Prints each (key, value) pair in args.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    """

    print(
        f"Enabled logging level: {logging.getLevelName(logger.getLogger().getEffectiveLevel())}"
    )
    #
    print("Arguments are:")
    for key, val in vars(args).items():
        print(f"\t{key}: {val}")
    print()
    return 0


def run_from_file(args):
    """Processes every folder listed in the args.input_folder text file.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    """

    logger.info(
        f"{Style.BRIGHT}Looking for folders listed in {args.input_folder}{Style.RESET_ALL}"
    )
    try:
        with open(args.input_folder, "r") as f:
            folders = list(line.strip() for line in f)
    except FileNotFoundError:
        logger.error(f"Unable to find the file {args.input_folder}")
        return 1
    except IsADirectoryError:
        logger.error(
            f"{args.input_folder} is a directory, please provide a file instead."
        )
        return 1

    folders = list(dict.fromkeys(folders))  # removes duplicate entries
    folders = [f for f in folders if f.strip()]  # remove blank lines
    folders_str = "\n-".join(folders)
    folders = [Path(f) for f in folders]  # convert to path objects

    bad = [f for f in folders if not f.is_dir()]
    for f in bad:
        logger.error(f"[Error] Not a directory or does not exist: {str(f)}")
    if bad:
        return 1

    s = "s" if len(folders) > 1 else ""
    logger.info(f"Folder{s} found:\n-{folders_str}")
    args.from_file = False
    run_in_folders(args, folders)


def run_recursive(args):
    """Processes every subfolder named 'images' in the args.input_folder folder.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    """

    subfolder_name = "images"
    msg = f"{Style.BRIGHT}Recursively looking for subfolders named `{subfolder_name}` inside `{args.input_folder}`{Style.RESET_ALL}"
    logger.info(msg)
    subfolders = find_subfolders(args.input_folder, "images")
    s = "s" if len(subfolders) > 1 else ""
    folders_str = "\n-".join(subfolders)
    logger.info(f"Found {len(subfolders)} subfolder{s}:\n-{folders_str}")
    args.recursive = False
    run_in_folders(args, subfolders)


def run_in_folders(args, folders):
    """Processes successively each folder in folders.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    folders: a list of folders to process
    """

    errors = []
    for folder in folders:
        try:
            args.input_folder = folder
            run(args)
        except KeyboardInterrupt:
            logger.warning(
                f"{Style.BRIGHT}Keyboard Interrupt received. Terminating...{Style.RESET_ALL}"
            )
            break
        except Exception as e:
            errors.append((folder, e))
            logger.error(
                f"{Style.BRIGHT}An error occured while processing {folder}. Continuing...{Style.RESET_ALL}"
            )
    #
    if errors:
        logger.error(
            f"\n{Style.BRIGHT}{len(errors)} out of {len(folders)} folders terminated with errors.{Style.RESET_ALL}"
        )
        for (folder, exc) in errors:
            logger.error(f"\n{Style.BRIGHT}In {folder}{Style.RESET_ALL}", exc_info=exc)


def run(args):
    """Processes the image folder args.input_folder.

    Parameters:
    args: the parsed command-line arguments (see: parse_cli_args())
    """
    nothing_to_do = args.skip_estimation and (not args.video_2d) and (not args.video_3d)

    if nothing_to_do:
        logger.info(
            f"{Style.BRIGHT}Nothing to do. Check your command-line arguments.{Style.RESET_ALL}"
        )
        return 0

    logger.info(f"{Style.BRIGHT}\nWorking in {args.input_folder}{Style.RESET_ALL}")

    core = Core(
        args.input_folder, args.output_folder, args.num_images_max, args.order,
        args.start_image_idx, only_render_legs=args.only_render_legs,
    )

    if not args.skip_estimation:
        core.pose2d_estimation(args.batch_size, args.pin_memory_disabled,
                               save_top_k_peaks=args.save_top_k_peaks,
                               keep_heatmaps=args.belief_propagation)
        core.save()
        core.calibrate_calc(0, core.max_img_id)
        if args.belief_propagation:
            core.run_belief_propagation()
        core.save()
    else:
        core.calibrate_calc(0, core.max_img_id)
        if args.belief_propagation:
            logger.warning('--belief-propagation requires fresh inference; '
                           'ignored because --skip-pose-estimation was set.')
        core.save()

    # Use output_fps if specified, otherwise use core.fps which comes from the input videos
    fps = args.output_fps if args.output_fps is not None else core.fps

    if args.video_2d:
        video.make_pose2d_video(
            core.plot_2d, core.num_images, core.input_folder, core.output_folder,
            fps=fps, start_image_idx=core.start_image_idx
        )

    if args.video_3d:
        video.make_pose3d_video(
            core.get_points3d(),
            core.plot_2d,
            core.num_images,
            core.input_folder,
            core.output_folder,
            fps=fps,
            start_image_idx=core.start_image_idx,
            only_render_legs=core.only_render_legs,
        )

    if args.delete_images:
        core.delete_images()

    return 0


def find_subfolders(path, name):
    """
    Implements a Breadth First Search algorithm to find all subfolders named `name`.

    Using a BFS allows to stop as soon as we find the target subfolder, without listing its content.
    Which is a performance improvement when target subfolders contain hundreds on thousands of images.

    Parameters:
    path: a path where to run the BFS
    name: the target filename we are looking for (typically "images" for an images subfolder)
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


if __name__ == "__main__":
    main()
