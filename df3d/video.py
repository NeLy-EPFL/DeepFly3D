import os.path
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import npimage
import packaging.version
from tqdm import tqdm

from df3d.config import config
from df3d.os_util import pick_image_path
from df3d.plot_util import plot_drosophila_3d
import df3d.logger as logger

output_video_downsampling = 2  # output cell width = source_width / this
default_fps = 30


def _cell_width_from_source(src_w, downsampling=None):
    """
    Per-cell width in px (rounded to an even count — libx264 needs even dims).
    """
    if downsampling is None:
        downsampling = output_video_downsampling
    return 2 * int(round(src_w / downsampling / 2))


def _peek_source_width(plot_2d, start_image_idx):
    """
    Width of the rendered 2D plot for cam 0 — same as source frame width.
    Used to size all output cells before iteration begins.
    """
    return plot_2d(0, start_image_idx, smooth=True, reprojection=False).shape[1]


def make_pose2d_video(plot_2d, num_images, input_folder,
                      output_folder, fps=default_fps, start_image_idx=0):
    """Creates pose2d estimation videos and writes it to output_folder.

    Parameters:
    plot_2d: a function callback which generates an image as a numpy array
    num_images: the number of images to use for the video
    input_folder: input folder containing the images
    output_folder: output folder where to write the video.
    start_image_idx: the index of the first image to include in the video (default: 0)
    """
    cell_width = _cell_width_from_source(_peek_source_width(plot_2d, start_image_idx))

    # Here we create a generator (keyword "yield")
    def imgs_generator():
        def stack(img_id):
            row1 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id, cell_width,
                                              reprojection=False)
                              for cam_id in (0, 1, 2)])
            row2 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id, cell_width,
                                              reprojection=False)
                              for cam_id in (4, 5, 6)])
            return np.vstack([row1, row2])

        for img_id in range(start_image_idx, start_image_idx + num_images):
            yield stack(img_id)

    generator = imgs_generator()

    video_name = 'video_pose2d_' + input_folder.replace('/', '_') + '.mp4'
    video_path = os.path.join(input_folder, output_folder, video_name)
    source_extension = pick_image_path(input_folder).rsplit('.', 1)[-1].lower()
    _make_video(video_path, generator, fps=fps,
                desc=f'Rendering 2D pose video from {source_extension}',
                total=num_images)


def make_pose3d_video(points3d, plot_2d, num_images, input_folder,
                      output_folder, fps=default_fps, start_image_idx=0,
                      only_render_legs=False):
    """Creates pose3d estimation videos and writes it to output_folder.

    Parameters:
    points3d: estimated 3D joints positions.
    plot_2d: a function callback which generates an image as a numpy array
    num_images: the number of images to use for the video
    input_folder: input folder containing the images
    output_folder: output folder where to write the video.
    start_image_idx: the index of the first image to include in the video (default: 0)
    only_render_legs: if True, omit antenna and stripe joints/bones from the 3D plots.
    """
    cell_width = _cell_width_from_source(_peek_source_width(plot_2d, start_image_idx))
    draw_joints = _leg_only_draw_joints() if only_render_legs else None

    # Create one figure + per-bone Line3D artists per camera once, then
    # reuse them across frames; per-frame we only update their data.
    cam_ids_3d = (4, 5, 6)
    bone_lines = {cam_id: _make_3d_canvas(cam_id, num_joints=points3d.shape[1],
                                          cell_width=cell_width,
                                          draw_joints=draw_joints)
                  for cam_id in cam_ids_3d}

    def imgs_generator():
        def stack(img_id):
            row1 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id, cell_width)
                              for cam_id in (0, 1, 2)])
            row2 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id, cell_width)
                              for cam_id in cam_ids_3d])
            row3 = np.hstack([_compute_3d_img(points3d, img_id, cam_id,
                                              bone_lines=bone_lines[cam_id],
                                              draw_joints=draw_joints)
                              for cam_id in cam_ids_3d])
            return np.vstack([row1, row2, row3])

        try:
            for img_id in range(start_image_idx, start_image_idx + num_images):
                yield stack(img_id)
        finally:
            for lines in bone_lines.values():
                plt.close(lines[0].axes.figure)

    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()
    video_name = 'video_pose3d_' + input_folder.replace('/', '_') + '.mp4'
    video_path = os.path.join(input_folder, output_folder, video_name)
    source_extension = pick_image_path(input_folder).rsplit('.', 1)[-1].lower()
    _make_video(video_path, generator, fps=fps,
                desc=f'Rendering 3D pose video from {source_extension}',
                total=num_images)


def _make_video(video_path, imgs, fps=default_fps, desc=None, total=None):
    """
    Write `imgs` (an iterable of equal-shape frames) to an mp4 at
    `video_path`. Output dimensions are taken from the first frame.
    """
    if fps is None:
        fps = default_fps

    first_frame = next(imgs)
    imgs = itertools.chain([first_frame], imgs)

    height, width = first_frame.shape[:2]
    logger.debug('Saving video to: ' + video_path)
    logger.debug(f'Video size is: ({width}, {height})')
    with npimage.VideoWriter(video_path, framerate=fps, crf=18,
                             codec='libx264', compression_speed='veryfast',
                             overwrite=True) as video_writer:
        if logger.info_enabled():
            imgs = tqdm(imgs, desc=desc, total=total)
        for img in imgs:
            video_writer.write(img)
    logger.info('Video created at {}\n'.format(video_path))


def _compute_2d_img(plot_2d, img_id, cam_id, cell_width, reprojection=True):
    """
    Render one 2d camera frame and resize it to `cell_width` wide,
    preserving the source aspect ratio.
    """
    img = plot_2d(cam_id, img_id, smooth=True, reprojection=reprojection)
    src_h, src_w = img.shape[:2]
    # Round to an even pixel count so the assembled video has even
    # height (libx264 requires even dimensions).
    target_height = 2 * int(round(cell_width * src_h / src_w / 2))
    return cv2.resize(img, (cell_width, target_height))


def _setup_3d_style():
    """Module-global matplotlib config for 3d pose plots."""
    plt.style.use('dark_background')
    if (packaging.version.Version(matplotlib.__version__)
            >= packaging.version.Version("3.9")):
        # Versions of matplotlib 3.9+ produce a slightly zoomed in 3d
        # plot compared to older versions (#55). Restore the old framing.
        plt.rcParams['axes3d.automargin'] = True


def _leg_only_draw_joints():
    """Return joint indices that aren't ANTENNA or STRIPE."""
    skeleton = config["skeleton"]
    Tracked = skeleton.Tracked
    return np.array([
        j for j in range(skeleton.num_joints)
        if not (skeleton.is_tracked_point(j, Tracked.ANTENNA)
                or skeleton.is_tracked_point(j, Tracked.STRIPE))
    ])


def _make_3d_figure(cell_width):
    """
    Build a square matplotlib figure that rasterizes at exactly
    `cell_width` px per side. matplotlib pins canvas size at
    figsize_inches * dpi, so we pick figsize=(2,2) and set dpi
    accordingly — these inch units are matplotlib's API, not exposed
    to df3d users.
    """
    return plt.figure(figsize=(2, 2), dpi=cell_width / 2)


def _make_3d_canvas(cam_id, num_joints, cell_width, lim=2, draw_joints=None):
    """
    Create a figure + axes + per-bone Line3D artists for repeated 3d
    rendering of one camera. Returns the list of Line3D artists; the
    parent figure is reachable via `lines[0].axes.figure`. Pass the
    list to `_compute_3d_img(..., bone_lines=lines)` to reuse the
    canvas every frame instead of rebuilding it.
    """
    _setup_3d_style()
    fig = _make_3d_figure(cell_width)
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout(pad=0)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return plot_drosophila_3d(
        ax, np.zeros((num_joints, 3)), cam_id=cam_id, lim=lim,
        thickness=np.ones(num_joints) * 1.5,
        draw_joints=draw_joints,
    )


def _compute_3d_img(points3d, img_id, cam_id, bone_lines=None,
                    draw_joints=None, cell_width=480):
    """Generates the 3D image showing joints positions based on points3d.

    Parameters
    ----------
    bone_lines : list of Line3D, optional
        Pre-built bone artists from `_make_3d_canvas`. If given, reuse
        their figure and only update bone positions — much faster than
        rebuilding the figure every frame. Caller is responsible for
        closing the figure (`plt.close(bone_lines[0].axes.figure)`).
        If None, a fresh `cell_width`-sized figure is created and
        closed before returning.
    cell_width : int
        Square output size in px. Ignored when `bone_lines` is given
        (its figure is reused at whatever size it was built with).

    Returns:
    A numpy array containing the resulting 3D image projected on 2D.
    """
    if bone_lines is not None:
        plot_drosophila_3d(None, points3d[img_id].copy(), cam_id=cam_id,
                           bone_lines=bone_lines, draw_joints=draw_joints)
        fig = bone_lines[0].axes.figure
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba(),
                        dtype=np.uint8)[:, :, :3]

    _setup_3d_style()
    fig = _make_3d_figure(cell_width)
    ax3d = fig.add_subplot(111, projection='3d')
    fig.tight_layout(pad=0)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    plot_drosophila_3d(
        ax3d,
        points3d[img_id].copy(),
        cam_id=cam_id,
        lim=2,
        thickness=np.ones((points3d.shape[1])) * 1.5,
        draw_joints=draw_joints)

    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    return data
