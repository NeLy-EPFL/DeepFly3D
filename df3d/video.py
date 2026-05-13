import os.path
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import packaging.version
from tqdm import tqdm

from df3d.config import config
from df3d.plot_util import plot_drosophila_3d
import df3d.logger as logger

img3d_dpi = 100  # this is the dpi for one image on the 3d video's grid
img3d_aspect = (2, 2)  # this is the aspect ration for one image on the 3d video's grid
img2d_aspect = (2, 1)  # this is the aspect ration for one image on the 3d video's grid
video_width = 5000  # total width of the 2d and 3d videos
default_fps = 30


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
    # Here we create a generator (keyword "yield")
    def imgs_generator():
        def stack(img_id):
            plot = lambda c, i: plot_2d(c, i, smooth=True)
            row1 = np.hstack([plot(cam_id, img_id) for cam_id in [0, 1, 2]])
            row2 = np.hstack([plot(cam_id, img_id) for cam_id in [4, 5, 6]])
            return np.vstack([row1, row2])

        for img_id in range(start_image_idx, start_image_idx + num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images,
    # just like for an iterator
    generator = imgs_generator()

    video_name = 'video_pose2d_' + input_folder.replace('/', '_') + '.mp4'
    video_path = os.path.join(input_folder, output_folder, video_name)
    _make_video(video_path, generator, fps=fps)


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
    draw_joints = _leg_only_draw_joints() if only_render_legs else None

    # Create one figure + per-bone Line3D artists per camera once, then
    # reuse them across frames; per-frame we only update their data.
    cam_ids_3d = (4, 5, 6)
    bone_lines = {cam_id: _make_3d_canvas(cam_id, num_joints=points3d.shape[1],
                                          draw_joints=draw_joints)
                  for cam_id in cam_ids_3d}

    def imgs_generator():
        def stack(img_id):
            row1 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id) for cam_id in (0, 1, 2)])
            row2 = np.hstack([_compute_2d_img(plot_2d, img_id, cam_id) for cam_id in cam_ids_3d])
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
    _make_video(video_path, generator, fps=fps)


def _make_video(video_path, imgs, fps=default_fps):
    """Code used to generate a video using cv2.

    Parameters:
    video_path: a path ending with .mp4, for instance: "/results/pose2d.mp4"
    imgs: an iterable or generator with the images to turn into a video
    """
    if fps is None:
        fps = default_fps

    first_frame = next(imgs)
    imgs = itertools.chain([first_frame], imgs)

    shape = int(first_frame.shape[1]), int(first_frame.shape[0])
    logger.debug('Saving video to: ' + video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_shape = _resize(current_shape=shape, new_width=video_width)
    logger.debug('Video size is: {}'.format(output_shape))
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape)

    progress_bar = tqdm if logger.info_enabled() else lambda x: x
    for img in progress_bar(imgs):
        resized = cv2.resize(img, output_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)

    video_writer.release()
    logger.info('Video created at {}\n'.format(video_path))


def _resize(current_shape, new_width):
    width, height = current_shape
    ratio = new_width / width
    return (int(width * ratio), int(height * ratio))


def _compute_2d_img(plot_2d, img_id, cam_id, reprojection=True):
    """Uses plot_2d to generate an image and resizes it using cv2.

    Returns:
    A numpy array containing the resized image.
    """
    img = plot_2d(cam_id, img_id, smooth=True, reprojection=reprojection)
    img = cv2.resize(img, (img2d_aspect[0]*img3d_dpi, img2d_aspect[1]*img3d_dpi))
    return img


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


def _make_3d_canvas(cam_id, num_joints, lim=2, draw_joints=None):
    """
    Create a figure + axes + per-bone Line3D artists for repeated 3d
    rendering of one camera. Returns the list of Line3D artists; the
    parent figure is reachable via `lines[0].axes.figure`. Pass the
    list to `_compute_3d_img(..., bone_lines=lines)` to reuse the
    canvas every frame instead of rebuilding it.
    """
    _setup_3d_style()
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout(pad=0)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return plot_drosophila_3d(
        ax, np.zeros((num_joints, 3)), cam_id=cam_id, lim=lim,
        thickness=np.ones(num_joints) * 1.5,
        draw_joints=draw_joints,
    )


def _compute_3d_img(points3d, img_id, cam_id, bone_lines=None, draw_joints=None):
    """Generates the 3D image showing joints positions based on points3d.

    Parameters
    ----------
    bone_lines : list of Line3D, optional
        Pre-built bone artists from `_make_3d_canvas`. If given, reuse
        their figure and only update bone positions — much faster than
        rebuilding the figure every frame. Caller is responsible for
        closing the figure (`plt.close(bone_lines[0].axes.figure)`).
        If None, a fresh figure is created and closed before returning.

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
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
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
