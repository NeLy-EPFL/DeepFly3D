import os.path
import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
from tqdm import tqdm

from deepfly.GUI.util.plot_util import plot_drosophila_3d
import deepfly.logger as logger


img3d_dpi = 100  # this is the dpi for one image on the 3d video's grid
img3d_aspect = (2, 2)  # this is the aspect ration for one image on the 3d video's grid
img2d_aspect = (2, 1)  # this is the aspect ration for one image on the 3d video's grid
video_width = 500  # total width of the 2d and 3d videos


def _make_pose2d_video(args):
    """ Creates pose2d estimation videos """
    # Here we create a generator (keyword "yield")
    def imgs_generator():
        def stack(img_id):
            plot = lambda c, i: args.plot_2d(c, i, smooth=True)
            row1 = np.hstack([plot(cam_id, img_id) for cam_id in [0, 1, 2]])
            row2 = np.hstack([plot(cam_id, img_id) for cam_id in [4, 5, 6]])
            return np.vstack([row1, row2])

        for img_id in range(args.num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images,
    # just like for an iterator
    generator = imgs_generator()

    video_name = 'video_pose2d_' + args.input_folder.replace('/', '_') + '.mp4'
    _make_video(args, video_name, generator)


def _make_pose3d_video(core):
    # Here we create a generator (keyword "yield")
    points3d = core.get_points3d()
    
    def imgs_generator():
        def stack(img_id):
            row1 = np.hstack([_compute_2d_img(core, img_id, cam_id) for cam_id in (0, 1, 2)])
            row2 = np.hstack([_compute_2d_img(core, img_id, cam_id) for cam_id in (4, 5, 6)])
            row3 = np.hstack([_compute_3d_img(points3d, img_id, cam_id) for cam_id in (4, 5, 6)])
            img = np.vstack([row1, row2, row3])
            return img

        for img_id in range(core.num_images):
            yield stack(img_id)

    # We can call next(generator) on this instance to get the images, just like for an iterator
    generator = imgs_generator()
    video_name = 'video_pose3d_' + core.input_folder.replace('/', '_') + '.mp4'
    _make_video(core, video_name, generator)


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
    logger.debug('Saving video to: ' + video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
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
    ratio = new_width / width;
    return (int(width * ratio), int(height * ratio))


def _compute_2d_img(core, img_id, cam_id):
    img = core.plot_2d(cam_id, img_id, smooth=True)
    img = cv2.resize(img, (img2d_aspect[0]*img3d_dpi, img2d_aspect[1]*img3d_dpi))
    return img


def _compute_3d_img(points3d, img_id, cam_id):
    import numpy as np

    plt.style.use('dark_background')
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
    fig.tight_layout(pad=0)

    ax3d = Axes3D(fig)
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
        thickness=np.ones((points3d.shape[1])) * 1.5)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data