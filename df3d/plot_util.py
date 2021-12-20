from __future__ import print_function

import cv2
import numpy as np
import scipy
from PIL import Image

from df3d.config import config


def plot_drosophila_2d(
        pts=None,
        draw_joints=None,
        img=None,
        colors=None,
        thickness=None,
        draw_limbs=None,
        circle_color=None,
        draw_order=None,
        zorder=None,
        r_list=None
):
    if colors is None:
        colors = config["skeleton"].colors
    if thickness is None:
        thickness = [config["line_thickness"]] * config["skeleton"].num_limbs
    if draw_joints is None:
        draw_joints = np.arange(config["skeleton"].num_joints)
    if draw_limbs is None:
        draw_limbs = np.arange(config["skeleton"].num_limbs)
    if draw_order is None:
        draw_order = np.arange(config["skeleton"].num_limbs)
        draw_order = np.intersect1d(draw_order, draw_limbs)
    if zorder is None:
        zorder = np.arange(config["skeleton"].num_joints)
    if r_list is None:
        r_list = [config["scatter_r"]]*config["skeleton"].num_joints

    # for joint_id in range(pts.shape[0]):
    for joint_id in np.argsort(zorder):
        limb_id = config["skeleton"].get_limb_id(joint_id)
        if (
                (pts[joint_id, 0] == 0 and pts[joint_id, 1] == 0)
                or limb_id not in draw_limbs
                or joint_id not in draw_joints
        ):
            continue

        color = colors[limb_id]

        cv2.circle(img, (pts[joint_id, 0], pts[joint_id, 1]), r_list[joint_id], color, thickness=-1)

        for bone in config["bones"]:
            if bone[0] == joint_id:
                cv2.line(
                    img,
                    (pts[bone[0]][0], pts[bone[0]][1]),
                    (pts[bone[1]][0], pts[bone[1]][1]),
                    color=color,
                    thickness=thickness[limb_id],
                )

    if circle_color is not None:
        img = cv2.circle(
            img=img,
            center=(img.shape[1] - 20, 20),
            radius=10,
            color=circle_color,
            thickness=-1,
        )

    return img


def plot_drosophila_heatmap(image=None, hm=None, concat=False, scale=1):
    """
    concat: Whether to return a single image or njoints images concatenated
    """
    assert image is not None and hm is not None
    inp = image
    if hm.ndim == 3 and not concat:
        hm = hm.sum(axis=0)
    if concat is False:
        img = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2]))
        for i in range(3):
            img[:, :, i] = inp[:, :, i]
        if scale != 1:
            img = cv2.resize(img, (int(img.shape[1] / (scale / 2)), int(img.shape[0] / (scale / 2))))

        hm_resized = cv2.resize(hm, (int(img.shape[1]), int(img.shape[0])))
        #hm_resized = np.array(Image.fromarray(hm).resize([int(img.shape[1]), int(img.shape[0])]))
        hm_resized = hm_resized.astype(float)

        img = img.copy() * 0.3
        hm_color = color_heatmap(hm_resized)
        img += hm_color * 0.7
        return img.astype(np.uint8)
    elif concat:
        concat_list = []
        for idx, hm_ in enumerate(hm):
            concat_list.append(plot_drosophila_heatmap(hm_, concat=False))
        return np.hstack(concat_list)


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def color_heatmap(x):
    # x = to_numpy(x)
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, 0.5, 0.6, 0.2) + gauss(x, 1, 0.8, 0.3)
    color[:, :, 1] = gauss(x, 1, 0.5, 0.3)
    color[:, :, 2] = gauss(x, 1, 0.2, 0.3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def points3d_to_zorder(points3d):
    assert points3d.ndim == 2
    z_list = points3d[:, 2].ravel()
    zorder = np.argsort(z_list)
    return zorder


def Rt_points3d(R, tvec, points3d):
    return np.matmul(R, points3d) + tvec


def rotate_points3d(pts_t):
    tmp = pts_t[:, :, 1].copy()
    pts_t[:, :, 1] = pts_t[:, :, 2].copy()
    pts_t[:, :, 2] = tmp
    pts_t[:, :, 2] *= -1
    pts_t[:, :, 1] *= -1

    return pts_t


def plot_drosophila_3d(
        ax_3d,
        points3d,
        cam_id,
        bones=config["bones"],
        ang=None,
        draw_joints=None,
        colors=None,
        zorder=None,
        thickness=None,
        lim=None,
        scatter=False,
        axis=False
):
    points3d = np.array(points3d)
    if draw_joints is None:
        draw_joints = np.arange(config["skeleton"].num_joints)
    if colors is None:
        colors = config["skeleton"].colors
    colors_tmp = ["#%02x%02x%02x" % c for c in colors]
    if zorder is None:
        zorder = config["skeleton"].get_zorder(cam_id)
    if thickness is None:
        thickness = np.ones((points3d.shape[0])) * 3

    colors = []
    for i in range(config["skeleton"].num_joints):
        colors.append(colors_tmp[config["skeleton"].get_limb_id(i)])
    colors = np.array(colors)

    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.w_xaxis.set_pane_color(white)
    ax_3d.w_yaxis.set_pane_color(white)

    ax_3d.w_xaxis.line.set_color(white)
    ax_3d.w_yaxis.line.set_color(white)
    ax_3d.w_zaxis.line.set_color(white)

    if ang is not None:
        ax_3d.view_init(ax_3d.elev, ang)
    else:
        if cam_id < 3:
            ax_3d.view_init(ax_3d.elev, -60 + 30 * cam_id)
        else:
            ax_3d.view_init(ax_3d.elev, -60 + 45 * cam_id)

    if lim:
        max_range = lim
        mid_x = 0
        mid_y = 0
        mid_z = 0
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    if "fly" in config["name"]:
        for j in range(config["skeleton"].num_joints):
            if config["skeleton"].is_tracked_point(j, config["skeleton"].Tracked.STRIPE) and config[
                "skeleton"].is_joint_visible_left(j):
                points3d[j] = (points3d[j] + points3d[j + (config["skeleton"].num_joints // 2)]) / 2
                points3d[j + config["skeleton"].num_joints // 2] = points3d[j]

    if scatter:
        for j in draw_joints:
            ax_3d.scatter(points3d[j, 0],
                points3d[j, 1],
                points3d[j, 2],
                c=colors[j],
                linewidth=thickness[config["skeleton"].get_limb_id(j)],
                zorder=zorder[j])

    for bone in bones:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(
                points3d[bone, 0],
                points3d[bone, 1],
                points3d[bone, 2],
                c=colors[bone[0]],
                linewidth=thickness[config["skeleton"].get_limb_id(bone[0])],
                zorder=zorder[bone[0]],
            )
    for bone in config["skeleton"].bones3d:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(
                points3d[bone, 0],
                points3d[bone, 1],
                points3d[bone, 2],
                c=colors[bone[0]],
                linewidth=5,
                zorder=zorder[bone[0]],
            )


def normalize_pose_3d(points3d, normalize_length=False, normalize_median=True, rotate=False):
    if normalize_median:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    if rotate:
        points3d = rotate_points3d(points3d)

    return points3d
