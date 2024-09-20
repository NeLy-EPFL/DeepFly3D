import cv2
import numpy as np

from df3d.config import config


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
    draw_joints=None,
    thickness=None,
    lim=None,
):
    points3d = np.array(points3d)
    draw_joints = np.arange(config["skeleton"].num_joints)
    colors = config["skeleton"].colors
    colors_tmp = ["#%02x%02x%02x" % c for c in colors]
    zorder = config["skeleton"].get_zorder(cam_id)

    colors = []
    for i in range(config["skeleton"].num_joints):
        colors.append(colors_tmp[config["skeleton"].get_limb_id(i)])
    colors = np.array(colors)

    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.xaxis.set_pane_color(white)
    ax_3d.yaxis.set_pane_color(white)

    ax_3d.xaxis.line.set_color(white)
    ax_3d.yaxis.line.set_color(white)
    ax_3d.zaxis.line.set_color(white)

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
            if config["skeleton"].is_tracked_point(
                j, config["skeleton"].Tracked.STRIPE
            ) and config["skeleton"].is_joint_visible_left(j):
                points3d[j] = (
                    points3d[j] + points3d[j + (config["skeleton"].num_joints // 2)]
                ) / 2
                points3d[j + config["skeleton"].num_joints // 2] = points3d[j]

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


def normalize_pose_3d(points3d, normalize_median=True, rotate=False):
    if normalize_median:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    if rotate:
        points3d = rotate_points3d(points3d)

    return points3d
