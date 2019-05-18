from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import skeleton


# this is important else throws error

def plot_drosophila_2d(pts=None, draw_joints=None, img=None, colors=None, thickness=None,
                       draw_limbs=None, circle_color=None, draw_order=None, zorder=None):
    if colors is None:
        colors = skeleton.colors
    if thickness is None:
        thickness = [2] * 10
    if draw_joints is None:
        draw_joints = np.arange(skeleton.num_joints)
    if draw_limbs is None:
        draw_limbs = np.arange(skeleton.num_limbs)
    if draw_order is None:
        draw_order = np.arange(skeleton.num_limbs)
        draw_order = np.intersect1d(draw_order, draw_limbs)
    if zorder is None:
        zorder = np.ones((skeleton.num_joints))


    #for joint_id in range(pts.shape[0]):
    for joint_id in np.argsort(zorder):
        limb_id = skeleton.get_limb_id(joint_id)
        if (pts[joint_id, 0] == 0 and pts[
            joint_id, 1] == 0) or limb_id not in draw_limbs or joint_id not in draw_joints:
            continue

        color = colors[limb_id]
        r = 5 if joint_id != skeleton.num_joints - 1 and joint_id != ((skeleton.num_joints // 2) - 1) else 8
        cv2.circle(img, (pts[joint_id, 0], pts[joint_id, 1]), r, color, -1)

        # TODO replace this with skeleton.bones
        if (not skeleton.is_tarsus_tip(joint_id)) and (not skeleton.is_antenna(
                joint_id)) and (joint_id != skeleton.num_joints - 1) and (
                joint_id != (skeleton.num_joints // 2 - 1)) and (not (
                pts[joint_id + 1, 0] == 0 and pts[joint_id + 1, 1] == 0)):
            cv2.line(img, (pts[joint_id][0], pts[joint_id][1]), (pts[joint_id + 1][0], pts[joint_id + 1][1]),
                     color=color,
                     thickness=thickness[limb_id])

    if circle_color is not None:
        img = cv2.circle(img=img, center=(img.shape[1]-20, 20), radius=10, color=circle_color, thickness=-1)

    return img


def plot_drosophila_heatmap(image=None, hm=None, concat=False, scale=1):
    '''
    concat: Whether to return a single image or njoints images concatenated
    '''
    assert (image is not None and hm is not None)
    inp = image
    if hm.ndim == 3 and not concat:
        # then sum the joint heatmaps
        hm = hm.sum(axis=0)
    if concat is False:
        img = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2]))
        for i in range(3):
            img[:, :, i] = inp[:, :, i]
        # scale to make it faster
        if scale != 1:
            img = scipy.misc.imresize(img, [int(img.shape[0] / scale), int(img.shape[1] / scale), img.shape[2]])

        hm_resized = scipy.misc.imresize(hm, [img.shape[0], img.shape[1], 3])
        hm_resized = hm_resized.astype(float) / 255

        img = img.copy() * .3
        hm_color = color_heatmap(hm_resized)
        img += hm_color * .7
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
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def points3d_to_zorder(points3d):
    assert(points3d.ndim==2)
    z_list =  points3d[:,2].ravel()
    zorder = np.argsort(z_list)
    return zorder


def Rt_points3d(R, tvec, points3d):
    return np.matmul(R, points3d) + tvec

def plot_drosophila_3d(ax_3d, points3d, cam_id, bones=skeleton.bones, ang=None, draw_joints=None, colors=None, zorder=None, thickness=None):
    points3d = np.array(points3d)
    if draw_joints is None:
        draw_joints = np.arange(skeleton.num_joints)
    if colors is None:
        colors = skeleton.colors
    colors_tmp = ['#%02x%02x%02x' % c for c in colors]
    if zorder is None:
        zorder = np.ones((points3d.shape[0]))
    colors = []
    for i in range(skeleton.num_joints):
        colors.append(colors_tmp[skeleton.get_limb_id(i)])
    if thickness is None:
        thickness = np.ones((points3d.shape[0]))*3
    colors = np.array(colors)

    # ax_3d.xaxis.pane.fill = False
    # ax_3d.yaxis.pane.fill = False
    # ax_3d.zaxis.pane.fill = False

    # ax_3d.set_aspect('equal')
    # ax_3d.axis('off')
    # ax_3d.axes.get_xaxis().set_visible(False)

    #ax_3d.axes.xaxis.set_ticklabels([])
    #ax_3d.axes.yaxis.set_ticklabels([])
    #ax_3d.axes.zaxis.set_ticklabels([])

    # ax_3d.xaxis.pane.set_edgecolor('w')
    # ax_3d.yaxis.pane.set_edgecolor('w')
    # ax_3d.zaxis.pane.set_edgecolor('w')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.w_xaxis.set_pane_color(white)
    ax_3d.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax_3d.w_xaxis.line.set_color(white)
    ax_3d.w_yaxis.line.set_color(white)
    ax_3d.w_zaxis.line.set_color(white)

    # ax_3d.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if ang is not None:
        ax_3d.view_init(ax_3d.elev, ang)
    else:
        if cam_id < 3:
            ax_3d.view_init(ax_3d.elev, -60 + 30 * cam_id)
        else:
            ax_3d.view_init(ax_3d.elev, -60 + 45 * cam_id)

    max_range = 2
    mid_x = 0
    mid_y = 0
    mid_z = 0
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    for j in range(skeleton.num_joints):
        if skeleton.is_stripe(j) and skeleton.is_joint_visible_left(j):
            points3d[j] = (points3d[j] + points3d[j+(skeleton.num_joints // 2)]) / 2
            points3d[j + skeleton.num_joints // 2] = points3d[j]

    points3d_draw = points3d.copy()
    colors_draw = colors

    '''
    for j in draw_joints:
        print(j, zorder[j])
        ax_3d.scatter3D(points3d_draw[j, 0], points3d_draw[j, 1], points3d_draw[j, 2],
                        c=colors_draw[j], s=100, zorder=zorder[j])
    '''


    for bone in bones:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(points3d[bone, 0], points3d[bone, 1], points3d[bone, 2], c=colors[bone[0]],
                       linewidth=thickness[skeleton.get_limb_id(bone[0])], zorder=zorder[bone[0]])
    for bone in skeleton.bones3d:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(points3d[bone, 0], points3d[bone, 1], points3d[bone, 2], c=colors[bone[0]],
                       linewidth=5, zorder=[bone[0]])



def normalize_pose_3d(points3d, normalize_length=False, normalize_median=True):
    if normalize_median:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    if normalize_length:
        length = [0.005, 0.01, 0.01, 0.01, 0.01]
        points3d = points3d.reshape(-1, 15, 3)
        for idx in range(points3d.shape[0]):
            print(idx)
            for j_idx in range(points3d[idx].shape[0]):
                if j_idx % 5 == 4:  # then tarsus-tip
                    continue
                diff = points3d[idx, j_idx + 1, :] - points3d[idx, j_idx, :]
                diff_norm = (diff / np.linalg.norm(diff)) * length[j_idx % 5]
                points3d[idx, j_idx + 1, :] = points3d[idx, j_idx, :] + diff_norm
                next_tarsus_tip = (j_idx - (j_idx % 5)) + 5
                points3d[idx, j_idx + 2:next_tarsus_tip, :] += (diff_norm - diff)
    return points3d
