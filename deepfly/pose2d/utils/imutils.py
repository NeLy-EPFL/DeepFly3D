from __future__ import absolute_import

import random

import cv2
import numpy as np
import scipy.misc
import torch.nn as nn
from torchvision.transforms import ToPILImage, ToTensor, ColorJitter, RandomAffine

from deepfly.Config import config
from deepfly.pose2d.utils.evaluation import get_preds
from .misc import *


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode="RGB"))


def save_image(img_path, img):
    scipy.misc.imsave(img_path, img)


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = scipy.misc.imresize(img, (oheight, owidth))
    img = im_to_torch(img)
    return img


def random_jitter(im, brightness, contrast, saturation, hue):
    topil = ToPILImage()
    totensor = ToTensor()
    jitter = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )

    im_transform = totensor(jitter(topil(im)))
    return im_transform


def random_rotation(im, hm, degrees, scale=(0.8, 1.2)):
    topil = ToPILImage()
    totensor = ToTensor()
    rot = RandomAffine(degrees=degrees, scale=scale)
    seed = np.random.randint(0, 2 ** 32)

    im_pil = topil(im)
    random.seed(seed)
    im_r = rot(im_pil)
    im = totensor(im_r)

    hm_new = torch.zeros(hm.size())
    for i in range(hm.size(0)):
        hm_pil = topil(hm[i, :, :].unsqueeze(0))
        random.seed(seed)
        hm_r = rot((hm_pil))
        hm_t = totensor(hm_r).squeeze()
        hm_new[i, :, :] = hm_t
    return im, hm_new


# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================


def gaussian(shape=(7, 7), sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return to_torch(h).float()


def draw_labelmap(img, pt, sigma, type="Gaussian"):
    ## Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    return to_torch(img)


# =============================================================================
# Helpful display functions
# =============================================================================


def gauss(x, a, b, c, d=0):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d


def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, 0.5, 0.6, 0.2) + gauss(x, 1, 0.8, 0.3)
    color[:, :, 1] = gauss(x, 1, 0.5, 0.3)
    color[:, :, 2] = gauss(x, 1, 0.2, 0.3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def imshow(img):
    npimg = im_to_numpy(img * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis("off")


def show_joints(img, pts):
    imshow(img)
    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], "yo")
    plt.axis("off")


def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp * 0.5 + color_heatmap(target[n, p, :, :]) * 0.5
            out = torch.cat((out, tgt), 2)

        imshow(out)
        plt.show()


def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[: img.shape[0], : img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float) / 255
        out_img = inp_small.copy() * 0.3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * 0.7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[
            row_offset : row_offset + size, col_offset : col_offset + size
        ] = out_img

    return full_img


def image_overlay_heatmap(inp, hm):
    inp = im_to_numpy(inp * 255).copy()
    img = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2]))
    for i in range(3):
        img[:, :, i] = inp[:, :, i]

    hm = to_numpy(hm)
    hm_resized = scipy.misc.imresize(hm, [inp.shape[0], inp.shape[1], 3])
    hm_resized = hm_resized.astype(float) / 255

    img = img.copy() * 0.3
    hm_color = color_heatmap(hm_resized)
    img += hm_color * 0.7
    return img


def image_overlay_pose(inp, pts, pts_max_value, joint_idx=None, joint_draw=None):
    if joint_draw is None:
        joint_draw = np.arange(0, config["skeleton"].num_joints)

    inp = im_to_numpy(inp * 255).copy()
    pts = to_numpy(pts)
    pts = (pts / np.array([pts_max_value[1], pts_max_value[0]])) * np.array(
        [inp.shape[1], inp.shape[0]]
    )
    pts = pts.astype(np.int)

    colors = config["skeleton"].colors
    for joint_id in range(pts.shape[0]):
        if joint_id in joint_draw:
            color = colors[config["skeleton"].get_limb_id(joint_id)]
            r = 5
            # if not (pts[joint_id, 0] < 5 and pts[joint_id, 1] < 5):
            cv2.circle(inp, (pts[joint_id, 0], pts[joint_id, 1]), r, color, -1)

    for bone in config["skeleton"].bones:
        if (
            bone[0] < pts.shape[0]
            and bone[1] < pts.shape[0]
            and bone[0] in joint_draw
            and bone[1] in joint_draw
        ):  # \
            # and not ((pts[bone[0], 0] < 5 and pts[bone[0], 1] < 5 and pts[bone[1], 0] < 5 and pts[bone[1], 1] < 5)):
            color = colors[int(bone[0] / 5)]
            cv2.line(
                inp,
                (pts[bone[0]][0], pts[bone[0]][1]),
                (pts[bone[1]][0], pts[bone[1]][1]),
                color=color,
                thickness=1,
            )

    return inp


def batch_with_heatmap(
    inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]), num_rows=2, parts_to_show=None
):
    batch_img = []
    # for n in range(min(inputs.size(0), 4)):
    for n in range(min(inputs.size(0), 18)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(
                inp.clamp(0, 1),
                outputs[n],
                num_rows=num_rows,
                parts_to_show=parts_to_show,
            )
        )
    return np.concatenate(batch_img)


def drosophila_image_overlay(
    inputs, target, hm_res, batch_size, train_joints, img_id=None
):
    gt = get_preds(target)
    from skimage.transform import resize

    img_overlay_list = list()
    img_overlay_heatmap_list = list()
    for img_idx in range(batch_size):
        img_overlay_list.append(
            image_overlay_pose(
                inputs[img_idx],
                gt[img_idx],
                joint_idx=[],
                joint_draw=train_joints,
                pts_max_value=hm_res,
            )
        )

        # img_overlay_list.append(plot_drosophila_2d(inputs[img_idx], gt[img_idx]))

        for j in train_joints:
            # to speed-up the visualization, we decrease the size of the images
            m = nn.AvgPool2d((12, 8), stride=(8, 4))
            hm = image_overlay_heatmap(m(inputs[img_idx]), target[img_idx][j])
            img_overlay_heatmap_list.append(hm)

    img_overlay_pose_stack = np.hstack(img_overlay_list)
    array = np.stack(img_overlay_heatmap_list, axis=0)
    nrows, ncols = 3, len(train_joints)
    nindex, height, width, intensity = array.shape
    img_overlay_hm_stack = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )

    img_overlay_hm_stack = resize(
        img_overlay_hm_stack,
        (img_overlay_pose_stack.shape[0], img_overlay_pose_stack.shape[1]),
    )
    img = img_overlay_stack = np.vstack([img_overlay_pose_stack, img_overlay_hm_stack])
    if img_id is not None:
        img = cv2.putText(
            img,
            "{}".format(img_id),
            (img.shape[1] - 150, 50),
            cv2.FONT_HERSHEY_PLAIN,
            4,
            (255, 0, 0),
            5,
            cv2.LINE_AA,
        )
    return img_overlay_stack
