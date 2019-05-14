import numpy as np
from PyQt5.QtGui import *
from skimage import feature
from skimage import transform


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


def rgb2qimage(rgb):
    """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError(
            "rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape

    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[..., 0] = rgb[..., 2]
    bgra[..., 1] = rgb[..., 1]
    bgra[..., 2] = rgb[..., 0]
    if rgb.shape[2] == 3:
        bgra[..., 3].fill(255)
        fmt = QImage.Format_RGB32
    else:
        bgra[..., 3] = rgb[..., 3]
        fmt = QImage.Format_ARGB32

    result = QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra


def image_overlay_heatmap(inp, hm):
    # inp = im_to_numpy(inp*255).copy()
    img = np.zeros((inp.shape[0], inp.shape[1], inp.shape[2]), dtype=np.uint8)
    for i in range(3):
        img[:, :, i] = inp[:, :, i]
    img = img.astype(float)
    # hm = to_numpy(hm)
    hm *= 255
    hm = hm.astype(np.uint8)
    hm_resized = transform.resize(hm, [inp.shape[0], inp.shape[1]])
    hm_resized = hm_resized.astype(float)

    img = img.copy() * .3
    hm_color = color_heatmap(hm_resized)
    img += hm_color * .7
    return img.astype(np.uint8)


def hm_to_pred(hm, num_pred=1, image_size=(1, 1)):
    """
    Returns the result in normalized coordinates assumig image size is [1,1]
    """
    pred = []
    if hm.ndim == 2:
        pred = feature.peak_local_max(hm, min_distance=1, threshold_abs=0.01, num_peaks=num_pred)

        if len(pred)==0: # in case hm is zero array
            pred = np.array([0,0])
        if num_pred==1:
            pred = np.squeeze(pred)
        pred = pred / [hm.shape[0], hm.shape[1]]
        pred *= image_size
        pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred[np.newaxis,:]
        tmp = pred[:,1].copy()
        pred[:,1] = pred[:,0]
        pred[:,0] = tmp
        return pred
    else:
        for hm_ in hm:
            pred.append(hm_to_pred(hm_))
    return pred
