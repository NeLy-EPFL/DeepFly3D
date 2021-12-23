import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from df3d.config import config


class LowPassFilter(object):
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = (
            0.0 if prev_x is None else (x - prev_x) * self.__freq
        )  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


def filter_batch(pts, filter_indices=None, config_oneuro=None, freq=None):
    from df3d.config import config

    assert pts.shape[-1] == 2 or pts.shape[-1] == 3
    if filter_indices is None:
        filter_indices = np.arange(config["skeleton"].num_joints)
    if config_oneuro is None:
        config_oneuro = {
            "freq": 100,  # Hz
            "mincutoff": 0.1,  # FIXME
            "beta": 2.0,  # FIXME
            "dcutoff": 1.0,  # this one should be ok
        }
    if freq is not None:
        config_oneuro["freq"] = freq

    f = [
        [OneEuroFilter(**config_oneuro) for j in range(pts.shape[-1])]
        for i in range(config["skeleton"].num_joints)
    ]
    timestamp = 0.0  # seconds
    pts_after = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            if j in filter_indices:
                pts_after[i, j, 0] = f[j][0](pts[i, j, 0], (i + 1) * 0.1)
                pts_after[i, j, 1] = f[j][1](pts[i, j, 1], (i + 1) * 0.1)
                pts_after[i, j, 2] = f[j][2](pts[i, j, 2], (i + 1) * 0.1)

            else:
                pts_after[i, j] = pts[i, j]
    return pts_after


def filter_batch_2d(pts, filter_indices=None, config=None, freq=None):
    assert pts.shape[-1] == 2 or pts.shape[-1] == 3
    if filter_indices is None:
        filter_indices = np.arange(config["skeleton"].num_joints)
    if config is None:
        config = {
            "freq": 100,  # Hz
            "mincutoff": 0.0001,  # FIXME # 0.1
            "beta": 30,  # FIXME
            "dcutoff": 1.0,  # this one should be ok
        }
    if freq is not None:
        config["freq"] = freq

    f = [
        [OneEuroFilter(**config) for j in range(pts.shape[-1])]
        for i in range(config["skeleton"].num_joints)
    ]
    timestamp = 0.0  # seconds
    pts_after = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            if j in filter_indices:
                pts_after[i, j, 0] = f[j][0](pts[i, j, 0], i * 0.1)
                pts_after[i, j, 1] = f[j][1](pts[i, j, 1], i * 0.1)
            else:
                pts_after[i, j] = pts[i, j]
    return pts_after


def smooth_pose2d(points2d, window_size=20, pad=20, std_thr=5):
    from scipy.ndimage.filters import gaussian_filter1d

    points2d_filter = points2d.copy()
    points2d_pad = np.zeros(
        (points2d.shape[0] + 2 * pad, points2d.shape[1], points2d.shape[2])
    )
    points2d_pad[pad:-pad, :] = points2d.copy()
    points2d_pad[:pad, :] = points2d[0, :]
    points2d_pad[-pad:, :] = points2d[-1, :]
    for img_id in range(20, points2d.shape[0] + 20):
        for j in range(points2d.shape[1]):
            for d in range(2):
                segment_std = segment_smooth = points2d_pad[
                    img_id - window_size // 2 : img_id + window_size // 2, j, d
                ]
                std = np.std(segment_std)
                if std < std_thr:
                    sigma = 7
                else:
                    sigma = 0.1
                filtered = gaussian_filter1d(
                    segment_smooth, sigma=sigma, mode="nearest"
                )[window_size // 2]
                points2d_filter[img_id - pad, j, d] = filtered
    return points2d_filter
