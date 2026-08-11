"""
Helpers used by `df3d.belief_propagation`.

Resurrected from the pre-2021 `deepfly.optim_util` module and adapted to
the current package layout.

Coordinate convention: 2D points here are (x_norm, y_norm) -- normalized
[0, 1] coordinates in pyba/opencv (column, row) order. Multiplying by
`image_shape` = (W, H) gives pixel coordinates (x_pix, y_pix);
multiplying by `hm_shape` = (W_hm, H_hm) gives heatmap-pixel coordinates.

Heatmap lookup goes through `cam.heatmaps[img_id, j_id]`, which df3d
populates with the network's per-joint probability maps before calling
BP. `pyba.Camera.project` has a batched (T, J, 3) -> (T, J, 2) signature;
`_project_single` below wraps it so the single-point BP code keeps
working.

Defaults for `image_shape` / `hm_shape` are resolved lazily inside the
function bodies because `config["image_shape"]` is populated by
`df3d.core.Core.__init__` and is therefore not available at import time.
"""
import numpy as np

from df3d.cv_util import triangulate_linear
from df3d.config import config


def energy_drosoph(
    cam_list,
    img_id,
    j_id,
    points2d,
    points3d=None,
    bone_length=None,
    image_shape=None,
    hm_shape=None,
):
    """
    Calculate energy from 2d observations.

    points2d: 2x3 array, observations from three cameras
    points3d: 15x3 array, used only to calculate the bone probability
    """
    if image_shape is None:
        image_shape = np.asarray(config['image_shape'])  # already (W, H)
    if hm_shape is None:
        # config stores heatmap_shape as (H, W) to match the network output's
        # row-major axes; reverse it so the multiplication with (x, y) points
        # gives (x_pix_hm, y_pix_hm).
        h_hm, w_hm = config['heatmap_shape']
        hm_shape = np.array([w_hm, h_hm])
    points2d_list = [p_.reshape(1, 2) for p_ in points2d * image_shape]
    p3d = triangulate_linear(cam_list, points2d_list)

    err_proj = error_reprojection(cam_list, (points2d * image_shape).astype(int))
    err_proj = np.mean(np.abs(err_proj))

    prob_heatm = probability_heatmap(
        cam_list, img_id, j_id, (points2d * hm_shape).astype(int)
    )

    # not the root of the chain
    prob_bone = None
    return p3d, err_proj, prob_heatm, prob_bone


def prob_from_heatmap(hm, p, eps=0.1):
    """
    points2d: pixel space
    hm: probability map in full image size, pixel space
    """
    prob = eps
    if not (p[1] >= hm.shape[0] or p[0] >= hm.shape[1] or p[0] < 0 or p[1] < 0):
        prob += float(hm[p[1], p[0]])
    return prob


def probability_heatmap(cam_list, img_id, j_id, points2d):
    """Score a set of candidate 2D points against per-camera heatmaps.

    `points2d` are (x_pix_hm, y_pix_hm) integer indices into each camera's
    heatmap for the given (img_id, j_id). The heatmap is `cam.heatmaps[
    img_id, j_id]` -- df3d's `Core.run_belief_propagation` attaches the
    network's heatmaps before BP starts.
    """
    prob = 1
    for cam, p in zip(cam_list, points2d):
        hm = None if cam.heatmaps is None else cam.heatmaps[img_id, j_id]
        prob *= prob_from_heatmap(hm, p) if hm is not None else 0.1
    return prob


def _project_single(cam, point3d):
    """Project one 3D point to 2D using `pyba.Camera`'s batched API.

    `pyba.Camera.project` requires shape (T, J, 3) and returns (T, J, 2);
    BP works with single points, so we wrap and squeeze.
    """
    point3d = np.asarray(point3d).reshape(1, 1, 3)
    return cam.project(point3d).reshape(2)


def error_reprojection(cam_list, points2d):
    """
    points2d: nx2 array containing projections
    """
    points2d_list = [p.reshape(1, 2) for p in points2d]
    point3d = triangulate_linear(cam_list, points2d_list)

    err = list()
    for cam, p in zip(cam_list, points2d):
        err.append(_project_single(cam, point3d) - p)
    return np.array(err)


def project_on_last(cam_list, p):
    p = [p_.reshape(1, 2) for p_ in p]

    point3d = triangulate_linear(cam_list[:-1], p)
    return _project_single(cam_list[-1], point3d)


def calc_bone_length(p):
    p = np.squeeze(p)
    bone_length = np.zeros(p.shape[0], dtype=float)
    for j in range(p.shape[0] - 1):
        bone_length[j] = np.sqrt(np.sum(np.power(p[j] - p[j + 1], 2), axis=1))
    return bone_length


def d_ij(p3d_c, p3d_p, param):
    dist = np.linalg.norm((p3d_p - p3d_c))
    mu, sig = param
    return np.exp(-np.power(dist - mu, 2.0) / (2 * np.power(sig, 2.0)))
