# pylint: skip-file
import glob
import pickle
import warnings

import numpy as np

from df3d import skeleton_fly as skeleton
from df3d.os_util import *
from df3d.plot_util import normalize_pose_3d


def apply_transformation(pts, R=None, t=None, s=None, tform=None):
    if tform is not None:
        R = tform["rotation"]
        s = tform["scale"]
        t = tform["translation"]
    return s * np.dot(pts, R) + t


def calc_bone_length(pts3d, warn=False):
    """
    Returns the distances of adjacent points inside pts3d
    """
    n_points = pts3d.shape[0]
    bone_length = np.zeros((n_points - 1))
    for idx in range(n_points - 1):
        bone_length[idx] = np.linalg.norm((pts3d[idx + 1] - pts3d[idx]))
    if warn:
        if np.any(bone_length[0] > bone_length[1:]):
            warnings.warn(
                "Coxa-femur is longer than other segments {}".format(bone_length)
            )

    return bone_length


def read_template_pose3d(path=config["procrustes_template"]):
    d = np.load(
        file=glob.glob(os.path.join(path, "df3d_result.pkl"))[0], allow_pickle=True
    )

    pts3d = d["points3d"]
    assert pts3d is not None
    return pts3d


def procrustes_seperate(
    pts,
    reflection="best",
    verbose=False,
    joint=(skeleton.Tracked.BODY_COXA, skeleton.Tracked.COXA_FEMUR),
):
    """
    Performs procrustes seperately for each three legs seperately
    """

    m_left = np.arange(0, 15)
    points3d_gt_left = read_template_pose3d()[:, m_left].copy()
    points3d_pred_left = pts[:, m_left].copy()
    pts_t_left = procrustes(
        pts=points3d_pred_left,
        template=points3d_gt_left,
        joint=joint,
        verbose=verbose,
        reflection=reflection,
        return_transf=False,
    )

    m_right = np.arange(skeleton.num_joints // 2, skeleton.num_joints // 2 + 15)
    points3d_gt_right = read_template_pose3d()[:, m_right].copy()
    points3d_pred_right = pts[:, m_right].copy()
    pts_t_right, tform = procrustes(
        pts=points3d_pred_right,
        template=points3d_gt_right,
        joint=joint,
        verbose=verbose,
        reflection=reflection,
        return_transf=True,
    )

    pts3d_proc = np.zeros_like(pts)
    pts3d_proc[:, m_left] = pts_t_left.copy()
    pts3d_proc[:, m_right] = pts_t_right.copy()

    return pts3d_proc


def procrustes(
    pts,
    template=None,
    reflection="best",
    verbose=False,
    joint=(skeleton.Tracked.BODY_COXA, skeleton.Tracked.COXA_FEMUR),
    return_transf=False,
):
    if template is None:
        template = read_template_pose3d()
    body_coxa_idx = [j for j in range(min(pts.shape[1], template.shape[1]))]
    body_coxa_idx = [
        j
        for j in body_coxa_idx
        if np.any([skeleton.is_tracked_point(j, k) for k in joint])
    ]

    # calculate the scaling factor
    n_limbs = 3
    bone_length_pts = np.zeros((pts.shape[0], n_limbs, 4))
    for img_id in range(pts.shape[0]):
        for limb_id in range(n_limbs):
            bone_length_pts[img_id, limb_id, :] = calc_bone_length(
                pts[img_id, 5 * limb_id : 5 * (limb_id + 1)]
            )
    bone_length_template = np.zeros((template.shape[0], n_limbs, 4))
    for img_id in range(template.shape[0]):
        for limb_id in range(n_limbs):
            bone_length_template[img_id, limb_id, :] = calc_bone_length(
                template[img_id, 5 * limb_id : 5 * (limb_id + 1)]
            )
    s = np.median(
        bone_length_template.reshape(bone_length_template.shape[0], -1), axis=0
    ) / np.median(bone_length_pts.reshape(bone_length_pts.shape[0], -1), axis=0)
    s = np.median(s)

    pts = normalize_pose_3d(pts)
    pts *= s

    template_bc = template[:, body_coxa_idx]
    pts_bc = pts[:, body_coxa_idx]

    template_bc = np.median(template_bc, axis=0)
    pts_bc = np.median(pts_bc, axis=0)

    d, Z, tform = __procrustes(
        template_bc, pts_bc, reflection=reflection, scaling=False
    )
    R_b, s_b, t_b = tform["rotation"], tform["scale"], tform["translation"]

    pts_t = apply_transformation(pts.copy(), R_b, t_b, s_b)

    if verbose:
        print("Body-coxa index:", body_coxa_idx)
        print("Tform,", tform)

    if return_transf:
        return pts_t, tform
    else:
        return pts_t


def __procrustes(X, Y, scaling=True, reflection="best"):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.0).sum()
    ssY = (Y0 ** 2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform
