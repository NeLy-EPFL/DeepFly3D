"""Rotate triangulated 3D poses into a fly-centric (body) coordinate frame.

The raw triangulated point cloud (and the procrustes-template-aligned cloud that
DeepFly3D saves as ``points3d``) sits at an essentially arbitrary orientation
inherited from the camera calibration / procrustes template: no data axis lines
up with a meaningful body direction, so a coordinate such as "tarsus-tip x" is a
mixture of anterior-posterior, medial-lateral and dorsal-ventral motion. This
module computes a body frame from the six thorax-coxa (leg-base) landmarks and
the dorsal thorax stripe, and rotates every point into it, so that the three
output axes become anatomically meaningful:

    x = anterior-posterior   (+x = anterior, toward the head)
    y = medial-lateral       (+y = the fly's left)
    z = dorsal-ventral       (+z = dorsal, toward the stripe; "up" / leg lift)

Height (leg lift / dorsal-ventral) is therefore the LAST axis (z), keeping the
(x, y, z) triple documented for ``points3d`` elsewhere in df3d, and matching the
_x/_y/_z column order used downstream (e.g. in scapepp). The transform is a
proper rigid rotation about the coxae centroid (det(R) = +1, no reflection, so
left/right chirality is preserved), computed per recording from that recording's
own landmarks -- there are no hardcoded angles, so it adapts automatically.
"""

import numpy as np

from df3d import skeleton_fly as skeleton


def _coxa_indices():
    """Joint indices of the six thorax-coxa (leg-base) landmarks, and the
    dorsal stripe landmarks, from the fly skeleton definition."""
    body_coxa = [j for j in range(skeleton.num_joints)
                 if skeleton.is_tracked_point(j, skeleton.Tracked.BODY_COXA)]
    stripe = [j for j in range(skeleton.num_joints)
              if skeleton.is_tracked_point(j, skeleton.Tracked.STRIPE)]
    # Split the six coxae into front/hind and left/right using the skeleton's
    # limb ids (front leg = first leg of each side) and left/right visibility.
    front, hind, left, right = [], [], [], []
    for j in body_coxa:
        limb = skeleton.get_limb_id(j)
        is_left = skeleton.is_limb_visible_left(limb)
        (left if is_left else right).append(j)
    # Within each side the legs are ordered front, middle, hind; the first coxa
    # of a side is the front leg, the last is the hind leg.
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    front = [left_sorted[0], right_sorted[0]]
    hind = [left_sorted[-1], right_sorted[-1]]
    return body_coxa, stripe, front, hind, left_sorted, right_sorted


def compute_body_frame(mean_pose):
    """Return (R, centroid) mapping the current frame to the body frame.

    ``mean_pose`` is a (num_joints, 3) array (e.g. the time-median pose).
    ``R`` has the body axes (AP, ML, DV) as columns, so
    ``points_body = (points - centroid) @ R``.
    """
    body_coxa, stripe, front, hind, left, right = _coxa_indices()
    coxae = mean_pose[body_coxa]
    centroid = np.nanmean(coxae, axis=0)

    anterior = np.nanmean(mean_pose[front], axis=0)
    posterior = np.nanmean(mean_pose[hind], axis=0)
    ap = anterior - posterior
    ap = ap / np.linalg.norm(ap)

    # Dorsal-ventral axis: normal of the best-fit plane through the six coxae,
    # signed so it points toward the dorsal stripe.
    _, _, vt = np.linalg.svd(coxae - np.nanmean(coxae, axis=0))
    dv = vt[2]
    stripe_center = np.nanmean(mean_pose[stripe], axis=0)
    if np.dot(stripe_center - centroid, dv) < 0:
        dv = -dv
    # Orthonormalize DV against AP.
    dv = dv - np.dot(dv, ap) * ap
    dv = dv / np.linalg.norm(dv)

    # Medial-lateral axis completes a right-handed frame (no reflection).
    ml = np.cross(dv, ap)
    R = np.column_stack([ap, ml, dv])
    if np.linalg.det(R) < 0:
        ml = -ml
        R = np.column_stack([ap, ml, dv])
    return R, centroid


def align_to_body_axes(points3d):
    """Rotate a (num_frames, num_joints, 3) pose array into the fly body frame.

    The frame is computed once from the time-median pose (the coxae are
    near-stationary in the body frame) and applied to every frame, so relative
    motion is preserved exactly and only the global orientation/origin changes.
    Returns a new array; the input is not modified.
    """
    points3d = np.asarray(points3d, dtype=float)
    if points3d.ndim != 3 or points3d.shape[2] != 3:
        raise ValueError("points3d must have shape (num_frames, num_joints, 3), "
                         f"got {points3d.shape}")
    if points3d.shape[1] != skeleton.num_joints:
        raise ValueError(f"expected {skeleton.num_joints} joints, "
                         f"got {points3d.shape[1]}")
    mean_pose = np.nanmedian(points3d, axis=0)
    R, centroid = compute_body_frame(mean_pose)
    return (points3d - centroid) @ R
