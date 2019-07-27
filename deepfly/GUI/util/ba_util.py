import numpy as np
from scipy.sparse import lil_matrix

def fun(
    params,
    cam_list,
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    points_2d,
    residual_mask=None,
):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    assert point_indices.shape[0] == points_2d.shape[0]
    assert camera_indices.shape[0] == points_2d.shape[0]

    camera_params = params[: n_cameras * 13].reshape((n_cameras, 13))
    points3d = params[n_cameras * 13 :].reshape((n_points, 3))
    cam_indices_list = list(set(camera_indices))

    points_proj = np.zeros(shape=(point_indices.shape[0], 2), dtype=np.float)
    for cam_id in cam_indices_list:
        # set the variables

        cam_list[cam_idx].set_rvec(camera_params[cam_id][0:3])
        cam_list[cam_idx].set_tvec(camera_params[cam_id][3:6])
        # cam_list[cam_id].set_focal_length(camera_params[cam_id][6], camera_params[cam_id][7])
        # cam_list[cam_id].set_focal_length(camera_params[cam_id][6], camera_params[cam_id][7])
        # cam_list[cam_id].set_distort(camera_params[cam_id][8:13])

        points2d_mask = camera_indices == cam_id
        points3d_where = point_indices[points2d_mask]
        points_proj[points2d_mask, :] = cam_list[cam_id.project(
            points3d[points3d_where]
        )

    res = points_proj - points_2d
    res = res.ravel()
    if residual_mask is not None:
        res *= residual_mask

    return res


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    assert camera_indices.shape[0] == point_indices.shape[0]
    n_camera_params = 13
    # m = camera_indices.size * 2 # x,y for each 3d point
    m = (
        camera_indices.size * 2
    )  # + n_cameras * 3 + n_cameras * 3 + 3  # residuals, reprojection error for each dimension and translation error and rotation error and align error
    # notice that camera_incides.size gives the number of observations

    n = (
        n_cameras * n_camera_params + n_points * 3
    )  # all the parameters, 13 camera parameters and x,y,z values for n_points
    A = lil_matrix((m, n), dtype=int)  # sparse matrix

    i = np.arange(camera_indices.size)
    for s in range(
        n_camera_params
    ):  # assign camera parameters to points residuals (reprojection error)
        A[2 * i, camera_indices * n_camera_params + s] = 1
        A[2 * i + 1, camera_indices * n_camera_params + s] = 1

    for s in range(3):  # assign 3d points to residuals (reprojection error)
        A[2 * i, n_cameras * n_camera_params + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * n_camera_params + point_indices * 3 + s] = 1
    return A
