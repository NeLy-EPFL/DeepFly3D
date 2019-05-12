import numpy as np
import cv2
'''
points3d: 3xn(euclidean) or 4xn(homogenous) matrix
points2d: 2xn matrix
proj: 4x3 matrix
R: 3x3 matrix
t: 1x3 matrix  or 3, vector
'''
def get_essential_matrix(points1, points2, intr):
    E, mask = cv2.findEssentialMat(points1=points1, points2=points2, cameraMatrix=intr, method=cv2.RANSAC, prob=0.9999, threshold=5)
    print("Essential matrix inlier ratio: {}".format(np.sum(mask)/mask.shape[0]))
    return E, mask


def get_extr_from_Rt(R,t):
    extrinsic = np.zeros(shape=(3,4))
    extrinsic[:3,:3] = R
    extrinsic[:,3] = np.squeeze(t)
    return extrinsic


def get_proj_matrix_from_Rt(R,t,intr):
    extr = get_extr_from_Rt(R, t)
    P = np.matmul(intr, extr)
    return P


def get_Rt_from_essential(E, points1, points2, intr):
    retval, R, t, mask, _ = cv2.recoverPose(E, points1=points1, points2=points2, cameraMatrix=intr, distanceThresh=100)
    return R,t,mask


def calculate_intersect_mask(points1, points2, ignore_joint_list=[]):
    bool_intersect = np.logical_and(points1!=0, points2!=0)
    for j in ignore_joint_list:
        bool_intersect[:,np.arange(j, bool_intersect.shape[1], 5)] = False
    return bool_intersect


def triangulate(P1, P2, points1, points2):
    points3d = cv2.triangulatePoints(P1, P2, points1.T.astype(float), points2.T.astype(float))
    if points3d.shape[0]==4:
        points3d = points3d.transpose()
    points3d = hom_to_eucl(points3d)
    return points3d


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]

'''
n-view linear triangulation
https://github.com/smidm/camera.py/blob/master/camera.py
'''


def nview_linear_triangulation_single(cameras, correspondences):
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert(len(cameras) >= 2)
    assert(type(cameras) == list)
    assert(correspondences.shape == (2, len(cameras)))

    def _construct_D_block(P, uv):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """
        return np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))


    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv)
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    return p2e(u[:, -1, np.newaxis])


def nview_linear_triangulations(cameras, image_points):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    """
    assert(type(cameras) == list)
    assert(type(image_points) == list)
    assert(len(cameras) == image_points[0].shape[1])
    assert(image_points[0].shape[0] == 2)

    world = np.zeros((3, len(image_points)))
    for i, correspondence in enumerate(image_points):
        world[:, i] = np.squeeze(nview_linear_triangulation_single(cameras, correspondence))
    return world


def triangulate_linear(cam_list, point_list):
    '''
    :param cam_list: list of camera object
    :param point_list: list of nx2 numpy arrays
    '''
    num_cameras = len(cam_list)
    num_points = point_list[0].shape[0]
    image_points = []
    for count_points in range(num_points):
        correspondence = np.empty(shape=(2,num_cameras))
        for count_cameras in range(num_cameras):
            correspondence[:,count_cameras] = point_list[count_cameras][count_points,:]
        image_points.append(correspondence)
    points3d = nview_linear_triangulations(cam_list, image_points)
    if points3d.shape[0]==3:
        points3d = points3d.transpose()
    return points3d


'''
n-view linear triangulation
https://github.com/smidm/camera.py/blob/master/camera.py
'''

def hom_to_eucl(points3d):
    points3d = points3d * (1./points3d[:,3][:,np.newaxis])
    return points3d[:,:3]


def R_to_rodrigues(R):
    return cv2.Rodrigues(R)[0]


def project_points(points3d, R, t, intr, distort=np.array([[[0,0,0,0]],[[0,0,0,0]]],dtype=np.float)):
    if R.ndim==2: # then rotation matrix
        R = R_to_rodrigues(R)
    points2d, jacobian = cv2.projectPoints(objectPoints=points3d.T, rvec=R, tvec=t, cameraMatrix=intr, distCoeffs=distort)
    points2d =  np.squeeze(points2d)
    points2d = points2d.reshape(-1,2)
    return points2d


def reprojection_error(points3d, points2d, cam):
    err_list = (points2d.reshape(-1,2) - cam.project(points3d).reshape(-1,2)).ravel()
    return np.mean(np.abs(err_list)), np.array(err_list)


def Rt_inverse(R, t):
    assert(R.shape[0]==3 and R.shape[1]==3)
    R = R.transpose()
    t = np.matmul(-R, t)
    return R, t
