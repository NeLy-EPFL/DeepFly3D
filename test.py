import glob
import os
import pickle
import shutil
import subprocess
import unittest
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from pyba.CameraNetwork import CameraNetwork
from pyba.config import df3d_bones, df3d_colors

from df3d import video
from df3d.core import Core


def delete_images(folder):
    for hgx in glob.glob(folder + "*.jpg"):
        os.remove(hgx)


def delete_df3d_folder(path):
    path_result = path + "df3d/"
    if os.path.exists(path_result):
        shutil.rmtree(path_result)


def check_df3d_result(folder):
    path = glob.glob(os.path.join(folder, "df3d_result*.pkl"))[0]
    df3d_res = pickle.load(open(path, "rb"))
    has_keys = all(
        [
            k in df3d_res
            for k in [
                "points2d",
                "points3d",
                "camera_ordering",
                "heatmap_confidence",
                0,
            ]
        ]
    )

    # make sure motion capture output has consistent shape
    pts3d, pts2d = df3d_res["points3d"], df3d_res["points2d"]
    pts3d_s, pts2d_s = pts3d.shape, pts2d.shape
    correct_shape = (
        pts2d_s[0] == 7
        and pts3d_s[0] == pts2d_s[1]
        and pts3d_s[1] == pts2d_s[2]
        and pts3d_s[-1] == 3
        and pts2d_s[-1] == 2
    )

    # missing points are marked with zeros, make sure there are none
    missing_3d_points = np.any(df3d_res["points3d"] == 0)

    # make sure 2d and 3d points are in sensible range
    pts2d_in_range = np.all(np.logical_and(pts2d >= 0, pts2d <= 1))
    pts3d_in_range = np.all(np.logical_and(pts3d >= -5, pts3d <= 5))

    return (
        has_keys
        and correct_shape
        and not missing_3d_points
        and pts2d_in_range
        and pts3d_in_range
    )


def get_reprojection_error(df3d_path):
    pr_path = df3d_path + "/df3d_result*.pkl"
    d = pickle.load(open(glob.glob(pr_path)[0], "rb"))
    points2d = d["points2d"]
    camNet = CameraNetwork(
        points2d=points2d * [480, 960],
        calib=d,
        bones=df3d_bones,
        colors=df3d_colors,
    )
    reproj = camNet.reprojection_error(reduce=True)

    return reproj


class TestDf3d(unittest.TestCase):
    def test_python_interface(self):
        path = "./sample/test/"
        path_result = "./sample/test/df3d/"
        delete_df3d_folder(path)
        core = Core(
            input_folder=path,
            num_images_max=100,
            output_subfolder="df3d",
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        core.pose2d_estimation()
        core.calibrate_calc(0, 100)
        core.save()

        video.make_pose2d_video(
            core.plot_2d, core.num_images, core.input_folder, core.output_folder
        )
        video.make_pose3d_video(
            core.get_points3d(),
            core.plot_2d,
            core.num_images,
            core.input_folder,
            core.output_folder,
        )

        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(5, get_reprojection_error(path_result))

    def test_expanding_videos(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        shutil.rmtree(path_result)
        delete_images(path)
        _ = subprocess.check_output(
            ["df3d-cli ./sample/test/  -vv --video-2d --video-3d -n 5"], shell=True
        )
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(5, get_reprojection_error(path_result))

    def test_skip_pose_estimation(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        _ = subprocess.check_output(
            [
                "df3d-cli ./sample/test/ -vv --skip-pose-estimation --video-2d --video-3d"
            ],
            shell=True,
        )
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(5, get_reprojection_error(path_result))

        # move resulting video for further inspection
        shutil.copy(glob.glob(path_result + "video_pose2d*.mp4")[0], "./")
        shutil.copy(glob.glob(path_result + "video_pose3d*.mp4")[0], "./")

    def test_order(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        _ = subprocess.check_output(
            ["df3d-cli ./sample/test/ -vv -order 0 1 2 3 4 5 6 --video-2d --video-3d"],
            shell=True,
        )
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(5, get_reprojection_error(path_result))

    def test_pyba(self):
        image_path = "./sample/test/camera_{cam_id}_img_{img_id}.jpg"
        pr_path = "./sample/test/df3d/df3d_result*.pkl"
        d = pickle.load(open(glob.glob(pr_path)[0], "rb"))
        points2d = d["points2d"]
        camNet = CameraNetwork(
            points2d=points2d * [480, 960],
            calib=d,
            image_path=image_path,
            bones=df3d_bones,
            colors=df3d_colors,
        )
        _, axs = plt.subplots(2, 1, figsize=(30, 5))
        axs[0].imshow(camNet.plot_2d(0, points="points2d"))
        axs[1].imshow(camNet.plot_2d(0, points="reprojection"))
        plt.savefig("pyba.jpg")

        _ = camNet.triangulate()
        _ = camNet.bundle_adjust(update_distort=False, update_intrinsic=False)
        reproj = camNet.reprojection_error()
        self.assertGreaterEqual(5, reproj)

    def test_prior_reprojection(self):
        calib_path = os.path.join("./data/calib.pkl")
        calib = pickle.load(open(calib_path, "rb"))

        pr_path = "./sample/test/df3d/df3d_result*.pkl"
        d = pickle.load(open(glob.glob(pr_path)[0], "rb"))
        points2d = d["points2d"]
        camNet = CameraNetwork(
            points2d=points2d * [480, 960],
            calib=calib,
            bones=df3d_bones,
            colors=df3d_colors,
        )
        reproj = camNet.reprojection_error(reduce=True)

        self.assertGreaterEqual(5, reproj)

    def test_posterior_reprojection(self):
        reproj = get_reprojection_error("./sample/test/df3d/")
        self.assertGreaterEqual(5, reproj)

    def test_cpu(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        shutil.rmtree(path_result)
        _ = subprocess.check_output(
            ["CUDA_VISIBLE_DEVICES=-1 df3d-cli ./sample/test/  -vv --video-3d"],
            shell=True,
        )

        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(5, get_reprojection_error(path_result))

    def test_missing_camera(self):
        path = "./sample/test_missing_camera/"
        path_result = path + "df3d/"
        delete_df3d_folder(path)
        _ = subprocess.check_output(
            [
                "df3d-cli ./sample/test_missing_camera/ --video-3d -vv -order 0 1 2 3 4 5 6"
            ],
            shell=True,
        )

        shutil.copy(glob.glob(path_result + "video_pose3d*.mp4")[0], "./")
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))
        self.assertGreaterEqual(10, get_reprojection_error(path_result))


if __name__ == "__main__":
    unittest.main()
