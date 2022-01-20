import unittest
import subprocess
import glob
import os
import pickle
import shutil
from shutil import copyfile

from pyba.CameraNetwork import CameraNetwork
import pickle
import glob
import numpy as np
from pyba.config import df3d_colors, df3d_bones
import matplotlib.pyplot as plt


def delete_images(folder):
    for hgx in glob.glob(folder + "*.jpg"):
        os.remove(hgx)


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

    return has_keys


class TestDf3d(unittest.TestCase):
    def test_cli(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        # remove result folder
        shutil.rmtree(path_result)
        # to test expanding videos
        delete_images(path)
        _ = subprocess.check_output(
            ["df3d-cli ./sample/test/  -vv --video-2d --video-3d"], shell=True
        )
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))

        # check --skip-pose-estimation
        _ = subprocess.check_output(
            [
                "df3d-cli ./sample/test/ -vv --skip-pose-estimation --video-2d --video-3d"
            ],
            shell=True,
        )
        self.assertTrue(check_df3d_result(path_result))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose2d*.mp4")[0]))
        self.assertTrue(os.path.exists(glob.glob(path_result + "video_pose3d*.mp4")[0]))

        # move resulting video for further inspection
        shutil.copy(glob.glob(path_result + "video_pose2d*.mp4")[0], "./")
        shutil.copy(glob.glob(path_result + "video_pose3d*.mp4")[0], "./")

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
        img = camNet.plot_2d(0, points="points2d")
        plt.figure(figsize=(20, 20))
        plt.imshow(img, cmap="gray")
        plt.savefig("pyba.jpg")

    def test_prior_reprojection(self):
        calib_path = os.path.join("./data/calib.pkl")
        calib = pickle.load(open(calib_path, "rb"))

        image_path = "./sample/test/camera_{cam_id}_img_{img_id}.jpg"
        pr_path = "./sample/test/df3d/df3d_result*.pkl"
        d = pickle.load(open(glob.glob(pr_path)[0], "rb"))
        points2d = d["points2d"]
        camNet = CameraNetwork(
            points2d=points2d * [480, 960],
            calib=calib,
            image_path=image_path,
            bones=df3d_bones,
            colors=df3d_colors,
        )
        reproj = camNet.reprojection_error(reduce=True)

        self.assertGreaterEqual(40, reproj)

    def test_posterior_reprojection(self):
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
        reproj = camNet.reprojection_error(reduce=True)

        self.assertGreaterEqual(40, reproj)

    def test_cpu(self):
        path = "./sample/test/"
        path_result = path + "df3d/"
        shutil.rmtree(path_result)
        _ = subprocess.check_output(
            ["CUDA_VISIBLE_DEVICES=-1 df3d-cli ./sample/test/  -vv"], shell=True
        )
        self.assertTrue(check_df3d_result(path_result))


if __name__ == "__main__":
    unittest.main()
