import glob
import os
import pickle
import shutil
import subprocess
import unittest

import matplotlib.pyplot as plt
import numpy as np
from pyba.CameraNetwork import CameraNetwork
from pyba.config import df3d_bones, df3d_colors

from df3d import video
from df3d.core import Core

import pathlib

TEST_DATA_LOCATION = str(pathlib.Path(__file__).parent / "data")
TEST_DATA_LOCATION_REFERENCE = f"{TEST_DATA_LOCATION}/reference"
TEST_DATA_LOCATION_REFERENCE_RESULT = f"{TEST_DATA_LOCATION_REFERENCE}/df3d/"
TEST_DATA_LOCATION_REFERENCE_RESULT_FILE_2D = f"{TEST_DATA_LOCATION_REFERENCE_RESULT}/df3d_result_2d.pkl"
TEST_DATA_LOCATION_REFERENCE_RESULT_FILE_3D = f"{TEST_DATA_LOCATION_REFERENCE_RESULT}/df3d_result_3d.pkl"
TEST_DATA_LOCATION_WORKING = f"{TEST_DATA_LOCATION}/working"
TEST_DATA_LOCATION_WORKING_RESULT = f"{TEST_DATA_LOCATION_WORKING}/df3d"

def clear_working_data():
    shutil.rmtree(TEST_DATA_LOCATION_WORKING)

def load_videos():
    os.makedirs(TEST_DATA_LOCATION_WORKING, exist_ok=True)
    for video in glob.glob(f"{TEST_DATA_LOCATION_REFERENCE}/*.mp4"):
        shutil.copy(video, TEST_DATA_LOCATION_WORKING)

def load_images():
    os.makedirs(TEST_DATA_LOCATION_WORKING, exist_ok=True)
    for image in glob.glob(f"{TEST_DATA_LOCATION_REFERENCE}/*.jpg"):
        shutil.copy(image, TEST_DATA_LOCATION_WORKING)

def load_results_2d():
    os.makedirs(TEST_DATA_LOCATION_WORKING_RESULT, exist_ok=True)
    shutil.copy(TEST_DATA_LOCATION_REFERENCE_RESULT_FILE_2D, get_results_save_path())

def load_results_3d():
    os.makedirs(TEST_DATA_LOCATION_WORKING_RESULT, exist_ok=True)
    shutil.copy(TEST_DATA_LOCATION_REFERENCE_RESULT_FILE_2D, get_results_save_path())

def get_results_save_path():
    return os.path.join(
        TEST_DATA_LOCATION_WORKING_RESULT,
        "df3d_result_{}.pkl".format(TEST_DATA_LOCATION_WORKING.replace("/", "_")),
    )

def get_results():
    with open(TEST_DATA_LOCATION_REFERENCE_RESULT_FILE_3D, "rb") as f:
        return pickle.load(f)


def delete_df3d_folder(path):
    path_result = path + "df3d/"
    if os.path.exists(path_result):
        shutil.rmtree(path_result)


def check_df3d_result(folder):
    print(os.path.join(folder, "df3d_result*.pkl"))
    print(glob.glob(os.path.join(folder, "df3d_result*.pkl")))
    path = glob.glob(os.path.join(folder, "df3d_result*.pkl"))[0]
    with open(path, "rb") as f:
        df3d_res = pickle.load(f)
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
    with open(glob.glob(pr_path)[0], "rb") as f:
        d = pickle.load(f)
    points2d = d["points2d"]
    camNet = CameraNetwork(
        points2d=points2d * [480, 960],
        calib=d,
        bones=df3d_bones,
        colors=df3d_colors,
    )
    reproj = camNet.reprojection_error(reduce=True)

    return reproj



"""Tests to run


IDEA -
have copies of the images and results files
run the code and produce output
check that it's the same

1. get core
    - with frames already existing
    - without frames already existing
2. do pose2d_estimation
3. do calibrate_calc
4. make 2d video
5. make 3d video



"""

class TestDeepFly3D(unittest.TestCase):
    def tearDown(self):
        clear_working_data()

    # @classmethod
    # def setUpClass(cls) -> None:
    #     pass

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     """Remove all the files we created once we're done running tests"""
    #     clear_working_data()

    def test_load_core_with_videos(self):
        """Test that we can create the Core in a folder that only contains videos.
        In this case we need to convert the videos to images first."""
        load_videos()

        core = Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_subfolder="df3d",
            num_images_max=100,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        self.assertEqual(core.num_images, 14, "Core didn't find all images in folder")
        self.assertEqual(core.image_shape, [960,480], "Core didn't get the right image shape")
        self.assertTrue(np.all(core.camera_ordering == np.array([0, 1, 2, 3, 4, 5, 6])), "Core didn't get correct camera ordering")

    def test_load_core_with_images(self):
        """Test that we can create the Core in a folder that already contains images"""
        load_images()

        core = Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_subfolder="df3d",
            num_images_max=100,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        self.assertEqual(core.num_images, 14, "Core didn't find all images in folder")
        self.assertEqual(core.image_shape, [960,480], "Core didn't get the right image shape")
        self.assertTrue(np.all(core.camera_ordering == np.array([0, 1, 2, 3, 4, 5, 6])), "Core didn't get correct camera ordering")

    def test_pose_estimation(self):
        """Test that we can create the Core in a folder that already contains images"""
        load_images()

        core = Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_subfolder="df3d",
            num_images_max=100,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )
        core.pose2d_estimation()

        reference_results = get_results()
        self.assertTrue(np.all(core.points2d == reference_results["points2d"]), f"2D pose estimation points not correct. Max diff = {np.max(abs(core.points2d - reference_results['points2d']))}")
        self.assertTrue(np.all(core.conf == reference_results["heatmap_confidence"]), f"2D pose estimation confidence heatmaps not correct. Max diff = {np.max(abs(core.conf - reference_results['heatmap_confidence']))}")

        core.save()

        with open(core.save_path, "rb") as f:
            saved_pose_data = pickle.load(f)

        self.assertTrue(np.all(saved_pose_data["points2d"] == reference_results["points2d"]), f"2D pose estimation points not saved correctly. Max diff = {np.max(abs(saved_pose_data['points2d'] == reference_results['points2d']))}")
        self.assertTrue(np.all(saved_pose_data["heatmap_confidence"] == reference_results["heatmap_confidence"]), f"2D pose estimation confidence heatmaps not saved correctly. Max diff = {np.max(abs(saved_pose_data['heatmap_confidence'] == reference_results['heatmap_confidence']))}")

    
    def test_calibration(self):
        load_images()
        # FIX: can't load in 2d results from pose estimation and resume from there - CameraNetwork tries to load calib data
        # need to r
        core = Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_subfolder="df3d",
            num_images_max=100,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        core.pose2d_estimation()
        core.calibrate_calc(0, 100)
        core.save()

        reference_results = get_results()
        with open(core.save_path, "rb") as f:
            saved_pose_data = pickle.load(f)
        
        self.assertTrue(np.allclose(saved_pose_data["points3d_wo_procrustes"], reference_results["points3d_wo_procrustes"], atol=1e-4), f"3D pose estimation points3d_wo_procrustes not correct. Max diff = {np.max(abs(saved_pose_data['points3d_wo_procrustes'] - reference_results['points3d_wo_procrustes']))}")
        self.assertTrue(np.allclose(saved_pose_data["points3d"], reference_results["points3d"], atol=1e-4), f"3D pose estimation points3d not correct. Max diff = {np.max(abs(saved_pose_data['points3d'] - reference_results['points3d']))}")

        def check_cameras_match(camera: int):
            for key in saved_pose_data[camera].keys():
                self.assertTrue(np.allclose(saved_pose_data[camera][key], reference_results[camera][key], atol=1e-4), f"3D pose estimation camera {camera} calibration property {key} not correct. Max diff = {np.max(abs(saved_pose_data[camera][key] - reference_results[camera][key]))}")

        for camera_id in range(6):
            check_cameras_match(camera_id)


    # def test_python_interface(self):
    #     """Tests the whole process of pose estimation and making 2D and 3D videos."""
    #     delete_df3d_folder(TEST_DATA_LOCATION)
    #     core = Core(
    #         input_folder=TEST_DATA_LOCATION,
    #         num_images_max=100,
    #         output_subfolder="df3d",
    #         camera_ordering=[0, 1, 2, 3, 4, 5, 6],
    #     )

    #     core.pose2d_estimation()
    #     core.calibrate_calc(0, 100)
    #     core.save()

    #     video.make_pose2d_video(
    #         core.plot_2d, core.num_images, core.input_folder, core.output_folder
    #     )
    #     video.make_pose3d_video(
    #         core.get_points3d(),
    #         core.plot_2d,
    #         core.num_images,
    #         core.input_folder,
    #         core.output_folder,
    #     )

    #     self.assertTrue(check_df3d_result(TEST_DATA_LOCATION_RESULT))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose2d*.mp4")[0]))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0]))
    #     self.assertGreaterEqual(5, get_reprojection_error(TEST_DATA_LOCATION_RESULT))

    # def test_expanding_videos(self):
    #     shutil.rmtree(TEST_DATA_LOCATION_RESULT)
    #     delete_images(TEST_DATA_LOCATION)
    #     _ = subprocess.check_output(
    #         [f"df3d-cli {TEST_DATA_LOCATION} -vv --order 0 1 2 3 4 5 6 --video-2d --video-3d -n 5"], shell=True
    #     )
    #     self.assertTrue(check_df3d_result(TEST_DATA_LOCATION_RESULT))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose2d*.mp4")[0]))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0]))
    #     self.assertGreaterEqual(5, get_reprojection_error(TEST_DATA_LOCATION_RESULT))

    # def test_skip_pose_estimation(self):
    #     _ = subprocess.check_output(
    #         [
    #             f"df3d-cli {TEST_DATA_LOCATION} -vv --order 0 1 2 3 4 5 6 --skip-pose-estimation --video-2d --video-3d"
    #         ],
    #         shell=True,
    #     )
    #     self.assertTrue(check_df3d_result(TEST_DATA_LOCATION_RESULT))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose2d*.mp4")[0]))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0]))
    #     self.assertGreaterEqual(5, get_reprojection_error(TEST_DATA_LOCATION_RESULT))

    #     # move resulting video for further inspection
    #     shutil.copy(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose2d*.mp4")[0], "./")
    #     shutil.copy(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0], "./")

    # def test_order(self):
    #     _ = subprocess.check_output(
    #         [f"df3d-cli {TEST_DATA_LOCATION} -vv --order 0 1 2 3 4 5 6 --video-2d --video-3d"],
    #         shell=True,
    #     )
    #     self.assertTrue(check_df3d_result(TEST_DATA_LOCATION_RESULT))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose2d*.mp4")[0]))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0]))
    #     self.assertGreaterEqual(5, get_reprojection_error(TEST_DATA_LOCATION_RESULT))

    # def test_pyba(self):
    #     image_path = TEST_DATA_LOCATION + "/camera_{cam_id}_img_{img_id}.jpg"
    #     pr_path = f"{TEST_DATA_LOCATION_RESULT}/df3d_result*.pkl"
    #     with open(glob.glob(pr_path)[0], "rb") as f:
    #         d = pickle.load(f)
    #     points2d = d["points2d"]
    #     camNet = CameraNetwork(
    #         points2d=points2d * [480, 960],
    #         calib=d,
    #         image_path=image_path,
    #         bones=df3d_bones,
    #         colors=df3d_colors,
    #     )
    #     _, axs = plt.subplots(2, 1, figsize=(30, 5))
    #     axs[0].imshow(camNet.plot_2d(0, points="points2d"))
    #     axs[1].imshow(camNet.plot_2d(0, points="reprojection"))
    #     plt.savefig("pyba.jpg")

    #     _ = camNet.triangulate()
    #     _ = camNet.bundle_adjust(update_distort=False, update_intrinsic=False)
    #     reproj = camNet.reprojection_error()
    #     self.assertGreaterEqual(5, reproj)

    # def test_prior_reprojection(self):
    #     calib_path = os.path.join("./data/calib.pkl")
    #     with open(calib_path, "rb") as f:
    #         calib = pickle.load(f)

    #     pr_path = "./sample/test/df3d/df3d_result*.pkl"
    #     with open(glob.glob(pr_path)[0], "rb") as f:
    #         d = pickle.load(f)
    #     points2d = d["points2d"]
    #     camNet = CameraNetwork(
    #         points2d=points2d * [480, 960],
    #         calib=calib,
    #         bones=df3d_bones,
    #         colors=df3d_colors,
    #     )
    #     reproj = camNet.reprojection_error(reduce=True)

    #     self.assertGreaterEqual(5, reproj)

    # def test_posterior_reprojection(self):
    #     reproj = get_reprojection_error("./sample/test/df3d/")
    #     self.assertGreaterEqual(5, reproj)

    # def test_cpu(self):
    #     subprocess.call(
    #         ['CUDA_VISIBLE_DEVICES=""', 'df3d-cli', TEST_DATA_LOCATION, '-vv', '--order', '0 1 2 3 4 5 6', '--video-3d'],
    #         shell=True,
    #     )

    #     self.assertTrue(check_df3d_result(TEST_DATA_LOCATION_RESULT))
    #     self.assertTrue(os.path.exists(glob.glob(TEST_DATA_LOCATION_RESULT + "video_pose3d*.mp4")[0]))
    #     self.assertGreaterEqual(5, get_reprojection_error(TEST_DATA_LOCATION_RESULT))


if __name__ == "__main__":
    unittest.main()
