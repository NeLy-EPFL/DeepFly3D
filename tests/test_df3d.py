import glob
import os
import pathlib
import pickle
import random
import shlex
import shutil
import subprocess
import unittest

import cv2
import numpy as np
import torch

import df3d.core
import df3d.video

TEST_DATA_LOCATION = str(pathlib.Path(__file__).parent / "data")
TEST_DATA_LOCATION_REFERENCE = f"{TEST_DATA_LOCATION}/reference"
TEST_DATA_LOCATION_RESULT = f"{TEST_DATA_LOCATION_REFERENCE}_df3d"
TEST_DATA_LOCATION_RESULT_FILE_2D = f"{TEST_DATA_LOCATION_RESULT}/df3d_result_2d.pkl"
TEST_DATA_LOCATION_RESULT_FILE_3D = f"{TEST_DATA_LOCATION_RESULT}/df3d_result_3d.pkl"
TEST_DATA_LOCATION_REFERENCE_VIDEO_2D = f"{TEST_DATA_LOCATION_RESULT}/video_pose2d.mp4"
TEST_DATA_LOCATION_REFERENCE_VIDEO_3D = f"{TEST_DATA_LOCATION_RESULT}/video_pose3d.mp4"
TEST_DATA_LOCATION_WORKING = f"{TEST_DATA_LOCATION}/working"
TEST_DATA_LOCATION_WORKING_RESULT = f"{TEST_DATA_LOCATION_WORKING}_df3d"


def reset_rngs():
    # See: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def clear_working_data():
    if os.path.exists(TEST_DATA_LOCATION_WORKING):
        shutil.rmtree(TEST_DATA_LOCATION_WORKING)
    if os.path.exists(TEST_DATA_LOCATION_WORKING_RESULT):
        shutil.rmtree(TEST_DATA_LOCATION_WORKING_RESULT)


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
    shutil.copy(TEST_DATA_LOCATION_RESULT_FILE_2D, get_results_save_path())


def load_results_3d():
    os.makedirs(TEST_DATA_LOCATION_WORKING_RESULT, exist_ok=True)
    shutil.copy(TEST_DATA_LOCATION_RESULT_FILE_3D, get_results_save_path())


def get_results_save_path():
    return os.path.join(
        TEST_DATA_LOCATION_WORKING_RESULT,
        "df3d_result_{}.pkl".format(TEST_DATA_LOCATION_WORKING.replace("/", "_")),
    )


def get_results_2d():
    with open(TEST_DATA_LOCATION_RESULT_FILE_2D, "rb") as f:
        return pickle.load(f)


def get_results_3d():
    with open(TEST_DATA_LOCATION_RESULT_FILE_3D, "rb") as f:
        return pickle.load(f)


def get_video_2d_frames():
    return get_video_frames(TEST_DATA_LOCATION_REFERENCE_VIDEO_2D)


def get_video_3d_frames():
    return get_video_frames(TEST_DATA_LOCATION_REFERENCE_VIDEO_3D)


def get_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    return frames


class TestDeepFly3D(unittest.TestCase):
    def setUp(self):
        clear_working_data()
        reset_rngs()

    def tearDown(self):
        clear_working_data()

    def test_load_core_with_videos(self):
        """Test that we can create the Core in a folder that only contains videos.
        In this case we need to convert the videos to images first."""
        load_videos()

        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        self.assertEqual(core.num_images, 15, "Core didn't find all images in folder")
        self.assertEqual(
            core.image_shape, [960, 480], "Core didn't get the right image shape"
        )
        self.assertTrue(
            np.all(core.camera_ordering == np.array([0, 1, 2, 3, 4, 5, 6])),
            "Core didn't get correct camera ordering",
        )

    def test_load_core_with_images(self):
        """Test that we can create the Core in a folder that already contains images"""
        load_images()

        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        self.assertEqual(core.num_images, 15, "Core didn't find all images in folder")
        self.assertEqual(
            core.image_shape, [960, 480], "Core didn't get the right image shape"
        )
        self.assertTrue(
            np.all(core.camera_ordering == np.array([0, 1, 2, 3, 4, 5, 6])),
            "Core didn't get correct camera ordering",
        )

    def test_pose_estimation(self):
        """Test that we can run pose estimation on images and get the right 2D points"""
        load_images()

        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )
        core.pose2d_estimation()

        reference_results = get_results_2d()

        assert core.points2d is not None, (
            "2D pose estimation completely failed - no points are available"
        )
        np.testing.assert_allclose(
            core.points2d,
            reference_results["points2d"],
            err_msg="2D pose estimation points not correct.",
            atol=0.02,
        )
        np.testing.assert_allclose(
            core.conf,
            reference_results["heatmap_confidence"],
            err_msg="2D pose estimation confidence heatmaps not correct.",
            atol=0.002,
        )

        core.save()

        with open(core.save_path, "rb") as f:
            saved_pose_data = pickle.load(f)

        np.testing.assert_allclose(
            saved_pose_data["points2d"],
            reference_results["points2d"],
            err_msg="2D pose estimation points not saved correctly.",
            atol=0.02,
        )
        np.testing.assert_allclose(
            saved_pose_data["heatmap_confidence"],
            reference_results["heatmap_confidence"],
            err_msg="2D pose estimation confidence heatmaps not saved correctly.",
            atol=0.002,
        )

    def test_calibration(self):
        """Test that we can run calibration to triangulate the 2D points into 3D points"""
        load_images()
        # FIX: can't load in 2d results from pose estimation and resume from there - CameraNetwork tries to load calib data which doesn't exist
        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        results_2d = get_results_2d()
        reference_results = get_results_3d()

        # manually set the pose estimation points to the reference
        core.points2d = results_2d["points2d"]
        core.conf = results_2d["heatmap_confidence"]
        core.calibrate_calc(0, 100)
        core.save()

        with open(core.save_path, "rb") as f:
            saved_pose_data = pickle.load(f)

        np.testing.assert_allclose(
            saved_pose_data["points3d_wo_procrustes"],
            reference_results["points3d_wo_procrustes"],
            err_msg="3D pose estimation points3d_wo_procrustes not correct.",
            atol=1e-5,
        )
        np.testing.assert_allclose(
            saved_pose_data["points3d"],
            reference_results["points3d"],
            err_msg="3D pose estimation points3d not correct.",
            atol=1e-5,
        )

        def check_cameras_match(camera: int):
            for key in saved_pose_data[camera].keys():
                np.testing.assert_allclose(
                    saved_pose_data[camera][key],
                    reference_results[camera][key],
                    err_msg="3D pose estimation camera {camera} calibration property {key} not correct.",
                    atol=1e-4,
                )

        for camera_id in range(7):
            check_cameras_match(camera_id)

    def test_video_2d(self):
        """Test that we can generate a video of the 2D pose estimation results"""
        load_images()
        load_results_3d()
        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        df3d.video.make_pose2d_video(
            core.plot_2d, core.num_images, core.input_folder, core.output_folder
        )

        video_name = "video_pose2d_" + core.input_folder.replace("/", "_") + ".mp4"
        video_path = os.path.join(core.output_folder, video_name)
        self.assertTrue(
            os.path.exists(video_path),
            f"Video of 2D poses wasn't successfully created - does not exist at {core.output_folder}/{video_name}",
        )

        reference_frames = get_video_2d_frames()
        test_frames = get_video_frames(video_path)

        self.assertEqual(
            len(test_frames),
            len(reference_frames),
            "Number of frames in output video doesn't match what it should",
        )
        for frame, (test_frame, reference_frame) in enumerate(
            zip(test_frames, reference_frames)
        ):
            np.testing.assert_almost_equal(
                test_frame,
                reference_frame,
                err_msg=f"Frame {frame} of 2D video doesn't match what it should",
            )

    def test_video_3d(self):
        """Test that we can generate a video of the 3D pose estimation results"""
        load_images()
        load_results_3d()
        core = df3d.core.Core(
            input_folder=TEST_DATA_LOCATION_WORKING,
            output_folder=TEST_DATA_LOCATION_WORKING_RESULT,
            num_images_max=0,
            camera_ordering=[0, 1, 2, 3, 4, 5, 6],
        )

        df3d.video.make_pose3d_video(
            core.get_points3d(),
            core.plot_2d,
            core.num_images,
            core.input_folder,
            core.output_folder,
        )

        video_name = "video_pose3d_" + core.input_folder.replace("/", "_") + ".mp4"
        video_path = os.path.join(core.output_folder, video_name)
        self.assertTrue(
            os.path.exists(video_path),
            f"Video of 3D poses wasn't successfully created - does not exist at {core.output_folder}/{video_name}",
        )

        reference_frames = get_video_3d_frames()
        test_frames = get_video_frames(video_path)

        self.assertEqual(
            len(test_frames),
            len(reference_frames),
            "Number of frames in output video doesn't match what it should",
        )
        for frame, (test_frame, reference_frame) in enumerate(
            zip(test_frames, reference_frames)
        ):
            np.testing.assert_almost_equal(
                test_frame,
                reference_frame,
                err_msg=f"Frame {frame} of 3D video doesn't match what it should",
            )

    def test_cli_default_output_dir(self):
        """
        Test that running df3d from the cli uses the correct default output directory
        """
        load_videos()

        subprocess.run(shlex.split("df3d-cli tests/data/working"))

        reference_results = get_results_2d()

        assert os.path.exists(f"{TEST_DATA_LOCATION_WORKING}_df3d"), (
            "results folder not in default location"
        )
        results_file = [
            file
            for file in os.listdir(f"{TEST_DATA_LOCATION_WORKING}_df3d")
            if file.startswith("df3d_result")
        ]
        assert len(results_file) == 1, "Couldn't find df3d_results file"

        with open(f"{TEST_DATA_LOCATION_WORKING}_df3d/{results_file[0]}", "rb") as f:
            saved_pose_data = pickle.load(f)

        np.testing.assert_allclose(
            saved_pose_data["points2d"],
            reference_results["points2d"],
            err_msg="2D pose estimation points not correct.",
            atol=0.02,
        )
        np.testing.assert_allclose(
            saved_pose_data["heatmap_confidence"],
            reference_results["heatmap_confidence"],
            err_msg="2D pose estimation confidence heatmaps not correct.",
            atol=0.002,
        )

    def test_delete_images(self):
        """
        Test that running df3d with the --delete-images option deletes the images when done
        """
        load_videos()

        subprocess.run(
            shlex.split(
                "df3d-cli tests/data/working --delete-images"
            )
        )

        assert (
            len(glob.glob(os.path.join(TEST_DATA_LOCATION_WORKING, "camera_*.jpg")))
            == 0
        ), "images weren't deleted properly after running"
        assert (
            len(glob.glob(os.path.join(TEST_DATA_LOCATION_WORKING, "camera_*.mp4")))
            == 7
        ), "videos were accidentally deleted after running"


if __name__ == "__main__":
    unittest.main()
