import os.path
from pathlib import Path
from itertools import product
import random
import shutil
import filecmp

import numpy as np
from PyQt5 import QtCore

from deepfly.GUI.main import DrosophAnnot

INPUT_DIRECTORY = Path(__file__).parent / '../data/test'
NB_IMGS_IN_INPUT_DIR = 15
NB_CAMERAS = 7

INPUT_DIRECTORY2 = Path(__file__).parent / '../data/test-with-error'
DIR2_ERROR_IMG1 = 2
DIR2_ERROR_IMG2 = 3
NB_IMGS_IN_INPUT_DIR2 = 6


def reset_input_directory():
    output_dir = INPUT_DIRECTORY / 'df3d'
    reference_dir = INPUT_DIRECTORY / 'df3d.sav'
    assert reference_dir.exists(), 'folder /df3d.sav/ does not exist'
    shutil.rmtree(output_dir)
    shutil.copytree(reference_dir, output_dir)
    assert output_dir.is_dir(), f'{output_dir} not created'


def test_input_directory_exists():
    assert INPUT_DIRECTORY.is_dir()


def test_imgs_in_input_dir():
    for c_id,i_id in product(range(NB_CAMERAS), range(NB_IMGS_IN_INPUT_DIR)):
        img = INPUT_DIRECTORY / f'camera_{c_id}_img_{i_id:06}.jpg'
        assert img.is_file(), f'{img} does not exist'


def test_can_instantiate(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    

def test_parse_cli_args_none():
    from deepfly.GUI.main import parse_cli_args
    args = parse_cli_args(["exec_name"])
    assert 'input_folder' not in args
    assert 'num_images_max' not in args

    
def test_parse_cli_args_only_folder(tmp_path):
    directory = tmp_path / "test"
    directory.mkdir()

    from deepfly.GUI.main import parse_cli_args
    args = parse_cli_args(["exec_name", str(directory)])
    assert 'input_folder' in args
    assert args['input_folder'] == str(directory)
    

def test_parse_cli_args_folder_and_num(tmp_path):
    directory = tmp_path / "test"
    directory.mkdir()
    
    from deepfly.GUI.main import parse_cli_args
    args = parse_cli_args(["exec_name", str(directory), "10"])
    assert 'input_folder' in args
    assert args['input_folder'] == str(directory)
    assert 'num_images_max' in args
    assert args['num_images_max'] == 10

    
def test_parse_cli_args_when_num_is_bad(tmp_path):
    directory = tmp_path / "test"
    directory.mkdir()
    
    from deepfly.GUI.main import parse_cli_args
    args = parse_cli_args(["exec_name", str(directory), "XX"])
    assert 'input_folder' in args
    assert args['input_folder'] == str(directory)
    assert 'num_images_max' not in args


def test_setup_input_folder_from_args(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    assert window.core.input_folder == os.path.abspath(INPUT_DIRECTORY)


def test_setup_input_folder_prompted(qtbot):
    class A(DrosophAnnot):
        def __init__(self, *args, **kwargs):
            DrosophAnnot.__init__(self, *args, **kwargs)
            self.called = False

        def prompt_for_directory(self):
            self.called = True
            return INPUT_DIRECTORY

    window = A()
    qtbot.addWidget(window)
    window.setup()
    assert window.called, "prompt dialog not called"
    assert window.core.input_folder == os.path.abspath(INPUT_DIRECTORY)


def test_setup_num_images_max_from_args(qtbot):
    N = 1
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(input_folder=INPUT_DIRECTORY, num_images_max=N)
    assert N < NB_IMGS_IN_INPUT_DIR, "Choose a smaller number of images"
    assert window.core.num_images_max == N
    assert window.core.num_images == N


def test_num_images(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(input_folder=INPUT_DIRECTORY)
    assert window.core.num_images == NB_IMGS_IN_INPUT_DIR


def test_camera_order(qtbot):
    ordering = list(range(NB_CAMERAS))
    random.shuffle(ordering)

    class A(DrosophAnnot):
        def __init__(self):
            DrosophAnnot.__init__(self)
            self.called = False

        def prompt_for_camera_ordering(self):
            self.called = True
            return list(ordering)  # create a copy to make sure ordering is not modified

    window = A()
    qtbot.addWidget(window)
    window.setup(input_folder=INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_camera_order, QtCore.Qt.LeftButton)
    assert window.called, "prompt dialog not called"
    assert np.all(window.core.cidread2cid == np.array(ordering)), window.core.cidread2cid


def test_calibration(qtbot):
    class A(DrosophAnnot):
        def __init__(self):
            DrosophAnnot.__init__(self)
            self.called = False
        
        def prompt_for_calibration_range(self):
            self.called = True
            return [0, NB_IMGS_IN_INPUT_DIR-1]

    window = A()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.button_calibrate_calc, QtCore.Qt.LeftButton)
    assert window.called, "prompt dialog not called"


def test_pose_save(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.button_pose_save, QtCore.Qt.LeftButton)    


def test_mode_image(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    
    qtbot.mouseClick(window.button_image_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.IMAGE
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == 1
    qtbot.mouseClick(window.button_prev, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_last, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR -1 
    

def test_mode_pose(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)

    qtbot.mouseClick(window.button_pose_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.POSE
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == 1
    qtbot.mouseClick(window.button_prev, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_last, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR -1 


def test_mode_heatmap(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)

    qtbot.mouseClick(window.button_heatmap_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.HEATMAP
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == 1
    qtbot.mouseClick(window.button_prev, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_last, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR -1 


def test_mode_correction(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    
    qtbot.mouseClick(window.button_correction_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.CORRECTION
    
    window.checkbox_correction_skip.setChecked(True)
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR -1  # skip activated

    window.checkbox_correction_skip.setChecked(False)
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == 1                        # skip not activated

    qtbot.mouseClick(window.button_prev, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    qtbot.mouseClick(window.button_last, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR -1 
    

def test_belief_propagation(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY2)
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    #
    qtbot.mouseClick(window.button_correction_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.CORRECTION
    #
    window.checkbox_correction_skip.setChecked(True)
    qtbot.mouseClick(window.button_first, QtCore.Qt.LeftButton)
    assert window.state.img_id == 0
    #
    qtbot.mouseClick(window.button_next, QtCore.Qt.LeftButton)
    assert window.state.img_id == DIR2_ERROR_IMG1
    #
    qtbot.mouseClick(window.button_last, QtCore.Qt.LeftButton)
    assert window.state.img_id == NB_IMGS_IN_INPUT_DIR2 -1 
    #
    qtbot.mouseClick(window.button_prev, QtCore.Qt.LeftButton)
    assert window.state.img_id == DIR2_ERROR_IMG2


def test_manual_corrections(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(INPUT_DIRECTORY)
    canvas = window.image_pose_list[1]
    #
    reset_input_directory()
    #
    qtbot.mouseClick(window.button_pose_estimate, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.button_correction_mode, QtCore.Qt.LeftButton)
    assert window.state.mode == window.state.mode.CORRECTION
    #
    for y in range(298, 60, -1):
        qtbot.mouseMove(canvas, pos=QtCore.QPoint(200, y))
    qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.button_pose_save, QtCore.Qt.LeftButton)    
    #
    output = (INPUT_DIRECTORY / 'df3d').absolute()
    expected = (INPUT_DIRECTORY / 'df3d.mc.expected').absolute()
    assert output.is_dir(), "output directory not found"
    assert expected.is_dir(), "directory for comparison not found"
    files = [p.relative_to(expected) for p in expected.iterdir()]
    match, mismatch, errors = filecmp.cmpfiles(output, expected, files)
    assert len(mismatch) == 0, f"Mismatching output files: {mismatch}"
    assert len(errors) == 0, f"Expected files not found: {errors}"
    #
    reset_input_directory()

    