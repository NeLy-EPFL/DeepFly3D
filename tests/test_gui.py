from PyQt5.QtWidgets import QApplication
from deepfly.GUI.main import DrosophAnnot
import os.path
from pathlib import Path
from itertools import product

INPUT_DIRECTORY = Path(__file__).parent / '../data/test'
NB_IMGS_IN_INPUT_DIR = 15
NB_CAMERAS = 7

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
    assert window.folder == os.path.abspath(INPUT_DIRECTORY)


def test_setup_input_folder_prompted(qtbot):
    class A(DrosophAnnot):
        def __init__(self, *args, **kwargs):
            DrosophAnnot.__init__(self, *args, **kwargs)

        def prompt_for_directory(self):
            return INPUT_DIRECTORY

    window = A()
    qtbot.addWidget(window)
    window.setup()
    assert window.folder == os.path.abspath(INPUT_DIRECTORY)


def test_setup_num_images_max_from_args(qtbot):
    N = 1
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(input_folder=INPUT_DIRECTORY, num_images_max=N)
    assert N < NB_IMGS_IN_INPUT_DIR, "Choose a smaller number of images"
    assert window.state.max_num_images == N
    assert window.state.num_images == N


def test_num_images(qtbot):
    window = DrosophAnnot()
    qtbot.addWidget(window)
    window.setup(input_folder=INPUT_DIRECTORY)
    assert window.state.num_images == NB_IMGS_IN_INPUT_DIR