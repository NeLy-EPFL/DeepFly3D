import pickle
import sys
from itertools import chain
import re

from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (QWidget, QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, 
                            QCheckBox, QPushButton, QLineEdit, QComboBox, QInputDialog, QMessageBox)

from .State import State, Mode
#from .util.os_util import *
from .Config import config
from deepfly.core import Core
from .ImageView import ImageView


def main():
    app = QApplication([])
    window = DrosophAnnot()

    cli_args = parse_cli_args(sys.argv)
    window.setup(**cli_args)
    
    screen = app.primaryScreen()
    size = screen.size()
    _, width = size.height(), size.width()
    # hw_ratio = 960 * 2 / 360.0

    hw_ratio = config["image_shape"][0] * 1.2 / config["image_shape"][1]
    app_width = width
    app_height = int(app_width / hw_ratio)
    window.resize(app_width, app_height)
    window.show()
    app.exec_()


def parse_cli_args(argv):
    args = {}
    try:
        args['input_folder'] = argv[1]
        args['num_images_max'] = int(argv[2])
    except (IndexError, ValueError):
        pass
    return args
    

class DrosophAnnot(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.img_id = 0
        self.mode = Mode.IMAGE
        self.core = None


    def setup(self, input_folder=None, num_images_max=None):
        self.core = Core(input_folder or self.prompt_for_directory(), num_images_max)
        self.setup_layout()
        self.set_mode(self.mode)
        

    def setup_layout(self):
        # Create checkboxes
        self.checkbox_solve_bp = QCheckBox("Auto-correct", self)
        self.checkbox_solve_bp.setChecked(True)
        self.checkbox_solve_bp.stateChanged.connect(self.update_frame)

        self.checkbox_correction_skip = QCheckBox("Skip to next error", self)
        self.checkbox_correction_skip.setChecked(True)
        self.checkbox_correction_skip.stateChanged.connect(self.update_frame)
        
        # Create buttons
        self.button_first           = self.make_button("<<",          self.onclick_first_image)
        self.button_prev            = self.make_button("<",           self.onclick_prev_image)
        self.button_next            = self.make_button(">",           self.onclick_next_image)
        self.button_last            = self.make_button(">>",          self.onclick_last_image)
        self.button_pose_mode       = self.make_button("Pose",        lambda b: self.set_mode(self.mode.POSE))
        self.button_image_mode      = self.make_button("Image",       lambda b: self.set_mode(self.mode.IMAGE))
        self.button_heatmap_mode    = self.make_button("Prob. Map",   lambda b: self.set_mode(self.mode.HEATMAP))
        self.button_correction_mode = self.make_button("Correction",  lambda b: self.set_mode(self.mode.CORRECTION))
        button_textbox_img_id_go    = self.make_button("Go",          self.onclick_goto_img)
        self.button_pose_save       = self.make_button("Save",        self.onclick_save_pose)
        self.button_calibrate_calc  = self.make_button("Calibration", self.onclick_calibrate)
        self.button_camera_order    = self.make_button("Camera ordering", self.onclick_camera_order)
        self.button_pose_estimate   = self.make_button("2D Pose Estimation", self.onclick_pose2d_estimation)
        
        self.button_image_mode.setCheckable(True) 
        self.button_correction_mode.setCheckable(True)
        self.button_pose_mode.setCheckable(True)
        self.button_heatmap_mode.setCheckable(True)
        
        self.textbox_img_id = QLineEdit(str(self.img_id), self)
        self.textbox_img_id.setFixedWidth(100)
        
        self.combo_joint_id = QComboBox(self)
        self.combo_joint_id.addItem("View all joints", [])
        for i in range(config["skeleton"].num_joints):
            self.combo_joint_id.addItem(f'View joint {i}', [i])
        self.combo_joint_id.activated[str].connect(self.update_frame)
        
        self.image_pose_list     = [ImageView(self.core, i) for i in [0, 1, 2]]
        self.image_pose_list_bot = [ImageView(self.core, i) for i in [4, 5, 6]]

        # Layouts
        layout_h_images = QHBoxLayout()
        layout_h_images.setSpacing(1)
        for image_pose in self.image_pose_list:
            layout_h_images.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())
        
        layout_h_images_bot = QHBoxLayout()
        layout_h_images.setSpacing(1)
        for image_pose in self.image_pose_list_bot:
            layout_h_images_bot.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())

        layout_h_buttons_top = QHBoxLayout()
        layout_h_buttons_top.setSpacing(3)
        layout_h_buttons_top.setAlignment(Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_estimate,  alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_calibrate_calc, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_camera_order,   alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_pose_save,      alignment=Qt.AlignLeft)
        layout_h_buttons_top.addStretch()
        layout_h_buttons_top.addWidget(self.button_image_mode,        alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_mode,         alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_correction_mode,   alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_heatmap_mode,      alignment=Qt.AlignRight)
        
        layout_h_buttons = QHBoxLayout()
        layout_h_buttons.setSpacing(1)
        layout_h_buttons.addWidget(self.button_first)
        layout_h_buttons.addWidget(self.button_prev)
        layout_h_buttons.addWidget(self.button_next)
        layout_h_buttons.addWidget(self.button_last)
        layout_h_buttons.addWidget(self.textbox_img_id)
        layout_h_buttons.addWidget(button_textbox_img_id_go)
        layout_h_buttons.addStretch()
        layout_h_buttons.addWidget(self.checkbox_correction_skip, alignment=Qt.AlignRight)
        layout_h_buttons.addWidget(self.checkbox_solve_bp,        alignment=Qt.AlignRight)
        layout_h_buttons.addStretch()
        layout_h_buttons.addWidget(self.combo_joint_id, alignment=Qt.AlignRight)
        
        layout_v = QVBoxLayout()
        layout_v.addLayout(layout_h_buttons_top)
        layout_v.addLayout(layout_h_images)
        layout_v.addLayout(layout_h_images_bot)
        layout_v.addLayout(layout_h_buttons)
        layout_v.setSpacing(0)

        self.setLayout(layout_v)
        self.setWindowTitle(self.core.input_folder)


    def make_button(self, text, onClick):
        b = QPushButton(text, self)
        b.setMaximumWidth(b.fontMetrics().boundingRect(text).width() + 27)
        b.clicked.connect(onClick)
        return b


    # ------------------------------------------------------------------
    # onclick callbacks


    def onclick_camera_order(self):
        cidread2cid = self.prompt_for_camera_ordering()
        if self.core.update_camera_ordering(cidread2cid):
            self.update_frame()
        else:
            msgBox = QMessageBox()
            msgBox.setText("Ordering not changed (wrong format or canceled).")
            msgBox.exec()


    def onclick_pose2d_estimation(self):
        self.core.pose2d_estimation()
        self.set_mode(Mode.POSE)


    def onclick_first_image(self):
        self.display_img(0)


    def onclick_last_image(self):
        self.display_img(self.core.max_img_id)


    def onclick_prev_image(self):
        if (self.mode == Mode.CORRECTION 
        and self.correction_skip_enabled
        and self.core.has_calibration() 
        and self.core.has_pose()
        ):
            self.display_img(self.core.next_error(self.img_id, backward=True))
        else:
            self.display_img(max(self.img_id - 1, 0))


    def onclick_next_image(self):
        if (self.mode == Mode.CORRECTION 
        and self.correction_skip_enabled
        and self.core.has_calibration() 
        and self.core.has_pose()
        ):
            self.display_img(self.core.next_error(self.img_id))
        else:
            self.display_img(min(self.core.max_img_id, self.img_id + 1))


    def onclick_calibrate(self):
        try:
            [min_img_id, max_img_id] = self.prompt_for_calibration_range()
        except BaseException:
            min_img_id, max_img_id = 0, self.core.max_img_id
        self.core.calibrate_calc(self, min_img_id, max_img_id)


    def onclick_save_pose(self):
        self.core.save_pose()
        self.core.save_corrections()


    def onclick_goto_img(self):
        try:
            img_id = int(self.textbox_img_id.text().replace("Heatmap: ", ""))
            self.display_img(img_id)
        except BaseException as e:
            print("Textbox img id is not integer {}".format(str(e)))


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.onclick_goto_img()
            self.setFocus()
        if event.key() == Qt.Key_A:
            self.onclick_prev_image()
        if event.key() == Qt.Key_D:
            self.onclick_next_image()
        if event.key() == Qt.Key_H:
            self.set_mode(Mode.HEATMAP)
        if event.key() == Qt.Key_I:
            self.set_mode(Mode.IMAGE)
        if event.key() == Qt.Key_X:
            self.set_mode(Mode.POSE)
        if event.key() == Qt.Key_C:
            self.set_mode(Mode.CORRECTION)
            self.update_frame()
        if event.key() == Qt.Key_T:
            for image_pose in self.image_pose_list:
                image_pose.save_correction()
            self.update_frame()

        
    # ------------------------------------------------------------------
    # prompt callbacks

    
    def prompt_for_directory(self):
        return str(QFileDialog.getExistingDirectory(
                self,
                directory="./",
                caption="Select Directory",
                options=QFileDialog.DontUseNativeDialog,
            ))


    def prompt_for_camera_ordering(self):
        text, ok_pressed = QInputDialog.getText(self, "Rename Images", "Camera order:", QLineEdit.Normal, "")
        if ok_pressed:
            cidread2cid = re.findall(r'[0-9]+', text) 
            cidread2cid = [int(x) for x in cidread2cid]
            return cidread2cid


    def prompt_for_calibration_range(self):
        text, okPressed = QInputDialog.getText(self, "Calibration", "Range of images:", QLineEdit.Normal, f"0-{self.core.max_img_id}")
        if okPressed:
            numbers = re.findall(r'[0-9]+', text)
            numbers = [int(x) for x in numbers]
            return numbers


    # ------------------------------------------------------------------


    def set_mode(self, mode):
        if (mode == Mode.IMAGE
        or (mode == Mode.POSE       and self.core.has_pose())
        or (mode == Mode.HEATMAP    and self.core.has_heatmap())
        or (mode == Mode.CORRECTION and self.core.has_pose())
        ):
            self.mode = mode
        else:
            print("Cannot set mode: {}".format(mode))
        
        self.button_correction_mode.setChecked(self.mode == Mode.CORRECTION)
        self.button_heatmap_mode.setChecked(self.mode == Mode.HEATMAP)
        self.button_image_mode.setChecked(self.mode == Mode.IMAGE)
        self.button_pose_mode.setChecked(self.mode == Mode.POSE)
        self.update_frame()
      

    def update_frame(self):
        self.display_img(self.img_id)
        

    def display_img(self, img_id):
        self.img_id = img_id
        self.textbox_img_id.setText(str(self.img_id))

        if self.belief_propagation_enabled:
            self.core.solve_bp(img_id)

        joints = self.combo_joint_id.currentData()
        for ip in chain(self.image_pose_list, self.image_pose_list_bot):
            ip.show(img_id, self.mode, joints)
        

    @property
    def belief_propagation_enabled(self):
        return (self.mode == Mode.CORRECTION) and self.checkbox_solve_bp.isChecked()


    @property
    def correction_skip_enabled(self):
        return self.checkbox_correction_skip.isChecked()


if __name__ == "__main__":
    main()