import numpy as np
import re

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (QWidget, QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, 
                            QCheckBox, QPushButton, QLineEdit, QComboBox, QInputDialog, QMessageBox, QLabel)
from PyQt5.QtGui import QImage, QPixmap, QPainter

from deepfly.core import Core


def main():
    import sys
    cli_args = parse_cli_args(sys.argv)

    app = QApplication([])
    window = DrosophAnnot()
    window.setup(**cli_args)
    window.set_width(app.primaryScreen().size().width())
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
        self.core = None


    def setup(self, input_folder=None, num_images_max=None):
        self.core = Core(input_folder or self.prompt_for_directory(), num_images_max)
        self.setup_layout()
        self.switch_to_image_mode()
    
    
    def set_width(self, width):
        hw_ratio = self.core.image_shape[0] * 1.2 / self.core.image_shape[1]
        height = int(width / hw_ratio)
        self.resize(width, height)
        

    def setup_layout(self):
        # --- Create checkboxes ---
        self.checkbox_solve_bp = QCheckBox("Auto-correct", self)
        self.checkbox_solve_bp.setChecked(True)
        self.checkbox_solve_bp.stateChanged.connect(self.update_frame)

        self.checkbox_correction_skip = QCheckBox("Skip to next error", self)
        self.checkbox_correction_skip.setChecked(True)
        self.checkbox_correction_skip.stateChanged.connect(self.update_frame)
        
        # --- Create buttons ---
        self.button_first           = self.make_button("<<",          self.onclick_first_image)
        self.button_prev            = self.make_button("<",           self.onclick_prev_image)
        self.button_next            = self.make_button(">",           self.onclick_next_image)
        self.button_last            = self.make_button(">>",          self.onclick_last_image)
        self.button_pose_mode       = self.make_button("Pose",        self.switch_to_pose_mode)
        self.button_image_mode      = self.make_button("Image",       self.switch_to_image_mode)
        self.button_heatmap_mode    = self.make_button("Prob. Map",   self.switch_to_heatmap_mode)
        self.button_correction_mode = self.make_button("Correction",  self.switch_to_correction_mode)
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
        for i in range(self.core.number_of_joints):
            self.combo_joint_id.addItem(f'View joint {i}', [i])
        self.combo_joint_id.activated[str].connect(self.update_frame)
        
        # --- Create widgets to display images and pose results ---
        def make_image_view(cam_id):
            iv = QLabel()
            iv.setScaledContents(True)
            iv.cam_id = cam_id
            return iv

        top_row    = [make_image_view(cam_id) for cam_id in [0, 1, 2]]
        bottom_row = [make_image_view(cam_id) for cam_id in [4, 5, 6]]
        self.image_views = top_row + bottom_row

        for image_view in self.image_views:
            image_view.installEventFilter(self)

        # --- Layouts ---
        layout_h_images = QHBoxLayout()
        layout_h_images.setSpacing(1)
        for image_pose in top_row:
            layout_h_images.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())
        
        layout_h_images_bot = QHBoxLayout()
        layout_h_images_bot.setSpacing(1)
        for image_pose in bottom_row:
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
        self.switch_to_correction_mode()


    def onclick_first_image(self):
        self.display_img(0)


    def onclick_last_image(self):
        self.display_img(self.core.max_img_id)


    def onclick_prev_image(self):
        if self.correction_skip_enabled():
            self.display_img(self.core.next_error(self.img_id, backward=True))
        else:
            self.display_img(max(self.img_id - 1, 0))


    def onclick_next_image(self):
        if self.correction_skip_enabled():
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
            self.switch_to_heatmap_mode()
        if event.key() == Qt.Key_I:
            self.switch_to_image_mode()
        if event.key() == Qt.Key_X:
            self.switch_to_pose_mode()
        if event.key() == Qt.Key_C:
            self.switch_to_correction_mode()
        if event.key() == Qt.Key_T:
            self.onclick_save_pose()

        
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


    def uncheck_mode_buttons(self):
        self.button_correction_mode.setChecked(False)
        self.button_heatmap_mode.setChecked(False)
        self.button_image_mode.setChecked(False)
        self.button_pose_mode.setChecked(False)


    def switch_to_image_mode(self):
        self.uncheck_mode_buttons()
        self.button_image_mode.setChecked(True)
        self.combo_joint_id.setEnabled(False)
        self.checkbox_solve_bp.setEnabled(False)
        self.checkbox_correction_skip.setEnabled(False)
        self.belief_propagation_enabled = lambda: False
        self.correction_skip_enabled = lambda: False
        self.display_method = lambda c,i,j: self.core.get_image(c, i)
        self.update_frame()


    def switch_to_pose_mode(self):
        if not self.core.has_pose():
            return False
        self.uncheck_mode_buttons()
        self.button_pose_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.checkbox_solve_bp.setEnabled(False)
        self.checkbox_correction_skip.setEnabled(False)
        self.belief_propagation_enabled = lambda: False
        self.correction_skip_enabled = lambda: False
        self.display_method = lambda c,i,j: self.core.plot_2d(c, i, joints=j)
        self.update_frame()


    def switch_to_correction_mode(self):
        if not self.core.has_pose():
            return False
        self.uncheck_mode_buttons()
        self.button_correction_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.checkbox_solve_bp.setEnabled(True)
        self.checkbox_correction_skip.setEnabled(True)
        self.belief_propagation_enabled = lambda: self.checkbox_solve_bp.isChecked()
        self.correction_skip_enabled = \
            lambda: self.core.has_calibration() and self.checkbox_correction_skip.isChecked()
        self.display_method = lambda c,i,j: self.core.plot_2d(c, i, with_corrections=True, joints=j)
        self.update_frame()


    def switch_to_heatmap_mode(self):
        if not self.core.has_heatmap():
            return False
        self.uncheck_mode_buttons()
        self.button_heatmap_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.checkbox_solve_bp.setEnabled(False)
        self.checkbox_correction_skip.setEnabled(False)
        self.belief_propagation_enabled = lambda: False
        self.correction_skip_enabled = lambda: False
        self.display_method = lambda c,i,j: self.core.plot_heatmap(c, i, joints=j)
        self.update_frame()


    def display_img(self, img_id):
        self.img_id = img_id
        self.textbox_img_id.setText(str(self.img_id))
        self.update_frame()


    def update_frame(self):
        if self.belief_propagation_enabled():
            self.core.solve_bp(self.img_id)

        for iv in self.image_views:
            self.update_image_view(iv)
            

    def update_image_view(self, iv):
        joints_to_display = self.combo_joint_id.currentData()
        image_array = self.display_method(iv.cam_id, self.img_id, joints_to_display)
        im = image_array.astype(np.uint8)
        height, width, _ = im.shape
        bytesPerLine = 3 * width
        qIm = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
        iv.setPixmap(QPixmap.fromImage(qIm))


    def eventFilter(self, iv, e):
        """ Event filter listening to ImageView's mouse events to handle manual corrections """

        left_press = e.type() == QEvent.MouseButtonPress and e.buttons() == Qt.LeftButton
        left_move = e.type() == QEvent.MouseMove and e.buttons() == Qt.LeftButton
        correction_mode = self.button_correction_mode.isChecked()
        
        if correction_mode and (left_press or left_move):
            x = int(e.x() * self.core.image_shape[0] / iv.frameGeometry().width())
            y = int(e.y() * self.core.image_shape[1] / iv.frameGeometry().height())

            if left_press:
                self.joint_being_corrected = self.core.nearest_joint(iv.cam_id, self.img_id, x, y)
                return False

            elif left_move:
                self.core.move_joint(iv.cam_id, self.img_id, self.joint_being_corrected, x, y)
                self.update_image_view(iv)
                return False

        return super().eventFilter(iv, e)


if __name__ == "__main__":
    main()