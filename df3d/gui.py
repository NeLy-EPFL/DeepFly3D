import numpy as np
import re

from PyQt5 import QtWidgets as QW
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap

from df3d.core import Core


def main():
    """Main entry point to run the GUI."""
    import sys
    cli_args = parse_cli_args(sys.argv)

    app = QW.QApplication([])
    window = DeepflyGUI()
    window.setup(**cli_args)
    window.set_width(app.desktop().size().width())
    window.show()
    app.exec_()

    
def parse_cli_args(argv):
    """Parses the argument string argv.

    Returns:
    A simple namespace with the parsed arguments values.
    """
    args = {}
    args['output_subfolder'] = 'df3d'
    try:
        args['input_folder'] = argv[1]
        args['num_images_max'] = int(argv[2])
        args['output_subfolder'] = argv[3]
    except (IndexError, ValueError):
        pass
    return args


class DeepflyGUI(QW.QWidget):
    """Graphical User Interface Widget for DeepFly."""

    def __init__(self):
        QW.QWidget.__init__(self)
        self.img_id = 0
        self.core = None


    def setup(
        self,
        input_folder=None,
        output_subfolder=None,
        num_images_max=None):
        """Configures the interface and prompts user for missing data."""

        if not input_folder:
            input_folder = self.prompt_for_directory()

        if not output_subfolder:
            output_subfolder = self.prompt_output_subdirectory_name()

        self.core = Core(input_folder, output_subfolder, num_images_max, None)
        self.setup_layout()
        self.onclick_image_mode()


    def set_width(self, width):
        """Sets the GUI's window's width and resizes its height accordingly."""

        hw_ratio = self.core.image_shape[0] * 1.2 / self.core.image_shape[1]
        height = int(width / hw_ratio)
        self.resize(width, height)


    def setup_layout(self):
        """Creates the GUI layout (buttons and stuff.)"""

        # --- Create checkboxes ---
        self.checkbox_solve_bp = QW.QCheckBox("Auto-correct", self)
        self.checkbox_solve_bp.setChecked(True)
        self.checkbox_solve_bp.stateChanged.connect(self.update_frame)

        # --- Create buttons ---
        def mb(text, onClick):
            b = QW.QPushButton(text, self)
            b.setMaximumWidth(b.fontMetrics().boundingRect(text).width() + 27)
            b.clicked.connect(onClick)
            return b

        self.button_prev = mb("<", self.onclick_prev_image)
        self.button_next = mb(">", self.onclick_next_image)
        self.button_last = mb(">>", self.onclick_last_image)
        self.button_first = mb("<<", self.onclick_first_image)
        self.button_prev_err = mb("< previous error", self.onclick_prev_error)
        self.button_next_err = mb("next error >", self.onclick_next_error)
        self.button_pose_save = mb("Save", self.onclick_save_pose)
        self.button_pose_mode = mb("Pose", self.onclick_pose_mode)
        self.button_image_mode = mb("Image", self.onclick_image_mode)
        self.button_heatmap_mode = mb("Prob. Map", self.onclick_heatmap_mode)
        button_textbox_img_id_go = mb("Go",  self.onclick_goto_img)
        self.button_calibrate_calc = mb("Calibration", self.onclick_calibrate)

        self.button_correction_mode = \
            mb("Correction", self.onclick_correction_mode)
        self.button_camera_order = \
            mb("Camera ordering", self.onclick_camera_order)
        self.button_pose_estimate = \
            mb("2D Pose Estimation", self.onclick_pose2d_estimation)

        self.button_image_mode.setCheckable(True)
        self.button_correction_mode.setCheckable(True)
        self.button_pose_mode.setCheckable(True)
        self.button_heatmap_mode.setCheckable(True)

        self.textbox_img_id = QW.QLineEdit(str(self.img_id), self)
        self.textbox_img_id.setFixedWidth(100)

        self.combo_joint_id = QW.QComboBox(self)
        self.combo_joint_id.addItem("View all joints", [])
        for i in range(self.core.number_of_joints):
            self.combo_joint_id.addItem(f'View joint {i}', [i])
        self.combo_joint_id.activated[str].connect(self.update_frame)

        # --- Create widgets to display images and pose results ---
        def make_image_view(cam_id):
            iv = QW.QLabel()
            iv.setScaledContents(True)
            iv.cam_id = cam_id
            return iv

        top_row    = [make_image_view(cam_id) for cam_id in [0, 1, 2]]
        bottom_row = [make_image_view(cam_id) for cam_id in [4, 5, 6]]
        self.image_views = top_row + bottom_row

        for image_view in self.image_views:
            image_view.installEventFilter(self)

        # --- Layouts ---
        layout_h_images = QW.QHBoxLayout()
        layout_h_images.setSpacing(1)
        for image_pose in top_row:
            layout_h_images.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())

        layout_h_images_bot = QW.QHBoxLayout()
        layout_h_images_bot.setSpacing(1)
        for image_pose in bottom_row:
            layout_h_images_bot.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())

        layout_h_buttons_top = QW.QHBoxLayout()
        layout_h_buttons_top.setSpacing(3)
        layout_h_buttons_top.setAlignment(Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_estimate)
        layout_h_buttons_top.addWidget(self.button_calibrate_calc)
        layout_h_buttons_top.addWidget(self.button_camera_order)
        layout_h_buttons_top.addWidget(self.button_pose_save)
        layout_h_buttons_top.addStretch()
        layout_h_buttons_top.addWidget(self.button_image_mode)
        layout_h_buttons_top.addWidget(self.button_pose_mode)
        layout_h_buttons_top.addWidget(self.button_correction_mode)
        #layout_h_buttons_top.addWidget(self.button_heatmap_mode)

        layout_h_buttons = QW.QHBoxLayout()
        layout_h_buttons.setSpacing(1)
        layout_h_buttons.addWidget(self.button_first)
        layout_h_buttons.addWidget(self.button_prev)
        layout_h_buttons.addWidget(self.button_next)
        layout_h_buttons.addWidget(self.button_last)
        layout_h_buttons.addWidget(self.textbox_img_id)
        layout_h_buttons.addWidget(button_textbox_img_id_go)
        layout_h_buttons.addStretch()
        layout_h_buttons.addWidget(self.button_prev_err)
        layout_h_buttons.addWidget(self.button_next_err)
        layout_h_buttons.addStretch()
        layout_h_buttons.addWidget(self.checkbox_solve_bp)
        layout_h_buttons.addStretch()
        layout_h_buttons.addWidget(self.combo_joint_id)

        layout_v = QW.QVBoxLayout()
        layout_v.addLayout(layout_h_buttons_top)
        layout_v.addLayout(layout_h_images)
        layout_v.addLayout(layout_h_images_bot)
        layout_v.addLayout(layout_h_buttons)
        layout_v.setSpacing(0)

        self.setLayout(layout_v)
        self.setWindowTitle(self.core.input_folder)


    # ------------------------------------------------------------------
    # user input


    def onclick_camera_order(self):
        """Prompts for a new camera ordering."""

        cidread2cid = self.prompt_for_camera_ordering()
        if cidread2cid is None:  # canceled
            return
        if self.core.update_camera_ordering(cidread2cid):
            self.update_frame()
        else:
            self.display_error_message("Ordering not changed (wrong format).")


    def onclick_pose2d_estimation(self):
        """Runs the core's pose2d_estimation routine and switches to correction mode."""
        self.core.pose2d_estimation()
        self.onclick_correction_mode()


    def onclick_first_image(self):
        """Displays the first image."""
        self.display_img(0)


    def onclick_last_image(self):
        """Displays the last image."""
        self.display_img(self.core.max_img_id)


    def onclick_prev_image(self):
        """Displays the previous image if is exists."""
        self.display_img(max(self.img_id - 1, 0))


    def onclick_next_image(self):
        """Displays the next image if it exists."""
        self.display_img(min(self.core.max_img_id, self.img_id + 1))


    def onclick_prev_error(self):
        """Among previous images, displays the last one with an error, if it exists."""
        prev_img = self.core.prev_error(self.img_id)
        if prev_img is not None:
            self.display_img(prev_img)
        else:
            msg = 'No error remaining among previous images'
            self.display_error_message(msg)


    def onclick_next_error(self):
        """Among next images, displays the first one with an error, if it exists."""
        next_img = self.core.next_error(self.img_id)
        if next_img is not None:
            self.display_img(next_img)
        else:
            msg = 'No error remaining among next images'
            self.display_error_message(msg)


    def onclick_calibrate(self):
        """Prompts for a calibration range and calibrates."""
        rng = self.prompt_for_calibration_range()
        if rng is not None:  # not canceled
            self.core.calibrate_calc(*rng)


    def onclick_save_pose(self):
        """Saves the pose estimation results and manual corrections."""
        self.core.save_pose()
        self.core.save_corrections()


    def onclick_goto_img(self):
        """Displays the image whose id is in the textbox, if valid id."""
        try:
            img_id = int(self.textbox_img_id.text())
            self.display_img(img_id)
            self.setFocus()
        except BaseException:
            msg = "Textbox content should be an integer image id"
            self.display_error_message(msg)
            self.textbox_img_id.setText(str(self.img_id))


    def onclick_image_mode(self):
        """Switches the GUI to image mode."""
        self.uncheck_mode_buttons()
        self.button_image_mode.setChecked(True)
        self.combo_joint_id.setEnabled(False)
        self.correction_controls_enabled(False)
        self.belief_propagation_enabled = lambda: False
        self.display_method = lambda c,i,j: self.core.get_image(c, i)
        self.update_frame()


    def onclick_pose_mode(self):
        """Switches the GUI to pose estimation results mode."""
        if not self.core.has_pose:
            return False
        self.uncheck_mode_buttons()
        self.button_pose_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.correction_controls_enabled(False)
        self.belief_propagation_enabled = lambda: False
        self.display_method = lambda c,i,j: self.core.plot_2d(c, i, joints=j)
        self.update_frame()


    def onclick_correction_mode(self):
        """Switches the GUI to manual corrections mode."""
        if not self.core.has_pose:
            return False
        self.uncheck_mode_buttons()
        self.button_correction_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.correction_controls_enabled(True)

        self.belief_propagation_enabled = lambda: \
            self.checkbox_solve_bp.isChecked()

        self.display_method = lambda c,i,j: \
            self.core.plot_2d(c, i, with_corrections=True, joints=j)

        self.update_frame()


    def onclick_heatmap_mode(self):
        """Switches the GUI to heatmap mode if heatmaps are available."""
        if not self.core.has_heatmap:
            return False
        self.uncheck_mode_buttons()
        self.button_heatmap_mode.setChecked(True)
        self.combo_joint_id.setEnabled(True)
        self.correction_controls_enabled(False)
        self.belief_propagation_enabled = lambda: False

        self.display_method = lambda c,i,j: \
            self.core.plot_heatmap(c, i, joints=j)

        self.update_frame()


    def keyPressEvent(self, event):
        switch = {
            Qt.Key_Return: self.onclick_goto_img,
            Qt.Key_A: self.onclick_prev_image,
            Qt.Key_D: self.onclick_next_image,
            Qt.Key_H: self.onclick_heatmap_mode,
            Qt.Key_I: self.onclick_image_mode,
            Qt.Key_X: self.onclick_pose_mode,
            Qt.Key_C: self.onclick_correction_mode,
            Qt.Key_T: self.onclick_save_pose,
        }
        default_action = lambda: None
        action = switch.get(event.key(), default_action)
        action()


    # ------------------------------------------------------------------
    # user feedback and prompt


    def prompt_for_directory(self):
        """Prompts for a directory and returns its path.

        Returns:
        String: path to a directory.
        """
        return str(QW.QFileDialog.getExistingDirectory(
                self,
                directory="./",
                caption="Select Directory",
                options=QW.QFileDialog.DontUseNativeDialog,
            ))


    def prompt_output_subdirectory_name(self):
        """Prompts for the ouput subdirectory name.

        Returns:
        String: the name of the subdirectory in which to write output.
        """
        ok_pressed = False
        while not ok_pressed:
            text, ok_pressed = QW.QInputDialog.getText(
                self, "Name of output sub-directory", "Name:",
                QW.QLineEdit.Normal, "df3d"
            )
        return str(text)


    def prompt_for_camera_ordering(self):
        """Prompts for a camera ordering.

        Returns:
        list[int]: a list of camera ids.
        """
        text, ok_pressed = QW.QInputDialog.getText(
            self, "Rename Images", "Camera order:", QW.QLineEdit.Normal, ""
        )
        if ok_pressed:
            cidread2cid = re.findall(r'[0-9]+', text)
            cidread2cid = [int(x) for x in cidread2cid]
            return cidread2cid


    def prompt_for_calibration_range(self):
        """Prompts for a range of ids to use for calibration.

        Returns:
        pair[int, int]: (min_id, max_id)
        """
        text, okPressed = QW.QInputDialog.getText(
            self, "Calibration", "Range of images:", QW.QLineEdit.Normal,
            f"0-{self.core.max_img_id}"
        )
        if okPressed:
            numbers = re.findall(r'[0-9]+', text)
            numbers = [int(x) for x in numbers]
            if len(numbers) == 2:
                return numbers[0], numbers[1]
            else:
                msg = 'Please provide a range such as: 0-10'
                self.display_error_message(msg)


    def display_error_message(self, message):
        """Displays an error message to user."""
        msgBox = QW.QMessageBox()
        msgBox.setText(message)
        msgBox.exec()


    # ------------------------------------------------------------------


    def uncheck_mode_buttons(self):
        self.button_correction_mode.setChecked(False)
        self.button_heatmap_mode.setChecked(False)
        self.button_image_mode.setChecked(False)
        self.button_pose_mode.setChecked(False)


    def correction_controls_enabled(self, enabled):
        self.button_next_err.setEnabled(enabled)
        self.button_prev_err.setEnabled(enabled)
        self.checkbox_solve_bp.setEnabled(enabled)


    def display_img(self, img_id):
        self.img_id = img_id
        self.textbox_img_id.setText(str(self.img_id))
        self.update_frame()


    def update_frame(self):
        #if self.belief_propagation_enabled():
        #    self.core.solve_bp(self.img_id)

        for iv in self.image_views:
            self.update_image_view(iv)

    def update_image_view(self, iv):
        joints = self.combo_joint_id.currentData()
        image_array = self.display_method(iv.cam_id, self.img_id, joints)
        im = image_array.astype(np.uint8)
        height, width, _ = im.shape
        bytesPerLine = 3 * width
        qIm = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qIm)
        pixmap = pixmap.scaledToWidth(400)
        iv.setPixmap(pixmap)


    def eventFilter(self, iv, e):
        """ Event filter listening to the image views mouse events
            to handle manual corrections
        """
        MousePress = QEvent.MouseButtonPress
        MouseMove = QEvent.MouseMove
        left_press = e.type() == MousePress and e.buttons() == Qt.LeftButton
        left_move = e.type() == MouseMove and e.buttons() == Qt.LeftButton
        correction_mode = self.button_correction_mode.isChecked()

        if correction_mode and (left_press or left_move):
            frame = iv.frameGeometry()
            x = int(e.x() * self.core.image_shape[0] / frame.width())
            y = int(e.y() * self.core.image_shape[1] / frame.height())

            if left_press:
                joint = self.core.nearest_joint(iv.cam_id, self.img_id, x, y)
                self.joint_being_corrected = joint
                return False

            elif left_move:
                joint = self.joint_being_corrected
                self.core.move_joint(iv.cam_id, self.img_id, joint, x, y)
                self.update_image_view(iv)
                return False

        return super().eventFilter(iv, e)


if __name__ == "__main__":
    main()