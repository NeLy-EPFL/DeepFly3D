import pickle
import sys
from itertools import chain
import re

from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (QWidget, QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, 
                            QCheckBox, QPushButton, QLineEdit, QComboBox, QInputDialog, QMessageBox)
from sklearn.neighbors import NearestNeighbors

from deepfly.pose3d.procrustes.procrustes import procrustes_seperate
from .CameraNetwork import CameraNetwork
from .State import State, Mode
from .util.os_util import *
from deepfly.core import Core
from .ImagePose import ImagePose

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
        self.state = None
        self.core = None


    def setup(self, input_folder=None, num_images_max=None):
        self.core = Core(input_folder or self.prompt_for_directory(), num_images_max)
        self.state = State(self.core.input_folder, num_images_max, self.core.output_folder)
    
        self.setup_layout()
        self.display_img(self.state.img_id)
        self.set_mode(self.state.mode)
        

    def setup_layout(self):
        # Create checkboxes
        self.checkbox_solve_bp = QCheckBox("Correction", self)
        self.checkbox_solve_bp.stateChanged.connect(self.onclick_checkbox_automatic)
        self.checkbox_solve_bp.setChecked(self.state.solve_bp)
        self.checkbox_correction_skip = QCheckBox("Skip", self)
        self.checkbox_correction_skip.setChecked(self.state.correction_skip)
        self.checkbox_correction_skip.stateChanged.connect(self.onclick_checkbox_correction)
        
        # Create buttons
        self.button_first           = self.make_button("<<",          self.onclick_first_image)
        self.button_prev            = self.make_button("<",           self.onclick_prev_image)
        self.button_next            = self.make_button(">",           self.onclick_next_image)
        self.button_last            = self.make_button(">>",          self.onclick_last_image)
        self.button_pose_mode       = self.make_button("Pose",        lambda b: self.set_mode(self.state.mode.POSE))
        self.button_image_mode      = self.make_button("Image",       lambda b: self.set_mode(self.state.mode.IMAGE))
        self.button_heatmap_mode    = self.make_button("Prob. Map",   lambda b: self.set_mode(self.state.mode.HEATMAP))
        self.button_correction_mode = self.make_button("Correction",  lambda b: self.set_mode(self.state.mode.CORRECTION))
        button_textbox_img_id_go    = self.make_button("Go",          self.read_img_id_from_textbox)
        self.button_pose_save       = self.make_button("Save",        self.onclick_save_pose)
        self.button_calibrate_calc  = self.make_button("Calibration", self.onclick_calibrate)
        self.button_camera_order    = self.make_button("Camera ordering", self.onclick_camera_order)
        self.button_pose_estimate   = self.make_button("2D Pose Estimation", self.onclick_pose2d_estimation)
        #
        self.button_image_mode.setCheckable(True) 
        self.button_correction_mode.setCheckable(True)
        self.button_pose_mode.setCheckable(True)
        self.button_heatmap_mode.setCheckable(True)
        
        # Create combo list
        self.textbox_img_id = QLineEdit(str(self.state.img_id), self)
        self.textbox_img_id.setFixedWidth(100)
        self.combo_joint_id = QComboBox(self)
        self.combo_joint_id.addItem("All")
        for i in range(config["skeleton"].num_joints):
            self.combo_joint_id.addItem("Prob. Map: " + str(i))
        self.combo_joint_id.activated[str].connect(self.onactivate_combo)
        self.combo_joint_id.setFixedWidth(100)
        
        # Create images
        self.image_pose_list = [
            ImagePose(self.state, self.core, self.core.camNetLeft[0], self.solve_bp),
            ImagePose(self.state, self.core, self.core.camNetLeft[1], self.solve_bp),
            ImagePose(self.state, self.core, self.core.camNetLeft[2], self.solve_bp),
        ]
        #
        self.image_pose_list_bot = [
            ImagePose(self.state, self.core, self.core.camNetRight[0], self.solve_bp),
            ImagePose(self.state, self.core, self.core.camNetRight[1], self.solve_bp),
            ImagePose(self.state, self.core, self.core.camNetRight[2], self.solve_bp),
        ]

        # Layouts
        layout_h_images = QHBoxLayout()
        layout_h_images.setSpacing(1)
        layout_h_images_bot = QHBoxLayout()
        layout_h_images.setSpacing(1)
        for image_pose in self.image_pose_list:
            layout_h_images.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())
        for image_pose in self.image_pose_list_bot:
            layout_h_images_bot.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())

        layout_h_buttons_top = QHBoxLayout()
        layout_h_buttons_top.setSpacing(3)
        layout_h_buttons_top.setAlignment(Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_estimate,  alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_pose_save,      alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_calibrate_calc, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(self.button_camera_order,   alignment=Qt.AlignLeft)
        layout_h_buttons_top.addStretch()

        layout_h_buttons_top.addWidget(self.button_heatmap_mode,      alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_image_mode,        alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_mode,         alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_correction_mode,   alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.checkbox_correction_skip, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.checkbox_solve_bp,        alignment=Qt.AlignRight)

        layout_h_buttons = QHBoxLayout()
        layout_h_buttons.setSpacing(1)
        layout_h_buttons.addWidget(self.button_first)
        layout_h_buttons.addWidget(self.button_prev)
        layout_h_buttons.addWidget(self.button_next)
        layout_h_buttons.addWidget(self.button_last)
        layout_h_buttons.addWidget(self.textbox_img_id)
        layout_h_buttons.addWidget(button_textbox_img_id_go)
        layout_h_buttons.addStretch()

        layout_h_buttons_second = QHBoxLayout()
        layout_h_buttons_second.setAlignment(Qt.AlignLeft)
        layout_h_buttons_second.setSpacing(1)
        layout_h_buttons_second.addWidget(self.combo_joint_id, alignment=Qt.AlignRight)
        
        layout_v = QVBoxLayout()
        layout_v.addLayout(layout_h_buttons_top)
        layout_v.addLayout(layout_h_images)
        layout_v.addLayout(layout_h_images_bot)
        layout_v.addLayout(layout_h_buttons)
        layout_v.addLayout(layout_h_buttons_second)
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
            msgBox.setText("Wrong format, ordering not changed.")
            msgBox.exec()


    def onclick_checkbox_automatic(self, state):
        self.state.solve_bp = (state == Qt.Checked)
        self.solve_bp()


    def onclick_checkbox_correction(self, state):
        self.state.correction_skip = (state == Qt.Checked)
        

    def onclick_pose2d_estimation(self):
        self.core.pose2d_estimation()
        self.set_mode(Mode.POSE)

        for ip in chain(self.image_pose_list, self.image_pose_list_bot):
            ip.cam = self.core.camNetAll[ip.cam.cam_id]

        self.update_frame()


    def onactivate_combo(self, text):
        if text == "All":
            self.set_heatmap_joint_id(-1)
        else:
            self.set_heatmap_joint_id(int(text.replace("Prob. Map: ", "")))
        self.setFocus()


    def onclick_first_image(self):
        self.display_img(0)


    def onclick_last_image(self):
        self.display_img(self.core.max_img_id)


    def onclick_prev_image(self):
        if (self.state.mode == Mode.CORRECTION 
            and self.state.correction_skip 
            and self.core.camNetLeft.has_calibration() 
            and self.core.camNetLeft.has_pose()
            ):
            self.display_img(self.core.next_error(self.state.img_id, step=-1))
        else:
            self.display_img(max(self.state.img_id - 1, 0))


    def onclick_next_image(self):
        if (self.state.mode == Mode.CORRECTION 
            and self.state.correction_skip 
            and self.core.camNetLeft.has_calibration() 
            and self.core.camNetLeft.has_pose()
            ):
            self.display_img(self.core.next_error(self.state.img_id, step=+1))
        else:
            self.display_img(min(self.core.max_img_id, self.state.img_id + 1))


    def onclick_calibrate(self):
        try:
            [min_img_id, max_img_id] = self.prompt_for_calibration_range()
        except BaseException:
            min_img_id, max_img_id = 0, self.core.max_img_id
        self.core.calibrate_calc(self, min_img_id, max_img_id)


    def onclick_save_pose(self):
        pts2d = np.zeros((7, self.core.num_images, config["num_joints"], 2), dtype=float)
        # pts3d = np.zeros((self.cfg.num_images, self.cfg.num_joints, 3), dtype=float)

        for cam in self.core.camNetAll:
            pts2d[cam.cam_id, :] = cam.points2d.copy()

        # overwrite by manual correction
        count = 0
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.core.num_images):
                if self.state.db.has_key(cam_id, img_id):
                    pt = self.state.db.read(cam_id, img_id) * config["image_shape"]
                    pts2d[cam_id, img_id, :] = pt
                    count += 1

        if "fly" in config["name"]:
            # some post-processing for body-coxa
            for cam_id in range(len(self.core.camNetAll.cam_list)):
                for j in range(config["skeleton"].num_joints):
                    if config["skeleton"].camera_see_joint(cam_id, j) and config[
                        "skeleton"
                    ].is_tracked_point(j, config["skeleton"].Tracked.BODY_COXA):
                        pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                        pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])

        dict_merge = self.core.camNetAll.save_network(path=None)
        dict_merge["points2d"] = pts2d

        # take a copy of the current points2d
        pts2d_orig = np.zeros(
            (7, self.core.num_images, config["num_joints"], 2), dtype=float
        )
        for cam_id in range(config["num_cameras"]):
            pts2d_orig[cam_id, :] = self.core.camNetAll[cam_id].points2d.copy()

        # ugly hack to temporarly incorporate manual corrections
        c = 0
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.core.num_images):
                if self.state.db.has_key(cam_id, img_id):
                    pt = self.state.db.read(cam_id, img_id) * config["image_shape"]
                    self.core.camNetAll[cam_id].points2d[img_id, :] = pt
                    c += 1
        print("Replaced points2d with {} manual correction".format(count))

        # do the triangulation if we have the calibration
        if self.core.camNetLeft.has_calibration() and self.core.camNetLeft.has_pose():
            self.core.camNetAll.triangulate()
            pts3d = self.core.camNetAll.points3d_m

            dict_merge["points3d"] = pts3d
            
        # apply procrustes
        if config["procrustes_apply"]:
            print("Applying Procrustes on 3D Points")
            dict_merge["points3d"] = procrustes_seperate(dict_merge["points3d"])

        # put old values back
        for cam_id in range(config["num_cameras"]):
            self.core.camNetAll[cam_id].points2d = pts2d_orig[cam_id, :].copy()

        pickle.dump(
            dict_merge,
            open(
                os.path.join(
                    self.core.output_folder,
                    "pose_result_{}.pkl".format(self.core.input_folder.replace("/", "_")),
                ),
                "wb",
            ),
        )
        print(
            "Saved the pose at: {}".format(
                os.path.join(
                    self.core.output_folder,
                    "pose_result_{}.pkl".format(self.core.input_folder.replace("/", "_")),
                )
            )
        )


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.read_img_id_from_textbox()
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
        if (   (mode == Mode.POSE       and self.core.camNetLeft.has_pose()     and self.core.camNetRight.has_pose() )
            or (mode == Mode.HEATMAP    and self.core.camNetLeft.has_heatmap())
            or  mode == Mode.IMAGE
            or (mode == Mode.CORRECTION and self.core.camNetLeft.has_pose())
        ):
            self.state.mode = mode
        else:
            print("Cannot set mode: {}".format(mode))
        
        if self.state.mode == Mode.CORRECTION:
            self.display_img(self.state.img_id)
        
        self.update_frame()

        self.button_correction_mode.setChecked(self.state.mode == Mode.CORRECTION)
        self.button_heatmap_mode.setChecked(self.state.mode == Mode.HEATMAP)
        self.button_image_mode.setChecked(self.state.mode == Mode.IMAGE)
        self.button_pose_mode.setChecked(self.state.mode == Mode.POSE)


    def update_frame(self):
        for image_pose in chain(self.image_pose_list, self.image_pose_list_bot):
            image_pose.update_image_pose()
        

    def display_img(self, img_id):
        self.state.img_id = img_id

        for ip in chain(self.image_pose_list, self.image_pose_list_bot):
            ip.clear_manual_corrections()
        
        if self.state.mode == Mode.CORRECTION:
            for ip in chain(self.image_pose_list, self.image_pose_list_bot):
                pt = self.state.db.read(ip.cam.cam_id, self.state.img_id)
                modified_joints = self.state.db.read_modified_joints(ip.cam.cam_id, self.state.img_id)
                if pt is None:
                    pt = ip.cam.points2d[self.state.img_id, :]
                else:
                    pt *= config["image_shape"]

                manual_correction = dict()
                for joint_id in modified_joints:
                    manual_correction[joint_id] = pt[joint_id]
                
                ip.update_manual_corrections(
                    pt,
                    ip.state.img_id,
                    joint_id=None,
                    manual_correction=manual_correction,
                )

            self.solve_bp()

        self.update_frame()
        self.textbox_img_id.setText(str(self.state.img_id))


    # ------------------------------------------------------------------


    def solve_bp(self, save_correction=False):
        if not (
            self.state.mode == Mode.CORRECTION
            and self.state.solve_bp
            and self.core.camNetLeft.has_calibration()
            and self.core.camNetLeft.has_pose()
        ):
            return

        # --------------
        # Compute prior
        prior = list()
        for ip in self.image_pose_list:
            for (joint_id, pt2d) in ip.manual_corrections().items():
                prior.append((ip.cam.cam_id, joint_id, pt2d / config["image_shape"]))
    
        pts_bp = self.core.camNetLeft.solveBP(self.state.img_id, config["bone_param"], prior=prior)
        pts_bp = np.array(pts_bp)

        # --------------
        # set points which are not estimated by bp
        for idx, ip in enumerate(self.image_pose_list):
            
            pts_bp_rep = self.state.db.read(ip.cam.cam_id, self.state.img_id)
            if pts_bp_rep is None:
                pts_bp_rep = ip.cam.points2d[self.state.img_id, :]
            else:
                pts_bp_rep *= config["image_shape"]
            
            pts_bp_ip = pts_bp[idx] * config["image_shape"]
            pts_bp_ip[pts_bp_ip == 0] = pts_bp_rep[pts_bp_ip == 0]

            # keep track of the manually corrected points
            ip.update_manual_corrections(pts_bp_ip, ip.state.img_id, None, ip.manual_corrections())
        self.update_frame()

        # save down corrections as training if any priors were given
        if prior and save_correction:
            print("Saving with prior")
            for ip in self.image_pose_list:
                ip.save_correction()

        print("Finished Belief Propagation")


    def set_heatmap_joint_id(self, joint_id):
        self.state.heatmap_joint_id = joint_id
        self.update_frame()


    def read_img_id_from_textbox(self):
        try:
            img_id = int(self.textbox_img_id.text().replace("Heatmap: ", ""))
            self.display_img(img_id)
        except BaseException as e:
            print("Textbox img id is not integer {}".format(str(e)))


    def set_joint_id_tb(self):
        self.set_heatmap_joint_id(int(self.textbox_joint_id.text()))



if __name__ == "__main__":
    main()