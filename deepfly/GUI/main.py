import pickle
import sys
from itertools import chain
import re

from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, \
                            QCheckBox, QPushButton, QLineEdit, QComboBox, QInputDialog, QMessageBox
from sklearn.neighbors import NearestNeighbors

from deepfly.pose3d.procrustes.procrustes import procrustes_seperate
from .CameraNetwork import CameraNetwork
from .State import State, View, Mode
from .util.optim_util import energy_drosoph
from .util.os_util import *
from deepfly.core import Core


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
        self.set_pose(self.state.img_id)
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
        pts2d = np.zeros(
            (7, self.core.num_images, config["num_joints"], 2), dtype=float
        )
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
        if event.key() == Qt.Key_L:
            self.set_view(View.Left)
        if event.key == Qt.Key_R:
            self.set_view(View.Right)


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


    def display_img(self, img_id):
        self.state.already_corrected = self.already_corrected(self.state.view, img_id)
        self.set_pose(img_id)


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
            self.set_pose(self.state.img_id)
        
        self.update_frame()

        self.button_correction_mode.setChecked(self.state.mode == Mode.CORRECTION)
        self.button_heatmap_mode.setChecked(self.state.mode == Mode.HEATMAP)
        self.button_image_mode.setChecked(self.state.mode == Mode.IMAGE)
        self.button_pose_mode.setChecked(self.state.mode == Mode.POSE)


    def update_frame(self):
        for image_pose in chain(self.image_pose_list, self.image_pose_list_bot):
            image_pose.update_image_pose()
        

    def set_pose(self, img_id):
        self.state.img_id = img_id

        for ip in chain(self.image_pose_list, self.image_pose_list_bot):
            ip.clear_mc()
        
        if self.state.mode == Mode.CORRECTION:
            for ip in chain(self.image_pose_list, self.image_pose_list_bot):
                pt = self.state.db.read(ip.cam.cam_id, self.state.img_id)
                modified_joints = self.state.db.read_modified_joints(
                    ip.cam.cam_id, self.state.img_id
                )
                if pt is None:
                    pt = ip.cam.points2d[self.state.img_id, :]
                else:
                    pt *= config["image_shape"]

                manual_correction = dict()
                for joint_id in modified_joints:
                    manual_correction[joint_id] = pt[joint_id]
                
                ip.dynamic_pose = DynamicPose(
                    pt,
                    ip.state.img_id,
                    joint_id=None,
                    manual_correction=manual_correction,
                )

            if self.core.camNetLeft.has_calibration():
                self.solve_bp()

            if self.core.camNetRight.has_calibration():
                self.solve_bp()

        self.update_frame()
        self.textbox_img_id.setText(str(self.state.img_id))


    # ------------------------------------------------------------------


    def already_corrected(self, view, img_id):
        if view == View.Left:
            return (
                self.state.db.has_key(0, img_id)
                or self.state.db.has_key(1, img_id)
                or self.state.db.has_key(2, img_id)
            )
        elif view == View.Right:
            return (
                self.state.db.has_key(4, img_id)
                or self.state.db.has_key(5, img_id)
                or self.state.db.has_key(6, img_id)
            )
        else:
            raise NotImplementedError

    
    def solve_bp(self, save_correction=False):
        if not (
            self.state.mode == Mode.CORRECTION
            and self.state.solve_bp
            and self.core.camNetLeft.has_calibration()
            and self.core.camNetLeft.has_pose()
        ):
            return

        prior = list()
        for ip in self.image_pose_list:
            if ip.dynamic_pose is not None:
                for (joint_id, pt2d) in ip.dynamic_pose.manual_correction_dict.items():
                    prior.append(
                        (ip.cam.cam_id, joint_id, pt2d / config["image_shape"])
                    )
        # print("Prior for BP: {}".format(prior))
        pts_bp = self.core.camNetLeft.solveBP(
            self.state.img_id, config["bone_param"], prior=prior
        )
        pts_bp = np.array(pts_bp)

        # set points which are not estimated by bp
        for idx, image_pose in enumerate(self.image_pose_list):
            pts_bp_ip = pts_bp[idx] * config["image_shape"]
            pts_bp_rep = self.state.db.read(image_pose.cam.cam_id, self.state.img_id)
            if pts_bp_rep is None:
                pts_bp_rep = image_pose.cam.points2d[self.state.img_id, :]
            else:
                pts_bp_rep *= config["image_shape"]
            pts_bp_ip[pts_bp_ip == 0] = pts_bp_rep[pts_bp_ip == 0]

            # keep track of the manually corrected points
            mcd = (
                image_pose.dynamic_pose.manual_correction_dict
                if image_pose.dynamic_pose is not None
                else None
            )
            image_pose.dynamic_pose = DynamicPose(
                pts_bp_ip, image_pose.state.img_id, joint_id=None, manual_correction=mcd
            )
        self.update_frame()

        # save down corrections as training if any priors were given
        if prior and save_correction:
            print("Saving with prior")
            for ip in self.image_pose_list:
                ip.save_correction()

        print("Finished Belief Propagation")


    def set_heatmap_joint_id(self, joint_id):
        self.state.hm_joint_id = joint_id
        self.update_frame()


    def read_img_id_from_textbox(self):
        try:
            img_id = int(self.textbox_img_id.text().replace("Heatmap: ", ""))
            self.state.already_corrected = self.already_corrected(
                self.state.view, img_id
            )
            self.set_pose(img_id)
        except BaseException as e:
            print("Textbox img id is not integer {}".format(str(e)))


    def set_joint_id_tb(self):
        self.set_heatmap_joint_id(int(self.textbox_joint_id.text()))




class DynamicPose:
    def __init__(self, points2d, img_id, joint_id, manual_correction=None):
        self.points2d = points2d
        self.img_id = img_id
        self.joint_id = joint_id
        self.manual_correction_dict = manual_correction
        if manual_correction is None:
            self.manual_correction_dict = dict()

    def set_joint(self, joint_id, pt2d):
        assert pt2d.shape[0] == 2
        self.points2d[joint_id] = pt2d
        self.manual_correction_dict[joint_id] = pt2d


class ImagePose(QWidget):
    def __init__(self, config, core, cam, f_solve_bp):
        QWidget.__init__(self)
        self.state = config
        self.core = core
        self.cam = cam
        self.dynamic_pose = None

        self.update_image_pose()
        self.show()

        self.f_solve_bp = f_solve_bp


    def clear_mc(self):
        self.dynamic_pose = None


    def update_image_pose(self):
        draw_joints = [
            j
            for j in range(config["skeleton"].num_joints)
            if config["skeleton"].camera_see_joint(self.cam.cam_id, j)
        ]
        corrected_this_camera = self.state.db.has_key(self.cam.cam_id, self.state.img_id)
        zorder = config["skeleton"].get_zorder(self.cam.cam_id)
        
        if self.state.mode == Mode.IMAGE:
            im = self.cam.get_image(self.state.img_id)
        elif self.state.mode == Mode.HEATMAP:
            draw_joints = (
                [self.state.hm_joint_id]
                if self.state.hm_joint_id != -1
                else [
                    j
                    for j in range(config["skeleton"].num_joints)
                    if config["skeleton"].camera_see_joint(self.cam.cam_id, j)
                ]
            )
            im = self.cam.plot_heatmap(
                img_id=self.state.img_id, concat=False, scale=2, draw_joints=draw_joints
            )
        elif self.state.mode == Mode.POSE:
            im = self.cam.plot_2d(
                img_id=self.state.img_id, draw_joints=draw_joints, zorder=zorder
            )
        elif self.state.mode == Mode.CORRECTION:
            circle_color = (0, 255, 0) if corrected_this_camera else (0, 0, 255)

            # calculate the joints with large reprojection error
            r_list = [config["scatter_r"]] * config["num_joints"]
            for joint_id in range(config["skeleton"].num_joints):
                if joint_id not in config["skeleton"].pictorial_joint_list:
                    continue
                camNet = self.core.camNetLeft if (self.cam.cam_id < 3) else self.core.camNetRight
                err_proj = self.core.get_joint_reprojection_error(self.state.img_id, joint_id, camNet)
                if err_proj > config["reproj_thr"][joint_id]:
                    r_list[joint_id] = config["scatter_r"] * 2

            im = self.cam.plot_2d(
                img_id=self.state.img_id,
                pts=self.dynamic_pose.points2d,
                circle_color=circle_color,
                draw_joints=draw_joints,
                zorder=zorder,
                r_list=r_list,
            )
        im = im.astype(np.uint8)
        height, width, channel = im.shape

        bytesPerLine = 3 * width
        qIm = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
        self.im = QPixmap.fromImage(qIm)

        self.update()


    def save_correction(self, thr=30):
        points2d_prediction = self.cam.get_points2d(self.state.img_id)
        points2d_correction = self.dynamic_pose.points2d

        err = np.abs(points2d_correction - points2d_prediction)
        check_joint_id_list = [
            j
            for j in range(config["num_joints"])
            if (j not in config["skeleton"].ignore_joint_id)
            and config["skeleton"].camera_see_joint(self.cam.cam_id, j)
        ]

        for j in check_joint_id_list:
            if np.any(err[j] > thr):
                err_max = np.max(err[check_joint_id_list])
                joint_id, ax = np.where(err == err_max)

                print(
                    "Saving camera {} with l1 {} on joint {}".format(
                        self.cam.cam_id, err_max, joint_id
                    )
                )
                # make sure we are not saving a points that cannot be seen from the camera
                unseen_joints = [
                    j
                    for j in range(config["skeleton"].num_joints)
                    if not config["skeleton"].camera_see_joint(self.cam.cam_id, j)
                ]
                points2d_correction[unseen_joints, :] = 0.0
                self.state.db.write(
                    points2d_correction / config["image_shape"],
                    self.cam.cam_id,
                    self.state.img_id,
                    train=True,
                    modified_joints=list(
                        self.dynamic_pose.manual_correction_dict.keys()
                    ),
                )

                return True

        return False


    def mouseMoveEvent(self, e):
        if self.state.mode == Mode.CORRECTION:
            x = int(
                e.x()
                * np.array(config["image_shape"][0])
                / self.frameGeometry().width()
            )
            y = int(
                e.y()
                * np.array(config["image_shape"][1])
                / self.frameGeometry().height()
            )
            if (
                self.dynamic_pose.joint_id is None
            ):  # find the joint to be dragged with nearest neighbor
                pts = self.dynamic_pose.points2d.copy()

                # make sure we don't select points we cannot see
                pts[
                    [
                        j_id
                        for j_id in range(config["skeleton"].num_joints)
                        if not config["skeleton"].camera_see_joint(
                            self.cam.cam_id, j_id
                        )
                    ]
                ] = [9999, 9999]

                nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(pts)

                _, indices = nbrs.kneighbors(np.array([[x, y]]))
                self.dynamic_pose.joint_id = indices[0][0]
                print("Selecting the joint: {}".format(self.dynamic_pose.joint_id))
            self.dynamic_pose.set_joint(self.dynamic_pose.joint_id, np.array([x, y]))
            self.update_image_pose()


    def mouseReleaseEvent(self, e):
        if self.state.mode == Mode.CORRECTION:
            self.dynamic_pose.joint_id = None  # make sure we forget the tracked joint
            self.update_image_pose()

            # solve BP again
            self.state.db.write(
                self.dynamic_pose.points2d / config["image_shape"],
                self.cam.cam_id,
                self.state.img_id,
                train=True,
                modified_joints=list(self.dynamic_pose.manual_correction_dict.keys()),
            )
            self.f_solve_bp(save_correction=True)
            self.update_image_pose()


    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.im)

        self.update()


if __name__ == "__main__":
    main()
