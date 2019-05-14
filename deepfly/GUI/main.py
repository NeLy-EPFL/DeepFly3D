#!/usr/bin/env python

import ast
import pickle
import sys
from enum import Enum

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from sklearn.neighbors import NearestNeighbors

from .CameraNetwork import CameraNetwork
from .PoseDB import PoseDB
from .im_util import *
from .optical_flow import optical_flow
from .optimLM import energy_drosoph
from .os_util import *


class Config():
    def __init__(self, folder):
        self.folder = folder
        self.mode = Mode.IMAGE
        self.view = View.Left
        self.img_id = 0
        self.hm_joint_id = -1  # -1 corresponds to all joints
        self.db = None
        self.camNet = None
        self.bone_param = None

        self.solve_bp = True  # Automatic correction
        self.already_corrected = False
        self.correction_skip = True  # Correction Skip

        self.num_cameras = 7
        self.num_images = None
        self.num_joints = skeleton.num_joints

        self.image_shape = [960, 480]
        self.heatmap_shape = [64, 128]

        self.max_num_images = None

        self.reproj_thr = {0: 30,
                           1: 30,
                           2: 30,
                           3: 30,
                           4: 30,
                           5: 30,
                           6: 30,
                           7: 30,
                           8: 30,
                           9: 30,
                           10: 30,
                           11: 30,
                           12: 30,
                           13: 30,
                           14: 30,
                           15: 30,
                           16: 30,
                           17: 30,
                           18: 30
                           }

        assert (len(self.reproj_thr) == (skeleton.num_joints // 2))


class Mode(Enum):
    IMAGE = 1
    HEATMAP = 2
    POSE = 3
    CORRECTION = 4


class View(Enum):
    Left = 0
    Right = 1


class DrosophAnnot(QWidget):
    def __init__(self):
        self.chosen_points = []
        QWidget.__init__(self)

        print("Input arguments: {}".format(sys.argv))
        if len(sys.argv) <= 1:
            dialog = QFileDialog.getExistingDirectory(self, directory='/data/', caption="Select Directory",
                                                      options=QFileDialog.DontUseNativeDialog)
            self.folder = str(dialog)
        elif len(sys.argv) >= 2:
            self.folder = str(sys.argv[1])
            self.folder = os.path.abspath(self.folder)
        if len(sys.argv) >= 3:
            max_num_images = int(sys.argv[2])
        else:
            max_num_images = None

        if self.folder.endswith('/'):
            self.folder = self.folder[:-1]

        self.cfg = Config(self.folder)
        self.cfg.max_num_images = max_num_images
        self.cfg.db = PoseDB(self.folder)

        self.cidread2cid, self.cid2cidread = read_camera_order(self.folder)

        max_img_id = get_max_img_id(self.folder)
        self.cfg.num_images = max_img_id + 1
        if self.cfg.max_num_images is not None:
            self.cfg.num_images = min(self.cfg.num_images, self.cfg.max_num_images)
        else:
            self.cfg.num_images = self.cfg.num_images

        print("Number of images: {}".format(self.cfg.num_images))
        self.cfg.bone_param = skeleton.bone_param

        calib = read_calib(self.folder)
        self.camNetAll = CameraNetwork(image_folder=self.folder, cam_id_list=[0, 1, 2, 3, 4, 5, 6],
                                       cid2cidread=self.cid2cidread, num_images=self.cfg.num_images,
                                       calibration=calib, num_joints=self.cfg.num_joints,
                                       heatmap_shape=self.cfg.heatmap_shape)
        self.camNetLeft = CameraNetwork(image_folder=self.folder, cam_id_list=[0, 1, 2, 3],
                                        num_images=self.cfg.num_images, calibration=calib,
                                        num_joints=self.cfg.num_joints, cid2cidread=self.cid2cidread[:4],
                                        heatmap_shape=self.cfg.heatmap_shape, cam_list=self.camNetAll.cam_list[:4])
        self.camNetRight = CameraNetwork(image_folder=self.folder, cam_id_list=[6, 5, 4, 3],
                                         num_images=self.cfg.num_images,
                                         calibration=calib, num_joints=self.cfg.num_joints,
                                         heatmap_shape=self.cfg.heatmap_shape,
                                         cid2cidread=self.cid2cidread[-4::][::-1],
                                         cam_list=self.camNetAll.cam_list[-4::][::-1])
        self.camNetMid = CameraNetwork(image_folder=self.cfg.folder, cam_id_list=[3], num_images=self.cfg.num_images,
                                       calibration=calib, num_joints=self.cfg.num_joints,
                                       heatmap_shape=self.cfg.heatmap_shape, cam_list=[self.camNetAll[3]],
                                       cid2cidread=[self.cid2cidread[3]])

        '''
        # make sure invisible joints are zeroed out
        for cam in self.camNetAll:
            joint_id_list = [j_id for j_id in range(skeleton.num_joints) if
                             not skeleton.camera_see_joint(cam.cam_id_read, j_id)]
            cam.points2d[:, joint_id_list, :] = 0
        '''

        self.camNet = self.camNetLeft
        self.cfg.camNet = self.camNet
        self.camNet.bone_param = self.cfg.bone_param

        # set the layout
        layout_h_images = QHBoxLayout()
        layout_h_buttons = QHBoxLayout()
        layout_h_buttons_second = QHBoxLayout()
        layout_h_buttons_second.setAlignment(Qt.AlignLeft)
        layout_h_buttons_top = QHBoxLayout()
        layout_h_buttons_top.setAlignment(Qt.AlignRight)
        layout_v = QVBoxLayout()
        self.image_pose_list = [ImagePose(self.cfg, self.camNet[0], self.solve_bp),
                                ImagePose(self.cfg, self.camNet[1], self.solve_bp),
                                ImagePose(self.cfg, self.camNet[2], self.solve_bp),
                                ImagePose(self.cfg, self.camNet[3], self.solve_bp)]
        for image_pose in self.image_pose_list:
            layout_h_images.addWidget(image_pose)
            image_pose.resize(image_pose.sizeHint())

        self.button_list_modes = list()
        self.checkbox_solve_bp = QCheckBox("Automatic Correction", self)
        self.checkbox_solve_bp.stateChanged.connect(self.checkbox_automatic_changed)
        if self.cfg.solve_bp:
            self.checkbox_solve_bp.setChecked(True)
        self.checkbox_correction_skip = QCheckBox("Correction Skip", self)
        if self.cfg.correction_skip:
            self.checkbox_correction_skip.setChecked(True)
        self.checkbox_correction_skip.stateChanged.connect(self.checkbox_correction_clicked)
        button_textbox_img_id_go = QPushButton("Go", self)
        button_pose_estimate = QPushButton("2D Pose Estimation", self)
        self.button_set_width(button_pose_estimate, "2D Pose Estimation")
        button_rename_images = QPushButton("Rename Images", self)
        self.button_set_width(button_rename_images, "Rename Images")
        self.button_image_mode = QPushButton("Image", self)
        self.button_image_mode.setCheckable(True)
        self.button_set_width(self.button_image_mode, "Image")
        self.button_heatmap_mode = QPushButton("Heatmap", self)
        self.button_set_width(self.button_heatmap_mode, "Heatmap")
        self.button_pose_mode = QPushButton("Pose", self)
        self.button_set_width(self.button_pose_mode, "Pose")
        self.button_correction_mode = QPushButton("Correction", self)
        self.button_set_width(self.button_correction_mode, "Correction")
        self.button_list_modes.append(self.button_image_mode)
        self.button_list_modes.append(self.button_correction_mode)
        self.button_list_modes.append(self.button_pose_mode)
        self.button_list_modes.append(self.button_heatmap_mode)
        for b in self.button_list_modes:
            b.setCheckable(True)
        self.button_image_mode.setCheckable(True)
        button_first = QPushButton('<<', self)
        button_prev = QPushButton('<', self)
        button_next = QPushButton('>', self)
        button_last = QPushButton('>>', self)
        button_left = QPushButton('L', self)
        button_right = QPushButton('R', self)
        self.button_set_width(button_first, "<<")
        self.button_set_width(button_last, ">>")
        self.button_set_width(button_prev, "<")
        self.button_set_width(button_next, ">")

        self.button_set_width(button_left, "L")
        self.button_set_width(button_right, "R")

        button_calibrate_calc = QPushButton('Calib Calc', self)
        button_pose_save = QPushButton('Pose Save', self)
        #button_optical_flow = QPushButton('Opt Flow', self)

        self.textbox_img_id = QLineEdit(str(self.cfg.img_id), self)
        self.textbox_img_id.setFixedWidth(100)
        self.combo_joint_id = QComboBox(self)
        self.combo_joint_id.addItem("All")
        for i in range(self.cfg.num_joints):
            self.combo_joint_id.addItem("Heatmap: " + str(i))
        self.combo_joint_id.activated[str].connect(self.combo_activated)
        self.combo_joint_id.setFixedWidth(100)
        #self.combo_camera_id = QComboBox(self)
        #for i in range(3):
        #    self.combo_camera_id.addItem(str(i))
        #self.textbox_final_img_id = QLineEdit(str(self.cfg.img_id), self)
        #self.textbox_final_img_id.setFixedWidth(100)

        button_textbox_img_id_go.clicked.connect(self.set_img_id_tb)
        self.button_heatmap_mode.clicked.connect(lambda b: self.set_mode(self.cfg.mode.HEATMAP))
        button_pose_estimate.clicked.connect(self.pose2d_estimation)
        button_rename_images.clicked.connect(self.rename_images)
        self.button_image_mode.clicked.connect(lambda b: self.set_mode(self.cfg.mode.IMAGE))
        self.button_correction_mode.clicked.connect(lambda b: self.set_mode(self.cfg.mode.CORRECTION))
        self.button_pose_mode.clicked.connect(lambda b: self.set_mode(self.cfg.mode.POSE))
        button_first.clicked.connect(self.first_image)
        button_last.clicked.connect(self.last_image)
        button_prev.clicked.connect(self.prev_image)
        button_next.clicked.connect(self.next_image)
        button_left.clicked.connect(lambda x: self.set_view(View.Left))
        button_right.clicked.connect(lambda x: self.set_view(View.Right))
        button_calibrate_calc.clicked.connect(self.calibrate_calc)
        button_pose_save.clicked.connect(self.save_pose)
        #button_optical_flow.clicked.connect(self.carry_annotation)

        layout_h_buttons_top.addWidget(button_pose_estimate, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(button_pose_save, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(button_calibrate_calc, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addWidget(button_rename_images, alignment=Qt.AlignLeft)
        layout_h_buttons_top.addStretch()
        layout_h_buttons_top.addWidget(self.button_heatmap_mode, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_image_mode, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_pose_mode, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.button_correction_mode, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.checkbox_correction_skip, alignment=Qt.AlignRight)
        layout_h_buttons_top.addWidget(self.checkbox_solve_bp, alignment=Qt.AlignRight)

        layout_h_buttons.addWidget(button_first)
        layout_h_buttons.addWidget(button_prev)
        layout_h_buttons.addWidget(button_next)
        layout_h_buttons.addWidget(button_last)
        layout_h_buttons.addWidget(self.textbox_img_id)
        layout_h_buttons.addWidget(button_textbox_img_id_go)
        layout_h_buttons.addStretch()
        #layout_h_buttons.addWidget(button_optical_flow, alignment=Qt.AlignRight)
        #layout_h_buttons.addWidget(self.combo_camera_id, alignment=Qt.AlignRight)
        #layout_h_buttons.addWidget(self.textbox_final_img_id, alignment=Qt.AlignRight)

        layout_h_buttons_second.addWidget(button_left, alignment=Qt.AlignLeft)
        layout_h_buttons_second.addWidget(button_right, alignment=Qt.AlignLeft)
        layout_h_buttons_second.addWidget(self.combo_joint_id, alignment=Qt.AlignRight)

        self.model = QFileSystemModel()
        self.model.setRootPath(self.folder)
        self.tree = QTreeView()
        self.tree.setModel(self.model)

        layout_v.addLayout(layout_h_buttons_top)
        layout_v.addLayout(layout_h_images)
        layout_v.addLayout(layout_h_buttons)
        layout_v.addLayout(layout_h_buttons_second)

        layout_h_tree = QVBoxLayout()
        layout_h_tree.addWidget(self.tree)

        self.setLayout(layout_v)
        self.setWindowTitle(self.folder)
        self.set_pose(self.cfg.img_id)
        self.set_mode(self.cfg.mode)

    def rename_images(self):
        text, okPressed = QInputDialog.getText(self, "Rename Images", "Camera order:", QLineEdit.Normal, "")
        if okPressed:
            text = text.replace(' ', ',')
            text = '[' + text + ']'
            cidread2cid = ast.literal_eval(text)
            if len(cidread2cid) != skeleton.num_cameras:
                print("Cannot rename images as there are no {} values".format(skeleton.num_cameras))
                return

            print('Camera order {}'.format(cidread2cid))

            write_camera_order(self.folder, cidread2cid)
            self.cidread2cid, self.cid2cidread = read_camera_order(self.folder)

            self.camNetAll.set_cid2cidread(self.cid2cidread)
            self.update_()

    def checkbox_automatic_changed(self, state):
        if state == Qt.Checked:
            self.cfg.solve_bp = True
        else:
            self.cfg.solve_bp = False
        print("Solve bp variable set: {}".format(self.cfg.solve_bp))

        self.solve_bp()

    def checkbox_correction_clicked(self, state):
        if state == Qt.Checked:
            self.cfg.correction_skip = True
        else:
            self.cfg.correction_skip = False

    def button_set_width(self, btn, text=' ', margin=20):
        width = btn.fontMetrics().boundingRect(text).width() + 7 + margin
        btn.setMaximumWidth(width)

    def carry_annotation(self):
        raise NotImplementedError
        cam_id = int(self.combo_camera_id.currentText())
        limit = int(self.textbox_final_img_id.text())
        self.image_pose_list[cam_id].carry_annotation(limit=limit)

    def pose2d_estimation(self):
        import os
        from drosoph3D.pose2d import ArgParse
        from drosoph3D.pose2d.utils.osutils import mkdir_p, isdir
        from drosoph3D.pose2d.utils.misc import get_time
        parser = ArgParse.create_parser()
        args, _ = parser.parse_known_args()
        args.checkpoint = os.path.join(args.checkpoint,
                                       get_time() + "_{}_{}_{}_{}_{}_{}_{}".format(
                                           "predict" if args.unlabeled else "training",
                                           args.arch, args.stacks, args.img_res,
                                           args.blocks,
                                           "mv" if args.multiview else "",
                                           "temp" if args.multiview else "",
                                           args.name))
        args.checkpoint = args.checkpoint.replace(' ', '_').replace('(', "_").replace(')',
                                                                                      '_')
        args.checkpoint = args.checkpoint.replace('__', '_').replace('--',
                                                                     '-')
        print("Checkpoint dir: {}".format(args.checkpoint))
        args.checkpoint_image_dir = os.path.join(args.checkpoint, "./images/")

        # create checkpoint dir and image dir
        if not isdir(args.checkpoint):
            mkdir_p(args.checkpoint)
        if not isdir(args.checkpoint_image_dir):
            mkdir_p(args.checkpoint_image_dir)
        if args.carry and not isdir(os.path.join(args.annotation_path, args.unlabeled + "_network")):
            mkdir_p(os.path.join(args.annotation_path, args.unlabeled + "_network"))
        if args.carry and not isdir(os.path.join(args.data_folder, args.unlabeled + "_network")):
            mkdir_p(os.path.join(args.data_folder, args.unlabeled + "_network"))

        args.unlabeled = self.folder

        curr_path = os.path.abspath(os.path.dirname(__file__))
        args.resume = os.path.join(curr_path, '../../weights/sh8_front_j8.tar')
        # args.resume = '../../weights/sh8_manual.tar'

        # run the main, get back the heatmap
        from drosoph3D.pose2d.drosophila import main
        args.max_img_id = self.cfg.num_images - 1
        val_hm, val_pred = main(args)

        calib = read_calib(self.folder)
        self.camNetAll = CameraNetwork(image_folder=self.folder, cam_id_list=[0, 1, 2, 3, 4, 5, 6],
                                       num_images=self.cfg.num_images,
                                       calibration=calib, num_joints=self.cfg.num_joints,
                                       heatmap_shape=self.cfg.heatmap_shape, cid2cidread=self.cid2cidread)
        self.camNetLeft = CameraNetwork(image_folder=self.folder, cam_id_list=[0, 1, 2, 3],
                                        num_images=self.cfg.num_images,
                                        calibration=calib, num_joints=self.cfg.num_joints,
                                        heatmap_shape=self.cfg.heatmap_shape, cam_list=self.camNetAll.cam_list[:4],
                                        cid2cidread=self.cid2cidread[:4])
        self.camNetRight = CameraNetwork(image_folder=self.folder, cam_id_list=[6, 5, 4, 3],
                                         num_images=self.cfg.num_images,
                                         calibration=calib, num_joints=self.cfg.num_joints,
                                         heatmap_shape=self.cfg.heatmap_shape,
                                         cam_list=self.camNetAll.cam_list[-4::][::-1],
                                         cid2cidread=self.cid2cidread[-4::][::-1])
        self.camNetMid = CameraNetwork(image_folder=self.cfg.folder, cam_id_list=[3], num_images=self.cfg.num_images,
                                       calibration=calib, num_joints=self.cfg.num_joints,
                                       heatmap_shape=self.cfg.heatmap_shape, cam_list=[self.camNetAll[3]],
                                       cid2cidread=self.cid2cidread[3])

        self.camNet = self.camNetLeft
        self.set_mode(Mode.POSE)
        self.set_view(self.cfg.view)

        self.update_()

    def set_mode(self, mode):
        if (mode == Mode.POSE and self.camNet.has_pose()) or (
                mode == Mode.HEATMAP and self.camNet.has_heatmap()) or mode == Mode.IMAGE or (
                mode == Mode.CORRECTION and self.camNet.has_pose()):
            self.cfg.mode = mode
        else:
            print("Cannot set mode: {}".format(mode))
        if self.cfg.mode == Mode.CORRECTION:
            self.set_pose(self.cfg.img_id)
        self.update_()

        for b in self.button_list_modes:
            b.setChecked(False)
        if self.cfg.mode == Mode.HEATMAP:
            self.button_heatmap_mode.setChecked(True)
        elif self.cfg.mode == Mode.POSE:
            self.button_pose_mode.setChecked(True)
        elif self.cfg.mode == Mode.IMAGE:
            self.button_image_mode.setChecked(True)
        elif self.cfg.mode == Mode.CORRECTION:
            self.button_correction_mode.setChecked(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.set_img_id_tb()
            self.setFocus()
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_H:
            self.set_mode(Mode.HEATMAP)
        if event.key() == Qt.Key_I:
            self.set_mode(Mode.IMAGE)
        if event.key() == Qt.Key_X:
            self.set_mode(Mode.POSE)
        if event.key() == Qt.Key_C:
            self.set_mode(Mode.CORRECTION)
            self.update_()
        if event.key() == Qt.Key_T:
            for image_pose in self.image_pose_list:
                image_pose.save_correction()
            self.update_()
        if event.key() == Qt.Key_L:
            self.set_view(View.Left)
        if event.key == Qt.Key_R:
            self.set_view(View.Right)

    def already_corrected(self, view, img_id):
        if view == View.Left:
            return self.cfg.db.has_key(0, img_id) or self.cfg.db.has_key(1, img_id) or self.cfg.db.has_key(2, img_id)
        elif view == View.Right:
            return self.cfg.db.has_key(4, img_id) or self.cfg.db.has_key(5, img_id) or self.cfg.db.has_key(6, img_id)
        else:
            raise NotImplementedError

    def combo_activated(self, text):
        if text == 'All':
            self.set_heatmap_joint_id(-1)
        else:
            self.set_heatmap_joint_id(int(text.replace("Heatmap: ", "")))
        self.setFocus()

    def first_image(self):
        img_id = 0
        self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)
        self.set_pose(img_id)

    def last_image(self):
        img_id = self.cfg.num_images - 1
        self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)
        self.set_pose(img_id)

    def prev_image(self):
        if self.cfg.mode != Mode.CORRECTION or (
                not self.cfg.correction_skip or not self.camNet.has_calibration() or not self.camNet.has_pose()):
            img_id = max(self.cfg.img_id - 1, 0)
            self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)
        else:
            print("Not checking errors for prev_image")
            img_id = self.prev_error(self.cfg.img_id)
            self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)

        self.set_pose(img_id)

    def next_image(self):
        if self.cfg.mode != Mode.CORRECTION or (
                not self.cfg.correction_skip or not self.camNet.has_calibration() or not self.camNet.has_pose()):
            img_id = min(self.cfg.num_images - 1, self.cfg.img_id + 1)
            self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)
        else:
            print("Not checking errors for next_image")
            img_id = self.next_error(self.cfg.img_id)
            self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)

        self.set_pose(img_id)

    def get_joint_reprojection_error(self, img_id, joint_id, pts=None):
        visible_cameras = [cam for cam in self.camNet if skeleton.camera_see_joint(cam.cam_id, joint_id)]
        if len(visible_cameras) >= 2:
            if pts is None:
                pts = np.array([cam.points2d[img_id, joint_id, :] for cam in visible_cameras])
            p3d, err_proj, prob_heatm, prob_bone = energy_drosoph(visible_cameras, img_id, joint_id,
                                                                  pts / [960, 480])
        else:
            err_proj = 0

        return err_proj

    def next_error(self, img_id):
        for img_id in range(img_id + 1, self.cfg.num_images):
            for joint_id in range(skeleton.num_joints):
                if joint_id not in skeleton.pictorial_joint_list:
                    continue
                points2d = np.array([cam.points2d[img_id, joint_id, :] for cam in self.camNet])
                if np.all(points2d == 0):
                    continue
                err_proj = self.get_joint_reprojection_error(img_id, joint_id)
                # if err_proj > self.cfg.reproj_thr[joint_id % (self.cfg.num_joints // 2)]:
                if err_proj > 15:
                    print("{} {} {}".format(img_id, joint_id, err_proj))
                    return img_id

        return self.cfg.num_images - 1

    def prev_error(self, curr_img_id):
        for img_id in range(curr_img_id - 1, 0, -1):
            for joint_id in range(skeleton.num_joints):
                if joint_id not in skeleton.pictorial_joint_list:
                    continue
                points2d = np.array([cam.points2d[img_id, joint_id, :] for cam in self.camNet])
                if np.all(points2d == 0):
                    continue
                err_proj = self.get_joint_reprojection_error(img_id, joint_id)
                # if err_proj > self.cfg.reproj_thr[joint_id % (self.cfg.num_joints // 2)]:
                if err_proj > 15:
                    print("{} {} {}".format(img_id, joint_id, err_proj))
                    return img_id

        return 0

    def solve_bp(self, save_correction=False):
        if not (
                self.cfg.mode == Mode.CORRECTION and self.cfg.solve_bp and self.camNet.has_calibration() and self.camNet.has_pose()):
            print("Cannot solve BP, not correction mode solve_bp is not set")
            return

        print("Solving BP")
        prior = list()
        for ip in self.image_pose_list:
            if ip.dynamic_pose is not None:
                for (joint_id, pt2d) in ip.dynamic_pose.manual_correction_dict.items():
                    prior.append((ip.cam.cam_id, joint_id, pt2d / self.cfg.image_shape))
        print("Prior for BP: {}".format(prior))
        pts_bp = self.cfg.camNet.solveBP(self.cfg.img_id, self.cfg.bone_param, prior=prior)
        pts_bp = np.array(pts_bp)

        # set points which are not estimated by bp
        for idx, image_pose in enumerate(self.image_pose_list):
            pts_bp_ip = pts_bp[idx] * image_pose.cfg.image_shape
            pts_bp_rep = self.cfg.db.read(image_pose.cam.cam_id, self.cfg.img_id)
            if pts_bp_rep is None:
                pts_bp_rep = image_pose.cam.points2d[self.cfg.img_id, :]
            else:
                pts_bp_rep *= image_pose.cfg.image_shape
            pts_bp_ip[pts_bp_ip == 0] = pts_bp_rep[pts_bp_ip == 0]
            # pts_bp_ip *= image_pose.cfg.image_shape

            # keep track of the manually corrected points
            mcd = image_pose.dynamic_pose.manual_correction_dict if image_pose.dynamic_pose is not None else None
            image_pose.dynamic_pose = DynamicPose(pts_bp_ip, image_pose.cfg.img_id, joint_id=None,
                                                  manual_correction=mcd)
        self.update_()

        # save down corrections as training if any priors were given
        if prior and save_correction:
            print('Saving with prior')
            for ip in self.image_pose_list:
                ip.save_correction()
        print("Finished bp")

    def set_pose(self, img_id):
        self.cfg.img_id = img_id

        for ip in self.image_pose_list:
            ip.clear_mc()
        if self.cfg.mode == Mode.CORRECTION:
            for ip in self.image_pose_list:
                pt = self.cfg.db.read(ip.cam.cam_id, self.cfg.img_id)
                modified_joints = self.cfg.db.read_modified_joints(ip.cam.cam_id, self.cfg.img_id)
                if pt is None:
                    pt = ip.cam.points2d[self.cfg.img_id, :]
                else:
                    pt *= self.cfg.image_shape

                manual_correction = dict()
                for joint_id in modified_joints:
                    manual_correction[joint_id] = pt[joint_id]
                ip.dynamic_pose = DynamicPose(pt, ip.cfg.img_id, joint_id=None, manual_correction=manual_correction)

            if self.camNet.has_calibration():
                self.solve_bp()

        self.update_()
        self.textbox_img_id.setText(str(self.cfg.img_id))

    def set_heatmap_joint_id(self, joint_id):
        self.cfg.hm_joint_id = joint_id
        self.update_()

    def set_img_id_tb(self):
        try:
            img_id = int(self.textbox_img_id.text().replace("Heatmap: ", ""))
            self.cfg.already_corrected = self.already_corrected(self.cfg.view, img_id)
            self.set_pose(img_id)
        except BaseException as e:
            print("Textbox img id is not integer {}".format(str(e)))

    def set_joint_id_tb(self):
        self.set_heatmap_joint_id(int(self.textbox_joint_id.text()))

    def set_view(self, v):
        self.cfg.view = v
        if self.cfg.view == View.Left:
            self.camNet = self.camNetLeft
        elif self.cfg.view == View.Right:
            self.camNet = self.camNetRight
        self.cfg.camNet = self.camNet

        for image_pose, cam in zip(self.image_pose_list, self.camNet):
            image_pose.cam = cam
        for image_pose in self.image_pose_list:
            image_pose.clear_mc()

        self.set_pose(self.cfg.img_id)

        self.update_()

    def calibrate_calc(self):
        text, okPressed = QInputDialog.getText(self, "Calibration", "Range of images:", QLineEdit.Normal,
                                               "0 {}".format(self.cfg.num_images - 1))

        if okPressed:
            text = text.replace(' ', ',')
            text = '[' + text + ']'

            try:
                [min_img_id, max_img_id] = ast.literal_eval(text)
            except:
                min_img_id, max_img_id = 0, self.cfg.max_num_images
            print("Calibration considering frames between {}:{}".format(min_img_id, max_img_id))

            self.camNetLeft[0].set_alpha(0 / 57.2, r=94)
            self.camNetLeft[1].set_alpha(-30 / 57.2, r=94)
            self.camNetLeft[2].set_alpha(-70 / 57.2, r=94)
            self.camNetMid[0].set_alpha(-125 / 57.2, r=94)
            self.camNetRight[0].set_alpha(+110 / 57.2, r=94)
            self.camNetRight[1].set_alpha(+150 / 57.2, r=94)
            self.camNetRight[2].set_alpha(+179 / 57.2, r=94)

            print([c.tvec for c in self.camNetAll])

            # take a copy of the current points2d
            pts2d = np.zeros((7, self.cfg.num_images, self.cfg.num_joints, 2), dtype=float)
            for cam_id in range(self.cfg.num_cameras):
                pts2d[cam_id, :] = self.camNetAll[cam_id].points2d.copy()

            # ugly hack to temporarly incorporate manual corrections
            c = 0
            for cam_id in range(self.cfg.num_cameras):
                for img_id in range(self.cfg.num_images):
                    if self.cfg.db.has_key(cam_id, img_id):
                        pt = self.cfg.db.read(cam_id, img_id) * self.cfg.image_shape
                        self.camNetAll[cam_id].points2d[img_id, :] = pt
                        c += 1
            print("Calibration: replaced {} points from manuall correction".format(c))

            # keep the pts only in the range
            for cam in self.camNetAll:
                cam.points2d = cam.points2d[min_img_id:max_img_id, :]

            self.camNetLeft.triangulate()
            self.camNetLeft.bundle_adjust(unique=False, prior=True)
            self.camNetRight.triangulate()
            self.camNetRight.bundle_adjust(unique=False, prior=True)
            self.camNetAll.triangulate()
            self.camNetAll.bundle_adjust(unique=False, prior=True)

            # put old values back
            for cam_id in range(self.cfg.num_cameras):
                self.camNetAll[cam_id].points2d = pts2d[cam_id, :].copy()

            self.calibrate_save()

    def calibrate_save(self):
        calib_path = "{}/calib_{}.pkl".format(self.folder, self.folder.replace('/', '_'))
        print("Saving calibration {}".format(calib_path))
        self.camNetLeft.save_network(calib_path)
        print("Saving calibration {}".format(calib_path))
        self.camNetRight.save_network(calib_path)

    def save_pose(self):
        pts2d = np.zeros((7, self.cfg.num_images, self.cfg.num_joints, 2), dtype=float)
        #pts3d = np.zeros((self.cfg.num_images, self.cfg.num_joints, 3), dtype=float)

        for cam in self.camNetAll:
            pts2d[cam.cam_id, :] = cam.points2d.copy()

        # overwrite by manual correction
        count = 0
        for cam_id in range(self.cfg.num_cameras):
            for img_id in range(self.cfg.num_images):
                if self.cfg.db.has_key(cam_id, img_id):
                    pt = self.cfg.db.read(cam_id, img_id) * self.cfg.image_shape
                    pts2d[cam_id, img_id, :] = pt
                    count += 1


        # some post-processing for body-coxa
        for cam_id in range(len(self.camNetAll.cam_list)):
            for j in range(skeleton.num_joints):
                if skeleton.camera_see_joint(cam_id, j) and skeleton.is_body_coxa(j):
                    pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                    pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])

        dict_merge = self.camNetAll.save_network(path=None)
        dict_merge["points2d"] = pts2d

        # take a copy of the current points2d
        pts2d_orig = np.zeros((7, self.cfg.num_images, self.cfg.num_joints, 2), dtype=float)
        for cam_id in range(self.cfg.num_cameras):
            pts2d_orig[cam_id, :] = self.camNetAll[cam_id].points2d.copy()

        # ugly hack to temporarly incorporate manual corrections
        c = 0
        for cam_id in range(self.cfg.num_cameras):
            for img_id in range(self.cfg.num_images):
                if self.cfg.db.has_key(cam_id, img_id):
                    pt = self.cfg.db.read(cam_id, img_id) * self.cfg.image_shape
                    self.camNetAll[cam_id].points2d[img_id, :] = pt
                    c += 1
        print("Replaced points2d with {} manual correction".format(count))

        # do the triangulationm if we have the calibration
        if self.camNet.has_calibration() and self.camNet.has_pose():
            #self.camNetLeft.triangulate()
            #pts3d[:, :self.cfg.num_joints // 2, :] = self.camNetLeft.points3d_m[:, :self.cfg.num_joints // 2, :]
            #self.camNetRight.triangulate()
            #pts3d[:, self.cfg.num_joints // 2:, :] = self.camNetRight.points3d_m[:, self.cfg.num_joints // 2:, :]

            self.camNetAll.triangulate()
            pts3d = self.camNetAll.points3d_m

            dict_merge["points3d"] = pts3d

        # put old values back
        for cam_id in range(self.cfg.num_cameras):
            self.camNetAll[cam_id].points2d = pts2d_orig[cam_id, :].copy()

        pickle.dump(dict_merge,
                    open(os.path.join(self.folder, "pose_result_{}.pkl".format(self.folder.replace('/', '_'))), 'wb'))
        print("Saved the pose at: {}".format(
            os.path.join(self.folder, "pose_result_{}.pkl".format(self.folder.replace('/', '_')))))

    def update_(self):
        for image_pose in self.image_pose_list:
            image_pose.update_()


class DynamicPose:
    def __init__(self, points2d, img_id, joint_id, manual_correction=None):
        self.points2d = points2d
        self.img_id = img_id
        self.joint_id = joint_id
        self.manual_correction_dict = manual_correction
        if manual_correction is None:
            self.manual_correction_dict = dict()

    def set_joint(self, joint_id, pt2d):
        assert (pt2d.shape[0] == 2)
        self.points2d[joint_id] = pt2d
        self.manual_correction_dict[joint_id] = pt2d


class ImagePose(QWidget):
    def __init__(self, config, cam, f_solve_bp):
        QWidget.__init__(self)
        self.cfg = config
        self.cam = cam

        self.dynamic_pose = None

        self.update_()
        self.show()

        self.f_solve_bp = f_solve_bp

    def clear_mc(self):
        self.dynamic_pose = None

    def update_(self):
        draw_joints = [j for j in range(skeleton.num_joints) if skeleton.camera_see_joint(self.cam.cam_id, j)]
        corrected_this_camera = self.cfg.db.has_key(self.cam.cam_id, self.cfg.img_id)
        if self.cfg.mode == Mode.IMAGE:
            im = self.cam.get_image(self.cfg.img_id)
        elif self.cfg.mode == Mode.HEATMAP:
            draw_joints = [self.cfg.hm_joint_id] if self.cfg.hm_joint_id != -1 else [j for j in
                                                                                     range(skeleton.num_joints) if
                                                                                     skeleton.camera_see_joint(
                                                                                         self.cam.cam_id, j)]
            print(self.cam.cam_id, draw_joints)
            im = self.cam.plot_heatmap(img_id=self.cfg.img_id, concat=False, scale=2, draw_joints=draw_joints)
        elif self.cfg.mode == Mode.POSE:
            zorder = skeleton.get_zorder(self.cam.cam_id)
            im = self.cam.plot_2d(img_id=self.cfg.img_id, draw_joints=draw_joints, zorder=zorder)
        elif self.cfg.mode == Mode.CORRECTION:
            circle_color = (0, 255, 0) if corrected_this_camera else (0, 0, 255)
            im = self.cam.plot_2d(img_id=self.cfg.img_id, pts=self.dynamic_pose.points2d,
                                  circle_color=circle_color, draw_joints=draw_joints)
        im = im.astype(np.uint8)
        height, width, channel = im.shape
        bytesPerLine = 3 * width
        qIm = QImage(im.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.im = QPixmap.fromImage(qIm)

        self.update()

    def carry_annotation(self, limit):
        for img_id in range(self.cfg.img_id, limit - 1):
            print("Cam id {}, image id {}".format(self.cam.cam_id, img_id))
            pts2d = self.cfg.db.read(self.cam.cam_id, img_id) * self.cfg.image_shape
            pts2d_propagate = optical_flow(self.cam.get_image(img_id, flip=self.cam.cam_id > 3),
                                           self.cam.get_image(img_id + 1, flip=self.cam.cam_id > 3), pts2d)
            points2d = self.cfg.db.read(self.cam.cam_id, img_id + 1)
            if points2d is not None:
                points2d *= self.cfg.image_shape
            else:
                points2d = self.cam.get_points2d(img_id=img_id + 1).copy()

            # set only which are corrected by hand
            joint_id_list = list(self.dynamic_pose.manual_correction_dict.keys())
            points2d[joint_id_list] = pts2d_propagate[joint_id_list].copy()

            self.cfg.db.write(points2d / self.cfg.image_shape, self.cam.cam_id, img_id + 1, train=False,
                              modified_joints=joint_id_list)

    def save_correction(self, thr=30):
        points2d_prediction = self.cam.get_points2d(self.cfg.img_id)
        points2d_correction = self.dynamic_pose.points2d

        err = np.abs(points2d_correction - points2d_prediction)
        check_joint_id_list = [j for j in range(skeleton.num_joints) if
                               (j not in skeleton.ignore_joint_id) and skeleton.camera_see_joint(self.cam.cam_id, j)]

        for j in check_joint_id_list:
            if np.any(err[j] > thr):
                err_max = np.max(err[check_joint_id_list])
                joint_id, ax = np.where(err == err_max)

                print("Saving camera {} with l1 {} on joint {}".format(self.cam.cam_id, err_max, joint_id))
                # make sure we are not saving a points that cannot be seen from the camera
                unseen_joints = [j for j in range(skeleton.num_joints) if
                                 not skeleton.camera_see_joint(self.cam.cam_id, j)]
                points2d_correction[unseen_joints, :] = 0.
                self.cfg.db.write(points2d_correction / self.cfg.image_shape, self.cam.cam_id, self.cfg.img_id,
                                  train=True,
                                  modified_joints=list(self.dynamic_pose.manual_correction_dict.keys()))

                return True

        return False

    def mouseMoveEvent(self, e):
        if self.cfg.mode == Mode.CORRECTION:
            x = int(e.x() * (self.cfg.image_shape[0] * 1.0) / self.frameGeometry().width())
            y = int(e.y() * (self.cfg.image_shape[1] * 1.0) / self.frameGeometry().height())
            if self.dynamic_pose.joint_id is None:  # find the joint to be dragged with nearest neighbor
                pts = self.dynamic_pose.points2d.copy()

                # make sure we don't select points we cannot see
                pts[[j_id for j_id in range(skeleton.num_joints) if not
                skeleton.camera_see_joint(self.cam.cam_id, j_id)]] = [9999, 9999]

                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)

                _, indices = nbrs.kneighbors(np.array([[x, y]]))
                self.dynamic_pose.joint_id = indices[0][0]
                print("Selecting the joint: {}".format(self.dynamic_pose.joint_id))
            self.dynamic_pose.set_joint(self.dynamic_pose.joint_id, np.array([x, y]))
            self.update_()

    def mouseReleaseEvent(self, e):
        if self.cfg.mode == Mode.CORRECTION:
            self.dynamic_pose.joint_id = None  # make sure we forget the tracked joint
            self.update_()

            # solve BP again
            self.cfg.db.write(self.dynamic_pose.points2d / self.cfg.image_shape, self.cam.cam_id, self.cfg.img_id,
                              train=True, modified_joints=list(self.dynamic_pose.manual_correction_dict.keys()))
            self.f_solve_bp(save_correction=True)
            self.update_()

    def get_image_name(self, img_id):
        img_path = self.get_image_path(self.cfg.folder, img_id)
        img_name = os.path.basename(img_path).replace('.jpg', '')
        return img_name

    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.im)

        self.update()


class PrintImage(QWidget):
    def __init__(self, pixmap, parent=None):
        QWidget.__init__(self, parent=parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(QRect(0, 0, self.pixmap.width(), self.pixmap.height()), self.pixmap)


def main():
    app = QApplication([])
    window = DrosophAnnot()
    # set the application dimensions w.r.t the scrren size
    screen = app.primaryScreen()
    size = screen.size()
    height, width = size.height(), size.width()
    hw_ratio = (960 * 2 / 360.)
    app_width = width
    app_height = int(app_width / hw_ratio)
    window.resize(app_width, app_height)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
