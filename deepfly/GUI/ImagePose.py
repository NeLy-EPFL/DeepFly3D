import numpy as np

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter
from .Config import config
from .State import Mode

class DynamicPose:
    def __init__(self, points2d, img_id, joint_id, manual_correction=None):
        self.points2d = points2d
        self.img_id = img_id
        self.joint_id = joint_id
        self.manual_correction_dict = manual_correction
        if manual_correction is None:
            self.manual_correction_dict = dict()

    def move_joint(self, joint_id, pt2d):
        assert pt2d.shape[0] == 2
        self.points2d[joint_id] = pt2d
        self.manual_correction_dict[joint_id] = pt2d


class ImagePose(QWidget):
    def __init__(self, config, core, cam, f_solve_bp):
        QWidget.__init__(self)
        self.state = config
        self.core = core
        self.cam = cam
        self._dynamic_pose = None

        self.update_image_pose()
        self.show()

        self.f_solve_bp = f_solve_bp


    def manual_corrections(self):
        return self._dynamic_pose.manual_correction_dict if self._dynamic_pose else {}


    def update_manual_corrections(self, points2d, img_id, joint_id, manual_correction=None):
        self._dynamic_pose = DynamicPose(points2d, img_id, joint_id, manual_correction)


    def clear_manual_corrections(self):
        self._dynamic_pose = None


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
            #
        elif self.state.mode == Mode.HEATMAP:
            draw_joints = (
                [self.state.heatmap_joint_id]
                if self.state.heatmap_joint_id != -1
                else [
                    j
                    for j in range(config["skeleton"].num_joints)
                    if config["skeleton"].camera_see_joint(self.cam.cam_id, j)
                ]
            )
            im = self.cam.plot_heatmap(
                img_id=self.state.img_id, concat=False, scale=2, draw_joints=draw_joints
            )
            #
        elif self.state.mode == Mode.POSE:
            im = self.cam.plot_2d(
                img_id=self.state.img_id, draw_joints=draw_joints, zorder=zorder
            )
            #
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
                pts=self._dynamic_pose.points2d,
                circle_color=circle_color,
                draw_joints=draw_joints,
                zorder=zorder,
                r_list=r_list,
            )
            #
        im = im.astype(np.uint8)
        height, width, channel = im.shape

        bytesPerLine = 3 * width
        qIm = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
        self.im = QPixmap.fromImage(qIm)

        self.update()


    def save_correction(self, thr=30):
        points2d_prediction = self.cam.get_points2d(self.state.img_id)
        points2d_correction = self._dynamic_pose.points2d

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
                        self._dynamic_pose.manual_correction_dict.keys()
                    ),
                )

                return True

        return False


    def find_nearest_joint(self, x, y):
        joints = range(config["skeleton"].num_joints)
        visible = lambda j_id: config["skeleton"].camera_see_joint(self.cam.cam_id, j_id)
        unvisible_joints = [j_id for j_id in joints if not visible(j_id)]
        
        pts = self._dynamic_pose.points2d.copy()
        pts[unvisible_joints] = [9999, 9999]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(pts)
        _, indices = nbrs.kneighbors(np.array([[x, y]]))
        return indices[0][0]
        

    def mouseMoveEvent(self, e):
        print('MouseMove', e.pos())
        if self.state.mode == Mode.CORRECTION:
            x = int(e.x() * np.array(config["image_shape"][0]) / self.frameGeometry().width())
            y = int(e.y() * np.array(config["image_shape"][1]) / self.frameGeometry().height())
            
            if self._dynamic_pose.joint_id is None:
                self._dynamic_pose.joint_id = self.find_nearest_joint(x, y)        
                print("Selecting the joint: {}".format(self._dynamic_pose.joint_id))

            self._dynamic_pose.move_joint(self._dynamic_pose.joint_id, np.array([x, y]))
            self.update_image_pose()


    def mouseReleaseEvent(self, _):
        if self.state.mode == Mode.CORRECTION:
            self._dynamic_pose.joint_id = None  # make sure we forget the tracked joint
            self.update_image_pose()

            # solve BP again
            self.state.db.write(
                self._dynamic_pose.points2d / config["image_shape"],
                self.cam.cam_id,
                self.state.img_id,
                train=True,
                modified_joints=list(self._dynamic_pose.manual_correction_dict.keys()),
            )
            self.f_solve_bp(save_correction=True)
            self.update_image_pose()


    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.im)
        self.update()
