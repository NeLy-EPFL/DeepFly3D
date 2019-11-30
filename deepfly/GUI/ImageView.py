import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter
from .State import Mode


class ImageView(QWidget):
    def __init__(self, core, camera_id):
        QWidget.__init__(self)
        self.core = core
        self.cam_id = camera_id 
        self.corrections_enabled = False
        self.joint_being_corrected = None
        self.displayed_img = None


    def show(self, img_id, mode):
        if mode == Mode.IMAGE:
            im = self.core.get_image(self.cam_id, img_id)
        #
        elif mode == Mode.POSE:
            im = self.core.plot_2d(self.cam_id, img_id)
        #
        elif mode == Mode.HEATMAP:
            im = self.core.plot_heatmap(self.cam_id, img_id)
        #
        elif mode == Mode.CORRECTION:
            im = self.core.plot_2d(self.cam_id, img_id, with_corrections=True)
        #
        else:
            raise RuntimeError(f'Unknown mode {mode}')
        #
        self.displayed_img = img_id
        self.corrections_enabled = (mode == Mode.CORRECTION)
        self.update_pixmap(im)


    def mouseMoveEvent(self, e):
        if self.corrections_enabled:
            x = int(e.x() * np.array(self.core.image_shape[0]) / self.frameGeometry().width())
            y = int(e.y() * np.array(self.core.image_shape[1]) / self.frameGeometry().height())
            self.move_joint(x, y)
            
            
    def move_joint(self, x, y):
        if self.joint_being_corrected is None:
            self.joint_being_corrected = self.core.nearest_joint(self.cam_id, self.displayed_img, x, y)
        self.core.move_joint(self.cam_id, self.displayed_img, self.joint_being_corrected, x, y)
        self.show(self.displayed_img, Mode.CORRECTION)
            

    def mouseReleaseEvent(self, _):
        self.joint_being_corrected = None  # forget the tracked joint


    def update_pixmap(self, image_array):
        im = image_array.astype(np.uint8)
        height, width, _ = im.shape
        bytesPerLine = 3 * width
        qIm = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qIm)
        self.update()


    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        self.update()