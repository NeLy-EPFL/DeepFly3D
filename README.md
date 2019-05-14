# DeepFly3D

![Alt text](images/gui.gif?raw=true "Title")

Drosophila-GUI is a PyTorch and PyQT5 implementation of the general processing of 2D-3D single tethered drosophila pose estimation. The aim is to provide the interface of estimation and further correction GUI for the 2D pose estimation, which is automatically converted to 3D pose.

Codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train).  We also use Stacked Hourglass model for 2D Pose estimation. We implement GUI and other niceties from strath using PyQT5.

* Installing DeepFly3D: [Installation](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md)
* Interacting with the GUI:  [GUI](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/gui.md)
* In case you want to do everything on the terminal: [w/ Terminal](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/terminal.md)

## Niceties
### Finding Errenous Estimations Automatically

### Auto-Correction

### Helping Correction
Auto-Correction is perfomed in case 2d pose estimation and calibration has completed. In the Correction mode, GUI tries to correct for errors using pictorial structures. In case you confirm the press ```T``` to make sure they are saved. Please check the paper for details.
![Alt text](images/correction.gif?raw=true "Title")

### Visualization
To visualize the pose_result file which is saved in the same folder as the images, you can use the notebook at ```notebook_visualize/visualize.ipynb```, by replacing the image folder in the first cell. Visualization should output figures close to the figures in the paper.

<img src="images/pose3D.png" width="960">

To visualize the time series instead, use the notebook ```notebook_visualize/time_series.ipynb```. It should output 2D/3D pose, together with the few selected time series.

<img src="images/time_series.png" width="960">
