# DeepFly3D
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/df3d.svg)](https://badge.fury.io/py/df3d)

![Alt text](images/pose3D.gif?raw=true "Title")

DeepFly3D is a PyTorch and PyQT5 implementation of 2D-3D tethered Drosophila pose estimation. It aims to provide an interface for pose estimation and to permit further correction of 2D pose estimates, which are automatically converted to 3D pose. 

DeepFly3D **does not require a calibration pattern**, it enforces **geometric constraints using pictorial structures**, which corrects most of the errors, and the **remaining errors are automatically detected can be dealt manually with GUI**.

* Installing DeepFly3D: [Installation](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md)
* How to Use: [Usage](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md)
* If you are interested in the online annotation tool instead: [DeepFly3DAnnotation](https://github.com/NeLy-EPFL/DeepFly3DAnnotation)
* To see the dataset used in the paper: [Dataverse](https://dataverse.harvard.edu/dataverse/DeepFly3D)
* How to Contribute: [Development](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/contribution.md)

## Changes
### Changes in 0.5
- 
### Changes in 0.4

- Using the CLI, the output folder can be changed using the `--output-folder` flag
- CLI and GUI now use the same pose estimation code, so changes will automatically propagate to both
- Minor tweaks in the GUI layout, functionality kept unchanged

### Changes in 0.3
- Results are saved in df3d folder instead of the image folder.
- Much faster startup time. 
- Cameras are automatically ordered using Regular Expressions.
- CLI improvements. Now it includes 3D pose.

### Changes in 0.2
- Changing name from deepfly3d to df3d
- Adding cli interface with df3d-cli
- Removing specific dependencies for numpy and scipy
- Removing L/R buttons, so you can see all the data at once
- Removing the front camera
- Faster startup time, less time spent on searching for the image folder
- Better notebooks for plotting
- Adding procrustes support. Now all the output is registere to template skeleton.
- Bug fixes in CameraNetwork. Now calibration with arbitrary camera sequence is possible.

### Known Problems
- Some insability in automatic correction

## GUI
![Alt text](images/gui2.png?raw=true "Title")
DeepFly3D provides a nice GUI to interact with the data. Using DeepFly3D GUI, you can visualize:

### Identifying erroneous estimates automatically
![Alt text](images/gui_finderror.gif?raw=true "Title")
DeepFly3D can automatically detect when 2D pose estimation is failed. 


### Assisting manual correction
 In the 'Correction' mode, the GUI tries to correct errors using pictorial structures. To save these corrections, press ```T```. Please check the associated manuscript (Günel et al. 2019) for implementation details.
![Alt text](images/correction.gif?raw=true "Title")
In the next iteration of training, you can use these examples to train the network!


To create more complicated figures, or replicate the figures from the paper, you can use the the pose_result file which is saved in the same folder as the images. The notebook, ```notebook_visualize/visualize.ipynb```, shows you the steps to create the following figure:


### References

<img src="images/pose3D.png" width="960">

<p align="center">
<img src="images/time_series.png" width="640">
</p>

```
@inproceedings{Gunel19DeepFly3D,
  author    = {Semih G{\"u}nel and
               Helge Rhodin and
               Daniel Morales and
               João Compagnolo and
               Pavan Ramdya and
               Pascal Fua},
  title     = {DeepFly3D, a deep learning-based approach for 3D limb and appendage tracking in tethered, adult Drosophila},
  bookTitle = {eLife},
  doi       = {10.7554/eLife.48571},
  year      = {2019}
}
``
