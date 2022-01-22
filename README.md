# DeepFly3D
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![Alt text](images/pose3D.gif?raw=true "Title")

DeepFly3D is a PyTorch and PyQT5 implementation of 2D-3D tethered Drosophila pose estimation. It aims to provide an interface for pose estimation and to permit further correction of 2D pose estimates, which are automatically converted to 3D pose. 

DeepFly3D **does not require a calibration pattern**, it enforces **geometric constraints using pictorial structures**, which corrects most of the errors, and the **remaining errors are automatically detected can be dealt manually with GUI**.

* Installing DeepFly3D: [Installation](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md)
* Interacting with the GUI:  [GUI](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/gui.md)
* Generating visualizations: [Visualization](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/visual.md)
* If you are interested in the online annotation tool instead: [DeepFly3DAnnotation](https://github.com/NeLy-EPFL/DeepFly3DAnnotation)
* To see the dataset used in the paper: [Dataverse](https://dataverse.harvard.edu/dataverse/DeepFly3D)

## Changes
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

* Raw data
* Probability maps
* Raw predictions
* Automatic corrections

And you can perform:

* 2D pose estimation
* Calibration without calibration pattern
* Saving the final 3D estimations
* Manual Correction

### Identifying erroneous estimates automatically
![Alt text](images/gui_finderror.gif?raw=true "Title")
DeepFly3D can automatically detect when 2D pose estimation is failed. 

### Assisting manual correction
 In the 'Correction' mode, the GUI tries to correct errors using pictorial structures. To save these corrections, press ```T```. Please check the associated manuscript (Günel et al. 2019) for implementation details.
![Alt text](images/correction.gif?raw=true "Title")
In the next iteration of training, you can use these examples to train the network!


### References
```
@inproceedings{Gunel19DeepFly3D,
  author    = {Semih Gunel and
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
```
