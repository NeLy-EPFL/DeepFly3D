# DeepFly3D

![Alt text](images/gui.gif?raw=true "Title")

Drosophila-GUI is a PyTorch and PyQT5 implementation of the general processing of 2D-3D single tethered drosophila pose estimation. The aim is to provide the interface of estimation and further correction GUI for the 2D pose estimation, which is automatically converted to 3D pose.

Codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train).  We also use Stacked Hourglass model for 2D Pose estimation. We implement GUI and other niceties from strath using PyQT5.

* Installing DeepFly3D: [Installation](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/install.md)
* Interacting with the GUI:  [GUI](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/docs/gui.md)

## Auto-Correction
Auto-Correction is perfomed in case 2d pose estimation and calibration has completed. In the Correction mode, GUI tries to correct for errors using pictorial structures. In case you confirm the press ```T``` to make sure they are saved. Please check the paper for details.
![Alt text](images/correction.gif?raw=true "Title")

## Visualization
To visualize the pose_result file which is saved in the same folder as the images, you can use the notebook at ```notebook_visualize/visualize.ipynb```, by replacing the image folder in the first cell. Visualization should output figures close to the figures in the paper.


<img src="images/pose3D.png" width="960">

## 2D Pose Estimation on terminal
If we want to just use the already trained weights for doing 2d pose estimation, and obtain heatmaps and prediction.

```
python drosophila.py -s 8 --resume ./weights/sh8_mpii.tar --num-output-image 10
```

Heatmap and predictions files are saved into the same folder as pickle files. Notice that heatmaps can go large pretty fast. If you have problems, you might want to disable saving heatmaps. However, in this case, performing automatic corrections are not possible.

## Retraining

To reproduce the results from the paper, first download the necessarry weights using:

```
./weights/download.sh
```

Then, you can start training using the file 'example/drosophila.py'. --num-output-image flag is useful for debugging, which saves some images of the ground truth in a folder under checkpoint.

```
python example/drosophila.py -s 8 --resume ./weights/sh8_mpii.tar --num-output-image 10
```
