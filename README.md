# DeepFly3D

![Alt text](images/gui.gif?raw=true "Title")

Drosophila-GUI is a PyTorch and PyQT5 implementation of the general processing of 2D-3D single tethered drosophila pose estimation. The aim is to provide the interface of estimation and further correction GUI for the 2D pose estimation, which is automatically converted to 3D pose.

Codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train).  We also use Stacked Hourglass model for 2D Pose estimation. We implement GUI from strath using PyQT5.

## Installing
Drosophila-Pose requires Python3 and an Anaconda environment. First, install the packages described in environment.xml in the anaconda format. This can be done by first cloning the repository:

```
git clone https://github.com/NeLy-EPFL/drosoph3D.git
cd drosoph3D
```
Then installing the necessary dependencies:
```
source env create --name NAME --file FILE 
```
which downloads the necessary packages using anaconda. 

For our case we will use

```
source env create --name drosoph --file environment.yml 
```

This command creates a new environment called drosoph and installs all the dependencies. Now you can activate the environment using:

```
source activate drosoph
```

Once the environment is activated you need to install the drosoph3D package with the following command,

```
pip install -e .
```
which uses the setup.py function to create the package. For the last step you need to install the 2d pose estimation weights for stacked hourglass (approximately 200MB.) You can complete this with running the following **in the root folder of the project**:

```
./weights/download.sh
```

Make sure you also have installed the CUDA drivers compatible with your GPU. You can check how to install CUDA drivers here: https://developer.nvidia.com/cuda-downloads

## Running GUI
After installing the dependencies we can initiliza the GUI using:

```
python -m drosoph3D.GUI.main /FULL/PATH/FOLDER NUM_IMAGES
```

For example:

```
python -m drosoph3D.GUI.main ./data/test/ 100
```

Or simply with the command line entry point:
```
drosoph ./data/test/ 100
```

This should start the GUI:

![Alt text](images/gui.png?raw=true "Title")


you can optionally remove `/FULL/PATH_FOLDER` and `NUM_IMAGES`, in which case pop-up apperas the select the folder. 

<img src="images/pop-up.png" width="480">

Before starting 2d pose estimation, __we need to make sure cameras are in the right order__, by using the button **Rename Images**. For instance, the 

![Alt text](images/wrong_order.png?raw=true "Title")

we can fix the order by clicking **Rename Images**, we can specify the order as:

![Alt text](images/rename.png?raw=true "Title")

which will fix the reordering issue:

![Alt text](images/correct_order.png?raw=true "Title")

Then, use **2d pose estimation** button to do 2d pose estimation. But, first make sure you installed the CUDA drivers and downloaded the necessary weights. After completing pose estimation with the **2d pose estimation** button, you can open the pose mode:

![Alt text](images/pose.png?raw=true "Title")

or heatmap mode to visualize the network predictions directly

![Alt text](images/heatmap.png?raw=true "Title")

## Pose Estimation and Calibration
2D pose estimation and calibration can be completed using the buttons on the GUI. They then enable automatic corrections and generating corresponding 3D pose. 
Calibraiton file is automatically saved in to the folder.  Once pose estimation and calibration are done you can extract the 2d and 3d pose using  ```Pose Save``` button. 

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
