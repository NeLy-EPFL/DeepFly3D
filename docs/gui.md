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
