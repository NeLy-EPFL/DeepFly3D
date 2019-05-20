## Visualization
In general, displaying pose estimation results should be as easy as:

```python
import matplotlib.pyplot as plt
from deepfly.GUI.CameraNetwork import CameraNetwork
camNet = CameraNetwork(image_folder=image_folder)
image_folder = './data/test'

plt.imshow(camNet[1].plot_2d())
```
and to display heatmaps: 

```python
plt.imshow(camNet[1].plot_2d())
```

To create more complicated figures, or replicate the figures from the paper, you can use the the pose_result file which is saved in the same folder as the images. The notebook, ```notebook_visualize/visualize.ipynb```, shows you the steps to create the following figure:

<img src="../images/pose3D.png" width="960">

To visualize the time series instead, use the notebook ```notebook_visualize/time_series.ipynb```. It should output 2D/3D pose, along with a few selected time series.

<p align="center">
<img src="../images/time_series.png" width="640">
</p>
