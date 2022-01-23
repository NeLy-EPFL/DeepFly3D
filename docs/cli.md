# Data Structure

The inteded usage of DeepFly3D is through command-line-intarface (CLI).
df3d-cli assumes there are videos or images in this format under the folder. 
if your path /your/image/path has images or videos, df3d-cli will run 2D pose estimation, calibration and triangulation and will save 2d pose, 3d pose and calibration parameters under the folder /your/image/path/df3d.

Idealy you would have images or videos under ```images/``` folder, with the specific naming convention:
```
.
+-- images/
|   +-- camera_0_img_0.jpg
|   +-- camera_1_img_0.jpg
|   +-- camera_2_img_0.jpg
|   +-- camera_3_img_0.jpg
|   +-- camera_4_img_0.jpg
|   +-- camera_5_img_0.jpg
|   +-- camera_6_img_0.jpg
...
```
or 

```
.
+-- images
|   +-- camera_0.mp4
|   +-- camera_1.mp4
|   +-- camera_2.mp4
|   +-- camera_3.mp4
|   +-- camera_4.mp4
|   +-- camera_5.mp4
|   +-- camera_6.mp4
```

In case of mp4 files, df3d will first expand them into images using ffmpeg. Please check the sample data for a real exampe: https://github.com/NeLy-EPFL/DeepFly3D/tree/master/sample/test

## Basic Usage

The basic usage is like this.
```bash
df3d-cli /your/image/path \
         --order 0 1 2 3 4 5 6 
```

camera order stands for the selection of cameras. The default camera ordering (0 1 2 3 4 5 6) stands for this. In case you have some other order, then you need to  tell which order.


Originally. 

![](https://github.com/NeLy-EPFL/DeepFly3D/blob/dev/images/camera_order.png)


Then if you have the following order, your
![image](https://user-images.githubusercontent.com/20509861/150675023-099f3d24-3c99-47bf-a2de-e2aa3665fdc9.png)



So for example, if your data looks like this, then your order should be 6 5 4 3 2 1 0.
![image](https://user-images.githubusercontent.com/20509861/150674985-c0035ab5-2b55-4dd0-8ffe-fc364857dae7.png)


# Advanced Usage

```bash
usage: df3d-cli [-h] [-v] [-vv] [-d] [--output-folder OUTPUT_FOLDER] [-r] [-f]
                [-o] [-n NUM_IMAGES_MAX]
                [-order [CAMERA_IDS [CAMERA_IDS ...]]] [--video-2d]
                [--video-3d] [--skip-pose-estimation]
                INPUT

DeepFly3D pose estimation

positional arguments:
  INPUT                 Without additional arguments, a folder containing
                        unlabeled images.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable info output (such as progress bars)
  -vv, --verbose2       Enable debug output
  -d, --debug           Displays the argument list for debugging purposes
  --output-folder OUTPUT_FOLDER
                        The name of subfolder where to write results
  -r, --recursive       INPUT is a folder. Successively use its subfolders
                        named 'images/'
  -f, --from-file       INPUT is a text-file, where each line names a folder.
                        Successively use the listed folders.
  -o, --overwrite       Rerun pose estimation and overwrite existing pose
                        results
  -n NUM_IMAGES_MAX, --num-images-max NUM_IMAGES_MAX
                        Maximal number of images to process.
  -order [CAMERA_IDS [CAMERA_IDS ...]], --camera-ids [CAMERA_IDS [CAMERA_IDS ...]]
                        Ordering of the cameras provided as an ordered list of
                        ids. Example: 0 1 4 3 2 5 6.
  --video-2d            Generate pose2d videos
  --video-3d            Generate pose3d videos
  --skip-pose-estimation
                        Skip 2D and 3D pose estimation
```

Therefore, you can create advanced queries in df3d-cli, for example:

```bash
df3d-cli -f /path/to/text.txt    \  # process each line from the text file 
         -r                      \  # recursively search for images folder under each line of the text line
         --order 0 1 2 3 4 5 6   \  # set the camera order
         --output-folder results \  # write results under  /your/image/path/results instead of  /your/image/path/df3d
         --vv                    \  # will print agressivelly, for debugging purposes
         --skip-pose-estimation  \  # will not run 2d pose estimation, instead will do calibration, triangulation and will save results
         --video-2d              \  # will make 2d video for each folder 
         --video-3d              \  # will make 3d video for each folder

```
# GUI

The changed poses will be saved under pose_corr files under output folder. 

# Output
df3d_result
which keys
