## Training

To reproduce the results from the paper, first download the necessarry weights using from the root folder of the project. This will download the weights for MPII dataset, together with the final weights for the paper:

```
./weights/download.sh
```

You can train the network starting from the MPII dataset. You can read more details about the meaning of each argument from [ArgParse](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/deepfly/pose2d/ArgParse.py) file. 

```
python pose2d/drosophila.py -s 8 --resume ./weights/sh8_mpii.tar 
```

This will create a new folder called checkpoint and will save training logs and loss curves. This will also save the best weights with the lowest validation error as  ```model_best.pth.tar```. To also save example training and validation images, use 
```--num-output-image``` argument, and set it to a non-zero value.



### Using DeepFly3D Annotation Tool for Training

```
python pose2d/drosophila.py -s 8 --resume ./weights/sh8_mpii.tar --num-output-image 10
```
To use the annotation tool, download the resulting json file from Google Firebase and place the image folders under the ```data``` folder. Then, set the  DeepFly3D will automatically parse the json file and add the annotated images to the training. Set ```jsonfile="drosophilaimaging-export.json``` and  ```self.json_file = os.path.join("../../data/", jsonfile)``` lines accordingly in the to make sure dataset file can find it. 


### Using Automatic/Manual Corrections for Training
Manual/Automatic corrections are stored inside the ```pose_corr*.pkl``` files under the image folder. To also incorporate them into the training, go to [pose2d/dataset/drosophila.py](https://github.com/NeLy-EPFL/DeepFly3D/blob/master/deepfly/pose2d/datasets/drosophila.py), and set the ```manual_path_list = ['../data/test']``` variable. DeepFly3D will search __recursively__ on the specified folder to find the correction files and will automatically add them.



## Prediction
For only making prediction, use ```--unlabeled``` argument. For instance,
```
python pose2d/drosophila.py --resume ./weights/sh8_deepfly.tar --unlabeled ../data/test
``` 
will estimate 2D pose and heatmaps again and will replace the current version. This saves the resulting heatmaps and prediction under the ```--unlabeled``` folder
