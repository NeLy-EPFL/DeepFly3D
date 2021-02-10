1. you will have three sources of annotations. the first is the old annotations for df3d paper. these are few (<1000). then you have gizem's annotations, which is around ~2500 frames. Then we can create new annotations using calibration and projection. as few as ~20 annotations are seem to enough to calibrate. then you can get all other image annotations in the same experiment for free. to start with you can ignore df3d annotations as they are quite few.

2. to create run notebook cells on prepare_annotation.ipynb create annotations. it will be saved as gizem_annot_train.pkl. you will need df3d_anomaly folder for this (https://drive.google.com/open?id=1_9XyfuAxtTVmWrdffpl1epd0rjgKsqx4).

2. to train the network you need to run  DeepFly3D/deepfly/pose2d/drosophila.py file. you should get at least <1 pixel mse error on the test set for a meaningful prediction. training should take take close to a day. to use pkl files you just created as annotations, you need to change paths for (https://github.com/NeLy-EPFL/DeepFly3D/blob/f29daef079c15c0e75860eb26222fe457e36f27c/deepfly/pose2d/drosophila.py#L199) and (https://github.com/NeLy-EPFL/DeepFly3D/blob/f29daef079c15c0e75860eb26222fe457e36f27c/deepfly/pose2d/drosophila.py#L216). 

3. you can test your network using the following command of df3d-cli: 
    df3d-cli -vv /mnt/NAS/FA/191129_ABO/Fly2/001_coronal/behData/images/ -n 100  --output-folder df3d_test --video-2d  --resume-front "your/weights/after/training"
    this will run df3d on the first 100 images, and make a video of 2d predictions under images/df3d_test folder. just check the video. 


