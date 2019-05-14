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
