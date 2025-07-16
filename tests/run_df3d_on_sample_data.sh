#!/bin/bash
# You can run this to confirm that your DeepFly3D installation is working correctly.
# If successful, you will get df3d results files in the directory `data/reference_df3d/`

df3d-cli data/reference/ -v --video-2d --video-3d
