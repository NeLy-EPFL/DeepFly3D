## Installing
DeepFly3D requires Python3, Anaconda environment and CUDA drivers for installation. It is __only tested on Ubuntu__. First, clone the repository:

```
git clone https://github.com/NeLy-EPFL/DeepFly3D.git
cd DeepFly3D
```
Then, run create a conda environment with
```
conda create -n deepfly python=3.6
```
which will create a new python environment. Then, activate the environment, and install jupyter notebook.
```
source activate deepfly
conda install jupyter
```
Once this is done  you can install the **deepfly** package with the following command,

```
pip install -e .
```

which uses the setup.py function to create the package. For the last step you need to install the 2d pose estimation weights for stacked hourglass (approximately 200MB.) You can complete this with running the following **in the root folder of the project**:

```
./weights/download.sh
```

Make sure you also have installed the CUDA drivers compatible with your GPU, otherwise it is not possible to make 2D predictions. You can check how to install CUDA drivers here: https://developer.nvidia.com/cuda-downloads
