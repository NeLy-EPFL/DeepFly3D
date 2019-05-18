## Installing
Drosophila-Pose requires Python3 and an Anaconda environment and __only tested on Ubuntu__. First, install the packages described in environment.xml in the anaconda format. This can be done by first cloning the repository:

```
git clone https://github.com/NeLy-EPFL/DeepFly3D.git
cd DeepFly3D
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

Once the environment is activated you need to install the deepfly package with the following command,

```
pip install -e .
```
which uses the setup.py function to create the package. For the last step you need to install the 2d pose estimation weights for stacked hourglass (approximately 200MB.) You can complete this with running the following **in the root folder of the project**:

```
./weights/download.sh
```

Make sure you also have installed the CUDA drivers compatible with your GPU. You can check how to install CUDA drivers here: https://developer.nvidia.com/cuda-downloads
