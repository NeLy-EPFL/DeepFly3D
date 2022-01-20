## Installing

## Installing with pip
Create a new anaconda environment, and pip install df3d package.
```bash
conda create -n df3d python=3.6
conda activate df3d
pip install df3d
```

## Old CUDA Drivers
**Only in case your cuda driver is not up-to-date**, additionally you might need to explicitly install cudatoolkit before pip installing df3d:

```bash
conda install pytorch torchvision torchaudio cudatoolkit="YOUR_CUDA_VERSION" -c pytorch
```

## Installing from the source
DeepFly3D requires Python3, Anaconda environment and CUDA drivers for installation. It is __only tested on Ubuntu and MacOS__. First, clone the repository:

```
git clone https://github.com/NeLy-EPFL/DeepFly3D
cd DeepFly3D
```
Then, run create a conda environment with
```
conda create -n df3d python=3.6
```
which will create a new python environment. Then, activate the environment.
```
conda activate df3d
```
Once this is done  you can install the **df3d** package with the following command,

```
pip install -e .
```

which uses the setup.py function to create the package.

Make sure you also have installed the CUDA drivers compatible with your GPU, otherwise it is not possible to make 2D predictions. You can check how to install CUDA drivers here: https://developer.nvidia.com/cuda-downloads
