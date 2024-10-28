import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nely-df3d",
    version="0.60",
    description="GUI and 3D pose estimation pipeline for tethered Drosophila.",
    author="Semih Gunel",
    author_email="gunelsemih@gmail.com",
    entry_points={
        "console_scripts": ["df3d = df3d.gui:main", "df3d-cli = df3d.cli:main"]
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/DeepFly3D",
    python_requires=">=3.8",
    install_requires=[
        "colorama",
        "matplotlib",
        "numpy",
        "opencv-python-headless>=4.8.1.78", # https://github.com/NeLy-EPFL/DeepFly3D/security/dependabot/4
        "PyQt5",
        "scikit-learn",
        "tqdm",
        "nely-pyba>=0.13",
        "nely-df2d>=0.14",
    ],
)
