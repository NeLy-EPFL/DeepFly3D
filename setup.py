from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deepfly",
    version='0.2',
    packages=["deepfly"],
    entry_points={"console_scripts": ["df3d = deepfly.gui:main", "df3d-cli = deepfly.cli:main"]},
    author="Semih Gunel",
    author_email="semih.gunel@epfl.ch",
    description="GUI and 3D pose estimation pipeline for tethered Drosophila.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/DeepFly3D",
    install_requires=[
        "PyQt5",
        "sklearn",
        "scipy<=1.2.1",
        "scikit-video",
        "scikit-image",
        "matplotlib",
        "torchvision",
        "opencv-python>=3.4.0.12",
        "tqdm",
        "colorama",
        "progress",
        "pims"
    ],
)
