from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="df3d",
    version="0.52",
    packages=["df3d"],
    entry_points={
        "console_scripts": ["df3d = df3d.gui:main", "df3d-cli = df3d.cli:main"]
    },
    author="Semih Gunel",
    author_email="gunelsemih@gmail.com",
    description="GUI and 3D pose estimation pipeline for tethered Drosophila.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/NeLy-EPFL/DeepFly3D",
    install_requires=[
        "PyQt5",
        "sklearn",
        "scipy<=1.2.1",
        "scikit-video",
        "scikit-image",
        "matplotlib",
        "opencv-python==4.1.2.30",
        "tqdm",
        "colorama",
        "progress",
        "pytorch_lightning",
        "pyba",
        "df2d",
    ],
)
