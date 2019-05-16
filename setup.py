from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
        name='deepfly',
        packages=['deepfly'],
        entry_points = {'console_scripts' : ['drosoph = deepfly.GUI.main:main']},
        author="Semih Gunel",
        author_email="semih.gunel@epfl.ch",
        description="GUI and 3D pose estimation pipeline for tethered Drosophila.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/NeLy-EPFL/anipose",
        install_requires=['PyQt5',
                          'sklearn',
                          'numpy',
                          'scipy',
                          'scikit-video',
                          'skimage',
                          'matplotlib',
                          'torchvision',
                        ]
        )



