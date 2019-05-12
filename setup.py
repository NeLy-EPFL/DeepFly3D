from setuptools import setup

setup(
        name='drosoph3D',
        packages=['drosoph3D'],
        entry_points = {'console_scripts' : ['drosoph = drosoph3D.GUI.main:main']}
        )
