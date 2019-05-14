from setuptools import setup

setup(
        name='deepfly',
        packages=['deepfly'],
        entry_points = {'console_scripts' : ['drosoph = deepfly.GUI.main:main']}
        )
