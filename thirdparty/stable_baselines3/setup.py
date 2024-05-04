from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

setup(
    name='stablebaselines3',
    version='0.1.0',
    packages=find_packages(),
    # url='https://github.com/Kami-code/dexart-release',
    license='',
    author="Xiaolong Wang's Lab",
    install_requires=[
        'transforms3d', 'gym==0.25.2', 'sapien==2.2.1', 'numpy', 'open3d==0.14.1'
    ],
)