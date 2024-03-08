## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    script=['training/train.py'],
    packages=['gazebo', 'robot_envs', 'task_env', 'training'],
    package_dir={'': 'src'},
)

setup(**setup_args)