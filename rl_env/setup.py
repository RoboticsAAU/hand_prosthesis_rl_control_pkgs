## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    #scripts=['scripts/test.py'],
    packages=[
        'gazebo',
        'robot_envs',
        'task_envs',
        'training',
        'utilities',
        'utilities.addons'  # Subpackage notation
    ],
    package_dir={'': 'src'}  # Packages are directly under 'src'
)

setup(**setup_args)
