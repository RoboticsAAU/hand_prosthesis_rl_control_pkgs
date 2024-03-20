## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['scripts/path_planners'],
    packages=[
        # Folders in the src directory
        'utils',
        # 'utilities.addons'  # Subpackage notation
    ],
    package_dir={'': 'src'},  # Packages are directly under 'src'
)

setup(**setup_args)


