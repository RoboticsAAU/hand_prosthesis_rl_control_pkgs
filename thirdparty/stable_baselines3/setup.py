from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    version='0.1.0',
    packages=['stable_baselines3'],
    package_dir={'': 'src'},  # Packages are directly under 'src'
    # url='https://github.com/Kami-code/dexart-release',
    install_requires=[
        'transforms3d', 'gym==0.25.2', 'sapien==2.2.1', 'numpy', 'open3d==0.14.1'
    ],
)

setup(**setup_args)
