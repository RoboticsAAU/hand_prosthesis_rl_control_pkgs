from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name="stable_baselines3",
    version='0.1.0',
    packages=['stable_baselines3'],
    package_dir={'': 'src'},  # Packages are directly under 'src'
    install_requires=[
        "gymnasium>=0.28.1,<0.30",
        "numpy>=1.20",
        "torch>=1.13",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
    ],
    python_requires=">=3.8",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/DLR-RM/stable-baselines3",
        "Documentation": "https://stable-baselines3.readthedocs.io/",
        "Changelog": "https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html",
        "SB3-Contrib": "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

setup(**setup_args)