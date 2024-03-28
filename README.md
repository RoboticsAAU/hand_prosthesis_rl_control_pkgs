# hand_prosthesis_rl_control_pkgs
This is the repository related to a project called "Semi-autonomous control of Hand Prosthesis using computer vision". 

## Install
The code has been tested on
* Ubuntu 20.04 with ROS Noetic

### Install dependencies
Install [catkin tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) and [vcstools](https://github.com/dirk-thomas/vcstool).
```sh
sudo apt-get install python3-catkin-tools python3-vcstool python3-osrf-pycommon
```


### Clone and compile
Create a workspace and clone the code:
```sh
mkdir -p hand_prosthesis_ws/src && cd hand_prosthesis_ws/src
git clone https://github.com/RoboticsAAU/hand_prosthesis_rl_control_pkgs.git
pip3 install -r ./hand_prosthesis_rl_control_pkgs/requirements.txt
vcs-import < ./hand_prosthesis_rl_control_pkgs/dependencies.yaml
rosdep install --from-paths src --ignore-src -r -y
cd ..
```

Now the code can be compiled (build option is not required):
```sh
catkin_make --cmake-args "-DCMAKE_EXPORT_COMPILE_COMMANDS=On"
```

