# hand_prosthesis_rl_control_pkgs
This is the repository related to a project called "Semi-autonomous control of Hand Prosthesis using computer vision". 

## Install
The code has been tested on
* Ubuntu 20.04 with ROS Noetic

### Install dependencies
```sh
sudo apt-get install python3-catkin-tools python3-vcstool
```

### Clone and compile
Create a workspace and clone the code:
```sh
mkdir -p hand_prosthesis_ws/src && cd hand_prosthesis_ws/src
git clone https://github.com/RoboticsAAU/hand_prosthesis_rl_control_pkgs.git
vcs-import < ./hand_prosthesis_rl_control_pkgs/dependencies.yaml
cd ..
```

Now the code can be compiled (make command is optional:
```sh
catkin_make --cmake-args "-DCMAKE_EXPORT_COMPILE_COMMANDS=On"
```