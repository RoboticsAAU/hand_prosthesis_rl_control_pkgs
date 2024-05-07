cmake_minimum_required(VERSION 2.8.3)
project(stable_baselines3)

# Set policy CMP0048 to manage version variables
cmake_policy(SET CMP0048 NEW)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

