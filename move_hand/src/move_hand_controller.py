#!/usr/bin/env python

import rospy
import numpy as np
import math
from gazebo_interface import GazeboInterface


class HandController:
    def __init__(self):
        # Create the gazebo interface
        self._gazebo_interface = GazeboInterface()

    def move_hand(self, position):
        # Publish the position and velocity of the hand
        self._gazebo_interface.publish_position(position)

    def move_in_circle(self):
        # Method to move the hand in a circle 
        # Move the hand in a circle
        # Angular resolution:
        resolution = 0.001
        i = 0
        while not rospy.is_shutdown():
            position = [math.cos((math.pi * 2) /  (i * resolution)), math.sin((math.pi * 2) / (i * resolution)), 1.0, 0.0, 0.0, 0.0]
            self.move_hand(position)
            
        pass


if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController()