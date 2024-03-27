#!/usr/bin/env python

import rospy
import numpy as np
import math
from typing import Dict, Any

# TODO: Compute trajectories for the hand
# TODO: The hand orientation could always point towards some point, or the hand could be oriented tangent to the trajectory


class HandController:
    def __init__(self, move_hand_config : Dict[str, Any]):
        # Create the gazebo interface
        self._move_hand_config = move_hand_config
        
        # Parameters for the state
        self._pose = None

    def update(self, state : Dict[str, Any]):
        # Update the hand controller state
        self._pose = state["pose"]

    def move_in_circle(self):
        # Method to move the hand in a circle 
        # Move the hand in a circle
        # Angular resolution:
        # Test move hand controller class
        # pose = Pose(position=Point(x=1.0, y=1.0, z=1.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0))
        # vel = Twist(linear=Point(x=0.0, y=0.0, z=0.0), angular=Point(x=0.0, y=0.0, z=0.0))
        vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        steps = 0
        while not rospy.is_shutdown():
            # pose.position.z += 0.001
            
            # vel.linear.x = - math.sin(steps * 0.001) 
            # vel.linear.y = math.cos(steps * 0.001) 
            # vel.angular.z = 3.0


            vel[0] = - math.sin(steps * 0.001)
            vel[1] = math.cos(steps * 0.001)
            vel[5] = 3.0
        
            self._gazebo_interface.set_velocity(vel)
            steps += 1
            self._gazebo_interface._rate.sleep()

            

if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController()
    hand_controller.move_in_circle()