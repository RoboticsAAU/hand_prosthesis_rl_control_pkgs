#!/usr/bin/env python

import rospy
import numpy as np
import math
from geometry_msgs.msg import Pose
from typing import Dict, Any

# TODO: Compute trajectories for the hand
# TODO: The hand orientation could always point towards some point, or the hand could be oriented tangent to the trajectory


class HandController:
    def __init__(self, move_hand_config : Dict[str, Any]):
        # Create the gazebo interface
        self._config = move_hand_config
        self._pose_buffer = []
        
        # Parameters for the state
        self._pose = None

    def update(self, hand_state : Dict[str, Any]):
        # Update the hand controller state
        self._pose = hand_state["pose"]

    def step(self) -> None:
        pass
    
    
    def plan_trajectory(self, obj_center : np.array) -> None:
        
        # Sample starting point on outer sphere
        def sample_spherical() -> np.array:
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec, axis=0)
            vec *= self._config["outer_radius"]
            return vec
        
        # Compute start pose
        rel_pos = sample_spherical()
        start_position = obj_center + rel_pos
        start_orientation = 
        
        # choose start pose
        # chose goal pose
        # plan trajectory


    def reset(self):
        pass

if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController()
    hand_controller.move_in_circle()