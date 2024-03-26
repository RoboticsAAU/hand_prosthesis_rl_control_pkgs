
import numpy as np
from typing import Tuple

# Local libraries
from move_hand.gazebo_interface import GazeboInterface
from move_hand.move_hand_controller import HandController

# TODO: Step
# TODO: Reset
# TODO: Observations

# Convert actions from the reinforcement learning algorithm to the hand control commands
# Call and define what a step is in regards of the world. 
# Define what a reset is in regards to the move hand.
# 

class MoveHand():
    def __init__(self) -> None:

        # Private objects
        self._gazebo_interface = GazeboInterface(hand_name='mia_hand')
        self._move_hand_controller = GazeboInterface(hand_name='mia_hand')

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    def reset(self) -> np.ndarray:
        """ Sample a random position for the hand to be in. Compute orientation of the hand as well. """
        

        pass

    def get_observation(self) -> np.ndarray:
        pass

    
