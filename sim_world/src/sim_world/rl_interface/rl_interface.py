import numpy as np
import rospy
from typing import Type, Dict, Callable, Any
from pathlib import Path

from sim_world.world_interfaces.world_interface import WorldInterface
from sim_world.world_interfaces.simulation_interface import SimulationInterface
from sim_world.object_handler.object_handler import ObjectHandler
from move_hand.control.move_hand_controller import HandController


class RLInterface():
    def __init__(self, world_interface: SimulationInterface, rl_env_update : Callable, sim_config : Dict[str, Any]):
        # Save the world interface and the update methods
        self._world_interface = world_interface
        self._rl_env_update = rl_env_update
        
        # Save the object handler
        self._object_handler = ObjectHandler(sim_config["objects"])
        
        # Instantiate the hand controller
        self._hand_controller = HandController(sim_config["move_hand"])
        
        # Spawn objects in the gazebo world
        self.spawn_objects_in_grid()
        
        # Initialise subscriber data container
        self.subscriber_data = {}
        

    def step(self, input_values : Dict[str, Any]):
        # Update the world interface with the input values
        self.set_action(input_values["action"])
        self.move_hand(input_values["hand_pose"])
        
        # Extract all the values from the interface and put them in a dictionary
        # Some values may be set to none depending on the interface, need to make sure the update methods can handle this using checks. 
        self.subscriber_data = self._world_interface.get_subscriber_data()
        rl_data = {
            "hand_data": self.subscriber_data["rl_data"]["hand_data"], 
            "obj_data": self.subscriber_data["rl_data"]["obj_data"][self._object_handler.curr_obj]
        }
        # Update the rl environment
        self._rl_env_update(rl_data)
        # Update move_hand_controller with new pose
        self._hand_controller.update(self.subscriber_data["move_hand_data"])
        
    def set_action(self, action):
        """
        Set the action for the hand.
        """
        self._world_interface.hand.set_action(action)
    
    def move_hand(self, pose):
        """
        Publish the position of the hand to the world interface.
        """
        self._world_interface.set_pose(self._world_interface.hand.name, pose)
    
    
    def spawn_objects_in_grid(self):
        """
        Spawn the objects in a grid in the gazebo world.
        """
        def find_factors(n):
            factors = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            return factors
        
        factors = find_factors(self._object_handler.config["num_objects"])
        grid_dims = min(factors, key=lambda pair: abs(pair[0] - pair[1]))
        
        x_vals = np.linspace(0, (grid_dims[0] - 1) * self._object_handler.config["inter_object_dist"], grid_dims[0])
        y_vals = np.linspace(0, (grid_dims[1] - 1) * self._object_handler.config["inter_object_dist"], grid_dims[1])
        grid = np.meshgrid(x_vals, y_vals)
        
        for index, object in enumerate(self._object_handler.objects):
            # Get rotation and translation
            R = np.eye(3)
            t = np.array(grid[0].flatten()[index], grid[1].flatten()[index], object["height"])
            
            # Create the pose
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            # Spawn the object
            self._world_interface.spawn_object(object["name"], object["sdf"], T)
            
    def update_context(self):
        """
        Updates the current object context. This includes spawning hand and computing approach trajectory.
        mode: str
            The mode to update the context. Can be either "random" or "sequential".
        """
        self._object_handler.update_current_object()
        # self.subscriber_data["rl_data"]["obj_data"][self._object_handler.curr_obj]
        
        # select new object (either random or sequential)
        # move hand to given start pose
        # choose destination point
        # calculate hand trajectory

