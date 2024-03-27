import numpy as np
from typing import Type, Dict, Callable, Any

from sim_world.world_interfaces.world_interface import WorldInterface
from sim_world.object_handler.object_handler import ObjectHandler



class RLInterface():
    def __init__(self, world_interface: Type[WorldInterface], update_methods: Dict[str, Callable], objects_config : Dict[str, Any]):
        # Save the world interface and the update methods
        self._world_interface = world_interface
        self._update_methods = update_methods
        
        # Save the object handler
        self._objects_config = objects_config
        self._object_handler = ObjectHandler()


    def step(self, input_values : Dict[str, Any]):
        # Update the world interface with the input values
        self.move_hand(input_values["hand_pose"])
        self.set_action(input_values["action"])
        
        # Extract all the values from the interface and put them in a dictionary
        # Some values may be set to none depending on the interface, need to make sure the update methods can handle this using checks. 
        output_values = self._world_interface.update()
        for method in self._update_methods.values():
            method(output_values)
    
    def set_action(self, action):
        # Set the action of the hand
        self._world_interface.hand.set_action(action)
    
    def move_hand(self, position):
        # Publish the position and velocity of the hand
        self._world_interface.publish_pose(position)
    
    def spawn_objects_in_grid(self):
        def find_factors(n):
            factors = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            return factors
        
        factors = find_factors(self._objects_config["num_objects"])
        grid_dims = min(factors, key=lambda pair: abs(pair[0] - pair[1]))
        
        x_vals = np.linspace(0, (grid_dims[0] - 1) * self._objects_config["inter_object_dist"], grid_dims[0])
        y_vals = np.linspace(0, (grid_dims[1] - 1) * self._objects_config["inter_object_dist"], grid_dims[1])
        grid = np.meshgrid(x_vals, y_vals)
        
        for index, object in enumerate(self._object_handler.get_random_objects()):
            # Get rotation and translation
            R = np.eye(3)
            t = np.array(grid[0].flatten()[index], grid[1].flatten()[index], object["height"])
            
            # Create the pose
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            self._world_interface.spawn_object(object["path"], T)
