import numpy as np

from geometry_msgs.msg import Pose
from typing import Dict, Callable, Any, Union

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
        self._hand_controller = HandController(sim_config["move_hand"], self._world_interface.hand.hand_rotation)
        
        # Spawn objects in the gazebo world
        self.spawn_objects_in_grid()
        
        # Initialise subscriber data container
        self.subscriber_data = {}
        

    def step(self, input_values : Dict[str, Any]):
        # Update the world interface with the input values
        self.set_action(input_values["action"])
        self.move_hand(self._hand_controller.step())
        
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
        
    def set_action(self, action : np.array):
        """
        Set the action for the hand.
        """
        self._world_interface.hand.set_action(action)
    
    def move_hand(self, pose : Union[Pose, np.array]):
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
        
        for index, (obj_name, obj_sdf) in enumerate(self._object_handler.objects.items()):
            # Get rotation and translation
            rotation_matrix = np.array([[1, 0, 0], 
                                        [0, 0, -1],
                                        [0, 1, 0]])
            #TODO: Add object spawn height. Currently hardcoded to 0.1
            position = np.array([grid[0].flatten()[index], grid[1].flatten()[index], 0.08])
            
            orientation = R.from_matrix(rotation_matrix).as_quat()
            
            pose = np.concatenate([position, orientation])
            rospy.logwarn_once("Spawning object: " + obj_name + " at pose: " + str(pose))
            # Spawn the object
            self._world_interface.spawn_object(obj_name, obj_sdf, pose)
    
    
    def update_context(self):
        """
        Updates the current object context. This includes spawning hand and computing approach trajectory.
        mode: str
            The mode to update the context. Can be either "random" or "sequential".
        """
        # Update the current object
        self._object_handler.update_current_object()
        
        # Update the hand controller trajectory
        object_pose = self.subscriber_data["rl_data"]["obj_data"][self._object_handler.curr_obj]
        obj_center = np.array([object_pose.position.x, object_pose.position.y, object_pose.position.z])
        self._hand_controller.plan_trajectory(obj_center)
        
        
