import numpy as np
import rospy

from geometry_msgs.msg import Pose, Point, Quaternion
from typing import Dict, Callable, Any, Union, List
from scipy.spatial.transform import Rotation as R

from sim_world.world_interfaces.simulation_interface import SimulationInterface
from sim_world.object_handler.object_handler import ObjectHandler
from move_hand.control.move_hand_controller import HandController

class RLInterface():
    def __init__(self, world_interface: SimulationInterface, sim_config : Dict[str, Any]):
        # Save the world interface and the update methods
        self._world_interface = world_interface
        
        # Save the object handler
        self._object_handler = ObjectHandler(sim_config["objects"])
        
        # Instantiate the hand controller
        self._hand_controller = HandController(sim_config["move_hand"], self._world_interface.hand.hand_rotation)
        
        # Spawn objects in the gazebo world
        self._object_poses = {}
        self.spawn_objects_in_grid(np.array([1.0, 0.0]))
        
        # Spawn hand in an appropriate position
        self.default_pose = Pose(position=Point(0, 0, 0), orientation=Quaternion(0.7071, 0, 0, 0.7071))
        self.move_hand(self.default_pose)
        
        # Initialise subscriber data container
        self._subscriber_data = {}
        

    def step(self, action : np.array) -> bool:
        """
        Function to step the RL interface.
        Returns True if the episode is done.
        """
        
        # Update the world interface with the input values
        self.set_action(action)
        # self.move_hand(self._hand_controller.step()[0])
        self.set_hand_poses(self._hand_controller.step(num_steps=50))
        
        
        # Extract all the values from the interface and put them in a dictionary
        # Some values may be set to none depending on the interface, need to make sure the update methods can handle this using checks. 
        self._subscriber_data = self._world_interface.get_subscriber_data()

        # Update move_hand_controller with new pose
        self._hand_controller.update(self._subscriber_data["move_hand_data"])
        
        # TODO: Remove later
        # Check if episode is done
        return self._hand_controller._pose_buffer.shape[1] == 0
        
    
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
    
    def set_hand_poses(self, poses : Union[List[Pose], np.ndarray]):
        self._world_interface.send_hand_poses(poses)
        
    
    def spawn_objects_in_grid(self, offset : np.array = np.zeros(2)):
        """
        Spawn the objects in a grid in the gazebo world.
        """
        # Find the x and y directions of the grid. The grid is chosen to be as square as possible. The factors resolve the number of objects into two factors that when multiplied together, gives n.
        def find_factors(n):
            factors = []
            # we only need to look up to the square root of n, because the other factor will be found by dividing n by the first factor
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            return factors
        
        factors = find_factors(self._object_handler.config["num_objects"])
        grid_dims = min(factors, key=lambda pair: abs(pair[0] - pair[1]))
        
        x_vals = np.linspace(0, (grid_dims[0] - 1) * self._object_handler.config["inter_object_dist"], grid_dims[0]) + offset[0]
        y_vals = np.linspace(0, (grid_dims[1] - 1) * self._object_handler.config["inter_object_dist"], grid_dims[1]) + offset[1]
        grid = np.meshgrid(x_vals, y_vals)
        
        for index, (obj_name, obj) in enumerate(self._object_handler.objects.items()):
            # Get rotation and translation
            rotation_matrix = np.array([[1, 0, 0], 
                                        [0, 0, -1],
                                        [0, 1, 0]])
            #TODO: Add object spawn height. Currently hardcoded to 0.08
            position = np.array([grid[0].flatten()[index], grid[1].flatten()[index], 0.08])
            
            orientation = R.from_matrix(rotation_matrix).as_quat()
            
            pose = np.concatenate([position, orientation])

            # Append pose to dict
            self._object_poses[obj_name] = pose

            # Spawn the object
            self._world_interface.spawn_object(obj_name, obj["sdf"], pose)
    
    
    def reset_object(self, obj_name : str):
        """
        Reset the pose of the current object.
        """
        self._world_interface.set_pose(obj_name, self._object_poses[obj_name])
    
    
    def update_context(self):
        """
        Updates the current object context. This includes spawning hand and computing approach trajectory.
        """
        
        # Update the current object
        if self._object_handler.curr_obj is not None:
            # self.move_hand(self.default_pose)
            self.set_hand_poses([self.default_pose])
            self.reset_object(self._object_handler.curr_obj)
            
        self._object_handler.update_current_object()
        
        # Update the hand controller trajectory
        object_pose = self.rl_data["obj_data"]
        obj_center = np.array([object_pose.position.x, object_pose.position.y, object_pose.position.z])
        self._hand_controller.plan_trajectory(obj_center, self._object_handler.objects[self._object_handler.curr_obj]["mesh"])
    
    
    @property
    def rl_data(self):
        self._subscriber_data = self._world_interface.get_subscriber_data()
        return {
            "hand_data": self._subscriber_data["rl_data"]["hand_data"], 
            "obj_data": self._subscriber_data["rl_data"]["obj_data"][self._object_handler.curr_obj]
        }
