import rospy
import numpy as np
import numpy.random as random
from geometry_msgs.msg import Pose
from typing import Dict, Any, Union
from scipy.spatial.transform import Rotation as R
from move_hand.path_planners import bezier_paths, navigation_function, path_planner

# TODO: Compute trajectories for the hand
# TODO: The hand orientation could always point towards some point, or the hand could be oriented tangent to the trajectory


class HandController:
    def __init__(self, move_hand_config : Dict[str, Any], hand_rotation : np.array):
        # Create the gazebo interface
        self._config = move_hand_config
        # To store the planned path
        self._pose_buffer = [] 
        
        # Assign the path planner
        try:
            if self._config["path_planner"] == "bezier":
                self._path_planner = bezier_paths.BezierPlanner()
            elif self._config["path_planner"] == "navigation_function":
                self._path_planner = navigation_function.NavFuncPlanner(world_dim=3, world_sphere_rad=self._config["outer_radius"]+0.5)
            else:
                raise ValueError("Path planner not recognized")
        except ValueError as e:
            rospy.logwarn("Failed to set path planner: ", e)
            rospy.logwarn("Defaulting to straight line path planner")
            self._path_planner = path_planner.PathPlanner()
        
        # Variable to store the pose of the hand's palm frame
        self._pose = Pose()
        
        # Store hand rotation
        self._hand_rotation = hand_rotation


    def update(self, hand_state : Dict[str, Any]) -> None:
        # Update the hand controller state
        self._pose = hand_state["pose"]


    def step(self) -> Union[Pose, np.array]:
        
        if len(self._pose_buffer) == 0:
            raise IndexError("Empty pose buffer")
        
        return self._pose_buffer.pop(0)
    
    
    def plan_trajectory(self, obj_center : np.array) -> None:
        # Obtain the start and goal pose
        start_pose = self._sample_start_pose(obj_center)
        goal_pose = self._sample_goal_pose(obj_center, start_pose)
        
        # Plan trajectory with the given path planner and parameters
        if self._config["path_planner"] == "bezier":
            path_params = {
                "num_way_points": random.randint(1, 5),
                "sample_type": "constant",
                "num_points": 1000,
            }
        elif self._config["path_planner"] == "navigation_function":
            path_params = {
                "num_rand_obs": random.randint(1, 5),
                "obs_rad": random.uniform(0.1, 0.5),
                "kappa": 5,
                "step_size": 0.1,
            }
        
        self._pose_buffer = self._path_planner.plan_path(start_pose[:3], goal_pose[:3], path_params)


    def reset(self):
        self._pose_buffer = []
    
    
    def _sample_start_pose(self, obj_center : np.array) -> np.array:
                # Sample starting point on outer sphere
        def sample_spherical() -> np.array:
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec, axis=0)
            vec *= self._config["outer_radius"]
            return vec
        
        # Compute the start position
        rel_pos = sample_spherical()
        start_position = rel_pos + obj_center

        # Compute the start orientation
        auxiliary_vec = np.array([0, 0, 1.])
        z_axis = -rel_pos

        y_axis = np.cross(auxiliary_vec, z_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # If x-axis is not pointing down, flip the orientation 180 degrees around the z-axis
        if x_axis[2] > 0:
            x_axis, y_axis = -x_axis, -y_axis

        # Create the rotation matrix and account for hand frame rotation  
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T @ self._hand_rotation
        # Convert it to quaternion
        start_orientation = R.from_matrix(rotation_matrix).as_quat()

        # Obtain the start pose as the combined position and orientation
        return np.concatenate([start_position, start_orientation])

    def _sample_goal_pose(self, obj_center : np.array, start_pose : np.array) -> np.array:
        rospy.logwarn_once("Goal set for testing purposes. Should be implemented in future")
        #TODO: Implement correct goal sampling
        return np.concatenate([(obj_center + np.array([0,0,0.1])), start_pose[3:]])


if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController()