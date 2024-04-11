import rospy
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
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
        self._pose_buffer = np.array([])
        
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


    def step(self) -> np.array:
        
        if self._pose_buffer.shape[1] == 0:
            raise IndexError("Empty pose buffer")
        
        first_pose, self._pose_buffer = self._pose_buffer[:,0], self._pose_buffer[:,1:]
        return first_pose
    
    
    def plan_trajectory(self, obj_center : np.array) -> None:
        # Obtain the start and goal pose
        #TODO: z-offset should be a parameter in yaml
        start_pose = self._sample_start_pose(obj_center, 0.1)
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
        
        #TODO: Implement orientation in the path planner
        self._pose_buffer = self._path_planner.plan_path(start_pose[:3], goal_pose[:3], path_params)
        to_append = np.vstack([start_pose[3:].copy()] * self._pose_buffer.shape[1]).T
        self._pose_buffer = np.append(self._pose_buffer, to_append, axis=0)
        

    def reset(self):
        self._pose_buffer = []
    
    
    def _sample_start_pose(self, obj_center : np.array, z_offset : float) -> np.array:
        """
        Sample a start pose for the hand. The start pose is sampled on the upper hemisphere.
        obj_center : np.array, The center of the object
        z_offset : float, Hemisphere offset from the object center in the z-direction
        """
        
        # Sample starting point on upper-half of unit sphere (i.e., z>0)
        def sample_spherical() -> np.array:
            vec = np.random.uniform(-1, 1, (3,))
            # Convert the point to upper hemisphere
            vec[2] = abs(vec[2])
            # Add offset
            vec /= np.linalg.norm(vec, axis=0)
            return vec
        
        # Compute the start position
        rel_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        while rel_pos[2] < z_offset/self._config["outer_radius"]:
            rel_pos = sample_spherical()
        
        start_position = rel_pos*self._config["outer_radius"] + obj_center

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
        #TODO: Implement correct goal sampling. E.g. using graspit, bounding bo, or minkowski
        # Compute relative vector
        rel_pos = start_pose[:3] - obj_center
        # Normalise relative vector
        rel_pos /= np.linalg.norm(rel_pos)
        # Multiply by inner radius
        rel_pos *= self._config["inner_radius"]
        # Convert back to world frame
        pos = rel_pos + obj_center         
        return np.concatenate([pos, start_pose[3:]])


if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController()