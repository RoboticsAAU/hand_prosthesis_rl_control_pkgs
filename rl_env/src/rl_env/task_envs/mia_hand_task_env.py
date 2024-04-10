import rospy
import numpy as np
import rospkg
import open3d as o3d
import gym
from gym.envs.registration import register
from pathlib import Path
from functools import cached_property
from typing import Dict, List, Any
from geometry_msgs.msg import Pose

from rl_env.utils.tf_handler import TFHandler
from rl_env.utils.point_cloud_handler import PointCloudHandler, ImaginedPointCloudHandler
from rl_env.utils.urdf_handler import URDFHandler

OBJECT_LIFT_LOWER_LIMIT = 0.03

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MiaHandWorld-v0',
        entry_point='task_envs.mia_hand_task_env:MiaHandWorldEnv',
        #timestep_limit=timestep_limit_per_episode,
    )

class MiaHandWorldEnv(gym.Env):
    def __init__(self, visual_sensor_config : Dict[str, Any], limits_config : Dict[str, Any]):
        """
        This Task Env is designed for having the Mia hand in the hand grasping world.
        It will learn how to move around without crashing.
        """
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MiaHandWorldEnv, self).__init__()        
        
        # Initialise handlers
        rospack = rospkg.RosPack()
        urdf_path = rospack.get_path("sim_world") + "/urdf/hands/mia_hand_default.urdf"
        self._urdf_handler = URDFHandler(urdf_path)
        self._pc_cam_handler = PointCloudHandler()
        self._pc_imagine_handler = ImaginedPointCloudHandler()
        self._tf_handler = TFHandler()
        
        # Get the configurations for the cameras and the imagined point clouds
        self._config_imagined = visual_sensor_config["config_imagined"]
        self._config_cameras = visual_sensor_config["config_cameras"]
        self._imagined_groups = {}
        
        # Bounds for joint positions
        self._joint_limits = limits_config["joint_limits"]
        self._pos_lb = np.array([self._joint_limits[finger + "_pos_range"][0] for finger in ["thumb", "index", "mrl"]])
        self._pos_ub = np.array([self._joint_limits[finger + "_pos_range"][1] for finger in ["thumb", "index", "mrl"]])
        
        # Bounds for joint velocities
        self._vel_limits = limits_config["velocity_limits"]
        self._vel_lb = np.array([self._vel_limits[finger + "_vel_range"][0] for finger in ["thumb", "index", "mrl"]])
        self._vel_ub = np.array([self._vel_limits[finger + "_vel_range"][1] for finger in ["thumb", "index", "mrl"]])
        
        # Save number of _joints
        self._dof = len(self._joint_limits)
        
        # Parameters for the state and observation space
        self._joints = None
        self._joints_vel = None
        self._pc_cam_handler.pc.append(o3d.geometry.PointCloud())
        self._object_pose = Pose()
        
        # Print the spaces
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self._action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self._obs_space))
        
        # TODO: Define these in setup
        self._end_episode_points = 0.0
        self._cumulated_steps = 0.0
        self._finger_object_dist = np.zeros(3)
    

    def step(self, action):
        pass
    
    def reset(self):
        pass
    
    def update(self, rl_data : Dict[str, Any]):
        """
        Update the values of the hand data (state and observation)
        """
        # Update the hand data
        self._joints = rl_data["hand_data"]["joints_pos"]
        self._joints_vel = rl_data["hand_data"]["joints_vel"]
        self._pc_cam_handler.pc[0] = rl_data["hand_data"]["point_cloud"]
        self._object_pose = rl_data["obj_data"]
        # self._object_pose = Pose()
    
    # Methods needed by the TrainingEnvironment
    def init_env_variables(self):
        """
        Inits variables needs to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set episode_done to false, because it's calculated asyncronously
        self._episode_done = False


    def set_action(self, action):
        """
        This method will set the velocity of the hand based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        
        # Get the action
        action = np.array([action[0], action[1], action[2]])
        
        # Get the new joint velocities
        velocities = self._joints_vel + action
        velocities.clip(self._as_low, self._as_high)
        
        self.move_fingers(velocities)
        
        rospy.logdebug("END Set Action ==>"+str(action))


    def get_obs(self):
        """
        Fetch observations from the Mia Hand
        :return: observation
        """
        rospy.logdebug("Start Get Observation ==>")

        observation = {
            "joints" : self._joints,
            "joints_vel" : self._joints_vel,
            "points" : self._pc_cam_handler.points
        }
        
        rospy.logdebug("Observations==>"+str(observation))
        rospy.logdebug("END Get Observation ==>")
        return observation
        

    def is_done(self):
        # We check if it has crashed based on the imu
        if self._object_pose.position.z > OBJECT_LIFT_LOWER_LIMIT:
            self._episode_done = True

        return self._episode_done


    def compute_reward(self):
        """
        Compute the reward for the given rl step
        :return: reward
        """
        # Check if episode is done
        if self.is_done():
            return self._end_episode_points
        
        # Obtain the shortest distance between finger and object
        finger_object_dist = np.min(self._finger_object_dist)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        
        # Obtain the combined joint velocity
        clipped_vel = np.clip(self._joints_vel, self._vel_lb, self._vel_ub)
        combined_joint_vel = np.sum(np.abs(clipped_vel))
        
        # Check if at least three fingers are in contact with object
        fingers_in_contact = np.sum(self._finger_object_dist < 0.03)
        
        # Reward for energy expenditure (based on distance to object)
        reward = -(finger_object_dist * combined_joint_vel) * 0.01
        if fingers_in_contact >= 2:
            # Reward for contact
            reward += 0.5 * fingers_in_contact
            
            # Reward for lifting object
            lift = np.clip(self._object_pose.position.z, 0, 0.2)
            reward += 10 * lift

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        
        return reward


    @cached_property
    def _action_space(self):
        return gym.spaces.Box(self._vel_lb, self._vel_ub, dtype = np.float32)


    @cached_property
    def _obs_space(self):
        state_space = gym.spaces.Box(self._pos_lb, self._pos_ub, dtype = np.float32)
        obs_dict = {"state": state_space}
        
        for cam_name, cam_config in self._config_cameras.items():
            for modality_name in cam_config.keys():
                key_name = f"{cam_name}-{modality_name}" 
                
                if modality_name == 'rgb':
                    resolution = cam_config[modality_name]["resolution"]
                    spec = gym.spaces.Box(low=0, high=1, shape=resolution + (3,))
                
                elif modality_name == 'depth':
                    max_depth = cam_config[modality_name]["max_depth"]
                    resolution = cam_config[modality_name]["resolution"]
                    spec = gym.spaces.Box(low=0, high=max_depth, shape=resolution + (1,))
                
                elif modality_name == 'point_cloud':
                    spec = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((cam_config[modality_name]["num_points"],) + (3,)))
                    
                else:
                    raise RuntimeError("Modality not supported")              
            
            obs_dict[key_name] = spec
            
        if self._config_imagined is not None:
            self.update_imagination()
            obs_dict["imagination"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((self._config_imagined["num_points"],) + (3,)))
            
        return gym.spaces.Dict(obs_dict)
    
    def get_all_observations(self):
        all_obs = self.get_camera_obs()
        state_obs = self.get_state_obs()
        all_obs.update(state_obs)
        return all_obs
    
    def get_state_obs(self):
        raise NotImplementedError()
        
    def get_camera_obs(self):
        # Initialize the observation dictionary
        obs_dict = {}
        
        # Get the observations from the cameras
        for cam_name, cam_config in self._config_cameras.items():
            for modality_name, modality_config in cam_config.items():
                key_name = f"{cam_name}-{modality_name}"
                
                if modality_name == 'rgb':
                    raise NotImplementedError()
                
                elif modality_name == 'depth':
                    raise NotImplementedError()
                
                elif modality_name == 'point_cloud':
                    # Remove table and enforce cardinality
                    self._pc_cam_handler.remove_plane()
                    self._pc_cam_handler.update_cardinality(modality_config["num_points"])

                    # Transform point cloud to reference frame
                    transform = self._tf_handler.get_transform_matrix(cam_name, modality_config["ref_frame"])
                    self._pc_cam_handler.transform(transform)
                    obs_dict[key_name] = self._pc_cam_handler.points[0]
                    
                else:
                    raise RuntimeError("Modality not supported")
        
        # Get the observations from the imagination
        if self._config_imagined is not None:
            self.update_imagination()
            obs_dict["imagined"] = self._pc_imagine_handler.points[0]
        
        return obs_dict
        
    def setup_imagination(self):
        # config has: "stl_files", "ref_frame", "groups", "num_points"
        # Define the imagined groups where each group corresponds to a movable joint in the hand (along with base palm)
        free_joints = self._urdf_handler.get_free_joints()
        self._imagined_groups = {free_joint : self._urdf_handler.get_free_joint_group(free_joint) for free_joint in free_joints}
        self._imagined_groups["j_" + self._config_imagined["ref_frame"]] = self._urdf_handler.get_link_group(self._config_imagined["ref_frame"])
        
        mesh_dict = {}  # Initialize an empty dictionary
        for stl_file in self._config_imagined["stl_files"]:
            # Get the origin and scale for the mesh
            visual_origin, scale = self._urdf_handler.get_visual_origin_and_scale(Path(stl_file).stem)
            
            # Get the link name assoicated with the mesh and obtain the origin (transformation) from the reference
            link_name = self._urdf_handler.get_link_name(Path(stl_file).stem)
            link_origin = self._urdf_handler.get_link_transform(self._config_imagined["ref_frame"], link_name)
            
            # Get the group index of the link
            group_index = None
            for group_name, links in self._imagined_groups.items():
                if not any(link in link_name for link in links):
                    continue
                # Save the group index and the group
                group_index = list(self._imagined_groups.keys()).index(group_name)
                group = self._imagined_groups[group_name]
                break
            
            # Fix the transformation to have both visual and link origin at the group parent frame
            # (necessary since urdf is not consistent with the link tree)
            group_parent = group[0]
            while link_name != group_parent:
                index = group.index(link_name)
                
                visual_origin = self._urdf_handler.get_link_transform(group[index - 1], link_name) @ visual_origin
                link_origin = link_origin @ np.linalg.inv(self._urdf_handler.get_link_transform(group[index - 1], link_name))
                
                link_name = group[index - 1]
            
            # Create a dictionary for each mesh file
            mesh_dict[Path(stl_file).stem] = {
                'path': stl_file,
                'scale_factors': scale,
                'visual_origin': visual_origin,
                'link_origin' : link_origin,
                'group_index' : group_index
            }
        
        # Sample the point clouds from the meshes using the ImaginedPointCloudHandler
        # It creates a point cloud for each mesh and stores it from index 1 to n
        self._pc_imagine_handler.sample_from_meshes(mesh_dict, 10*self._config_imagined["num_points"])
        
        # Update the hand point cloud, which is the combination of the individual groups transformed to the palm frame
        # We here downsample the point cloud to the desired number of points (for even distribution of points)
        self._pc_imagine_handler.update_hand(self._config_imagined["num_points"])
        
    
    def update_imagination(self):
        # Go through each point cloud group and update the transform
        for index, group_links in enumerate(self._imagined_groups.values()):
            # Set frame and group index
            frame = group_links[0]
            group_index = index + 1
            
            # Get the transform to the reference frame and convert it to a transformation matrix
            transform = self._tf_handler.get_transform_matrix(frame, self._config_imagined["ref_frame"])            
            
            # Get the relative transform from the initial transform (describes relative finger movement)
            rel_transform = np.linalg.inv(self._pc_imagine_handler.initial_transforms[group_index]) @ transform
            
            # Update the transform of the group
            self._pc_imagine_handler.transforms[group_index] = self._pc_imagine_handler.initial_transforms[group_index] @ rel_transform

            # rospy.logdebug(f"URDF From {frame} to {self._config_imagined['ref_frame']}:\n {self._pc_imagine_handler.transforms[group_index]}")
            # rospy.logdebug(f"TF2 From {frame} to {self._config_imagined['ref_frame']}:\n {transform}")
            # rospy.logdebug(f"Relative transform:\n {rel_transform}")

        # Update the overall hand point cloud based on the updated group point clouds
        self._pc_imagine_handler.update_hand()