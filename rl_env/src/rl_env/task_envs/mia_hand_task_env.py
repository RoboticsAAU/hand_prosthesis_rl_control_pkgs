import rospy
import numpy as np
import rospkg
import open3d as o3d
import gymnasium as gym
import glob
from time import time, sleep
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
from pathlib import Path
from functools import cached_property
from typing import Dict, List, Any, Optional
from geometry_msgs.msg import Pose
from contact_republisher.msg import contacts_msg

from sim_world.rl_interface.rl_interface import RLInterface
from rl_env.utils.tf_handler import TFHandler
from rl_env.utils.point_cloud_handler import PointCloudHandler, ImaginedPointCloudHandler
from rl_env.utils.urdf_handler import URDFHandler


class MiaHandWorldEnv(gym.Env):
    def __init__(self, rl_interface : RLInterface, rl_config : Dict[str, Any]):
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
        self._rl_interface = rl_interface
        
        # Get the configurations for the cameras and the imagined point clouds
        self._config_imagined = rl_config["visual_sensors"]["config_imagined"]
        self._config_cameras = rl_config["visual_sensors"]["config_cameras"]
        
        self._rl_config = rl_config
        self.force_config = {
            "index_fle": {
                # "range" is the accepted force angle about y-axis of each finger frame. 
                "range": tuple(np.deg2rad([-70, 70])),
                # "rotation" is the SO3 rotation from desired finger frame (x-axis pointing out from dorsal) to actual finger frame
                "rotation": np.eye(3)
            },
            "middle_fle": {
                "range": tuple(np.deg2rad([-70, 70])),
                "rotation": np.eye(3)
            },
            "ring_fle": {
                "range": tuple(np.deg2rad([-70, 70])),
                "rotation": np.eye(3)
            },
            "little_fle": {
                "range": tuple(np.deg2rad([-70, 70])),
                "rotation": np.eye(3)
            },
            "thumb_fle": {
                "range": tuple(np.deg2rad([-70, 70])),
                "rotation": np.eye(3)
            },
            "palm": {
                "range": tuple(np.deg2rad([-90, 90])),
                "rotation": np.array([[0,0,-1],
                                      [0,-1,0],
                                      [-1,0,0]], dtype=np.float64)
            }
        }
        self._imagined_groups = {}
        
        # Joint position bounds in observation space
        self._obs_pos_lb = np.array([limits["min_position"] for limits in self._rl_interface._world_interface.hand._joint_limits.values()])
        self._obs_pos_ub = np.array([limits["max_position"] for limits in self._rl_interface._world_interface.hand._joint_limits.values()])

        # Velocity bounds in observation space
        self._obs_vel_lb = np.array([limit[0] for limit in self._rl_interface._world_interface.hand._joint_velocity_limits.values()])
        self._obs_vel_ub = np.array([limit[1] for limit in self._rl_interface._world_interface.hand._joint_velocity_limits.values()])
        
        # Bounds for joint velocities in action space.
        # Notice that the thumb input control is 1-dimensional that maps to 2DoF (thumb_fle and thumb_opp), and we assume range for thumb_fle corresponds to control range
        action_space_joints = ["j_index_fle", "j_mrl_fle", "j_thumb_fle", "j_wrist_rotation", "j_wrist_exfle", "j_wrist_ulra"]
        self._act_vel_lb = np.array([limit[0] for name, limit in self._rl_interface._world_interface.hand._joint_velocity_limits.items() if name in action_space_joints])
        self._act_vel_ub = np.array([limit[1] for name, limit in self._rl_interface._world_interface.hand._joint_velocity_limits.items() if name in action_space_joints])
        
        # state bounds
        self._state_lb = np.concatenate((self._obs_pos_lb, self._obs_vel_lb))
        self._state_ub = np.concatenate((self._obs_pos_ub, self._obs_vel_ub))

        # Save number of joints
        self._dof = len(self._rl_interface._world_interface.hand._joint_velocity_limits)
        
        # Timestamp used to wait at end pose
        self._prev_pose_ts = time()
        
        # Parameters for the state and observation space
        self._joints = None
        self._joints_vel = None
        self._pc_cam_handler.pc.append(o3d.geometry.PointCloud())
        self._object_pose = Pose()
        self._contacts = []
                
        self.setup_imagination(stl_ignores=["1.001.stl", "UR_flange.stl"])
        
        # Print the spaces
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # TODO: Define these in setup
        self._end_episode_points = 0.0
        self._cumulated_steps = 0
        self._episode_count = 0
        self._finger_object_dist = np.zeros(3)
    
    
    def step(self, action):
        
        done = False
        if self._rl_interface._hand_controller.buffer_empty:
            done = (time() - self._prev_pose_ts) > self._rl_config["general"]["dur_at_obj"]
        else:
            self._prev_pose_ts = time()
        
        self._rl_interface.step(action)
        self.update()
        obs = self._get_obs()
        info = {}
        reward = self._compute_reward(obs)
        self._cumulated_steps += 1

            
        self._rl_interface._pub_episode_done.publish(done)

        # TODO: Remove following, as it is only for debugging
        contact_check = self.check_contact(self._contacts)
        if any(contact_check.values()):        
            rospy.logwarn("Is palmar: " + str(contact_check))     
        
        return obs, reward, done, False, info
    
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._episode_count += 1
        
        rospy.logdebug("Resetting MiaHandWorldEnv")
        self._init_env_variables()
        if self._joints is not None and np.any(np.logical_or(self._joints < self._obs_pos_lb - 0.1, self._obs_pos_ub + 0.1 < self._joints)):
            rospy.logwarn(f"Joint positions out of bounds:\n {self._joints}")
            rospy.logwarn("Resetting hand model")   
            self._reset_hand()
        self._rl_interface.update_context()
        # Set finger position as average of joint limits
        average_pos = [sum(limits)/2.0 for limits in zip(self._obs_pos_lb, self._obs_pos_ub)]
        self._rl_interface._world_interface.hand.set_finger_pos(average_pos)
        self.update()
        obs = self._get_obs()
        info = {}
        rospy.logdebug("END Resetting MiaHandWorldEnv")
        
        return obs, info
    
    
    def _reset_hand(self):
        """Resets the hand completely. This is used when the hand becomes unstable due to exceeded joint bounds.
        """

        rospy.loginfo("RESET HAND START")
        self._rl_interface._world_interface._controllers_connection.reset_controllers()
        self._rl_interface._world_interface.check_system_ready()
        self._rl_interface._world_interface.respawn_hand(self._rl_interface.default_pose)
        self._rl_interface._world_interface._controllers_connection.reset_controllers()
        self._rl_interface._world_interface.check_system_ready()
        rospy.loginfo("RESET HAND END\n")
    
    
    def update(self):
        """
        Update the values of the hand data (state and observation)
        """
        # Update the hand data
        self._joints = self._rl_interface.rl_data["hand_data"]["joints_pos"]
        self._joints_vel = self._rl_interface.rl_data["hand_data"]["joints_vel"]
        self._pc_cam_handler.pc[0] = self._rl_interface.rl_data["hand_data"]["point_cloud"]
        self._object_pose = self._rl_interface.rl_data["obj_data"]
        self._contacts = self._rl_interface.rl_data["hand_data"]["contacts"]
        
    
    # Methods needed by the TrainingEnvironment
    def _init_env_variables(self):
        """
        Inits variables needs to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set episode_done to false, because it's calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This method will set the velocity of the hand based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        raise NotImplementedError() 


    def _get_state_obs(self):
        """
        Fetch observations from the Mia Hand
        :return: observation
        """

        observation = {"state" : np.concatenate((self._joints, self._joints_vel))}
        
        rospy.logdebug("Observations: "+str(observation))
        return observation


    def _compute_reward(self, observation):
        """
        Compute the reward for the given rl step
        :return: reward
        """
        
        reward = 0
        
        # Obtain the combined joint velocity
        combined_joint_vel = np.sum(np.abs(observation["state"][self._dof:]))
        
        # Check if at least three fingers are in contact with object
        hand_contacts = self.check_contact(self._contacts)
        fingers_in_contact = list(hand_contacts.values()).count(True)
        
        
        #TODO: Penalise hitting ground plane?
        
        # Soft reward for contact (requires at least one finger in contact)
        reward += 0.1 * max(0, fingers_in_contact*(1 - (self._episode_count/200))) 
        
        # Strict reward for contact (requires thumb and at least one other finger)
        if hand_contacts["thumb_fle"] and fingers_in_contact >= 2:
            # Reward for contact
            reward += 0.5 * fingers_in_contact

        # Reward for effort expenditure
        reward += -0.05*combined_joint_vel/sum(self._act_vel_ub) * min(1, max(0, 0.01*(self._episode_count - 100)))
        
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        
        return reward


    @cached_property
    def action_space(self):
        action_space = gym.spaces.Box(self._act_vel_lb, self._act_vel_ub, dtype = np.float32)
        rospy.logwarn("ACTION SPACE===> "+str(action_space))
        return action_space


    @cached_property
    def observation_space(self):
        state_space = gym.spaces.Box(self._state_lb, self._state_ub, dtype = np.float32)
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
            obs_dict["imagined"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((self._config_imagined["num_points"],) + (3,)))
            
        rospy.logwarn(f"OBSERVATION SPACE: {obs_dict}")
        return gym.spaces.Dict(obs_dict)
    
    def _get_obs(self):
        all_obs = self._get_camera_obs()
        state_obs = self._get_state_obs()
        all_obs.update(state_obs)
        return all_obs

        
    def _get_camera_obs(self):
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
                    # TODO: Decide whether to include table or not
                    # Remove table and enforce cardinality
                    self._pc_cam_handler.pc[0].paint_uniform_color(np.array([0.4745, 0.8353, 0.9922]))
                    self._pc_cam_handler.remove_plane()
                    self._pc_cam_handler.update_cardinality(modality_config["num_points"])

                    # Transform point cloud to reference frame
                    transform = self._tf_handler.get_transform_matrix(modality_config["optical_frame"], modality_config["ref_frame"])
                    self._pc_cam_handler.pc[0] = self._pc_cam_handler.transform(self._pc_cam_handler.pc[0], transform)
                    obs_dict[key_name] = self._pc_cam_handler.pc[0].points
                    
                else:
                    raise RuntimeError("Modality not supported")
        
        # Get the observations from the imagination
        if self._config_imagined is not None:
            self.update_imagination()
            obs_dict["imagined"] = self._pc_imagine_handler.points[0]
        
        return obs_dict
        
    def setup_imagination(self, stl_ignores : Optional[List[str]] = None):
        # Get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        
        # Get the stl files from the mia description package
        stl_folder = rospack.get_path(self._config_imagined["stl_package"]) + "/meshes/stl"
        stl_files = [file for file in glob.glob(stl_folder + "/*") if (Path(file).name not in stl_ignores)]
        
        # Extract stl files for the correct hand
        filtered_stl_files = [file for file in stl_files if (("mirrored" in file) != self._rl_interface._world_interface.hand._general_config["right_hand"])]
        
        # config has: "stl_files", "ref_frame", "groups", "num_points"
        # Define the imagined groups where each group corresponds to a movable joint in the hand (along with base palm)
        free_joints = self._urdf_handler.get_free_joints()
        self._imagined_groups = {free_joint : self._urdf_handler.get_free_joint_group(free_joint) for free_joint in free_joints}
        self._imagined_groups["j_" + self._config_imagined["ref_frame"]] = self._urdf_handler.get_link_group(self._config_imagined["ref_frame"])
        
        mesh_dict = {}  # Initialize an empty dictionary
        for stl_file in filtered_stl_files:
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
        self._pc_imagine_handler.update_hand(self._config_imagined["num_points"])   
    
    def check_contact(self, contacts : contacts_msg) -> Dict[str, bool]:
        """ 
        Checks whether there is a valid contact between the hand and the object.
        param contacts: The contacts message.
        return: Dictionary with the contact status for each finger.
        """
        
        # Container of contact checks
        hand_contact = {link_name : False for link_name in self.force_config.keys()}
        
        for contact in contacts.contacts:
            
            # Early continue if the contact is with the ground plane or if force vector is zero (spurious contact)
            if ("ground_plane" in (contact.collision_1 + contact.collision_2) or
                np.all(np.isclose(contact.forces_1, 0, atol=1e-6)) or
                np.all(np.isclose(contact.forces_2, 0, atol=1e-6))):
                continue
            
            # Get name of link in contact and the corresponding force. 
            if self._rl_interface._world_interface.hand.name in contact.collision_1:
                link_name = contact.collision_1.split("::")[1]
                finger_force = np.array(contact.forces_1, dtype=np.float64)
            else:
                link_name = contact.collision_2.split("::")[1]
                finger_force = np.array(contact.forces_2, dtype=np.float64)
            
            # Early continue if collision object is not of relevance 
            if link_name not in list(self.force_config.keys()):
                continue
            
            # Rotate force vector if the frame does not match the assumed frame 
            finger_force = self.force_config[link_name]["rotation"] @ finger_force
            
            # Compute angle of force vector w.r.t. x-axis in the xz-plane
            # Note that x is not taken to be negative, since the finger force points inside the hand/finger
            y_rot = np.arctan2(finger_force[2], finger_force[0])
            
            # Check if the force vector is pointing out of the hand and within the bounds (indicating palmar contact)
            lower_bound, upper_bound = self.force_config[link_name]["range"]
            hand_contact[link_name] = (lower_bound < y_rot) and (y_rot < upper_bound) 
            
        return hand_contact
        
            
