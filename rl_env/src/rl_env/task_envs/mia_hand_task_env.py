import rospy
import numpy as np
from pathlib import Path
from gym import spaces
from gym.envs.registration import register
from functools import cached_property
from rl_env.robot_envs.mia_hand_env import MiaHandEnv
from sim_world.world_interfaces.gazebo_interface import GazeboInterface

OBJECT_LIFT_LOWER_LIMIT = -0.03

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MiaHandWorld-v0',
        entry_point='task_envs.mia_hand_task_env:MiaHandWorldEnv',
        #timestep_limit=timestep_limit_per_episode,
    )

class MiaHandWorldEnv(MiaHandEnv):
    def __init__(self, gazebo_interface : GazeboInterface):
        """
        This Task Env is designed for having the Mia hand in the hand grasping world.
        It will learn how to move around without crashing.
        """
        self._gazebo_interface = gazebo_interface
        
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MiaHandWorldEnv, self).__init__()
        
        # Get the robot name
        robot_namespace = self._gazebo_interface.hand_setup.name
        rospy.logdebug("ROBOT NAMESPACE==>"+str(robot_namespace))
        
        # Get the configurations for the cameras and the imagined point cloud
        self.config_imagined = self._gazebo_interface.hand_setup.config["config_imagined"]
        self.config_cameras = self._gazebo_interface.hand_setup.config["config_cameras"]
        self.imagined_groups = {}
        
        # Bounds for joint positions
        joint_limits = self._gazebo_interface.hand_setup.config["joint_limits"]
        self.pos_lb = np.array([joint_limits[finger + "_pos_range"][0] for finger in ["thumb", "index", "mrl"]])
        self.pos_ub = np.array([joint_limits[finger + "_pos_range"][1] for finger in ["thumb", "index", "mrl"]])
        
        # Bounds for joint velocities
        vel_limits = self._gazebo_interface.hand_setup.config["velocity_limits"]
        self.vel_lb = np.array([vel_limits[finger + "_vel_range"][0] for finger in ["thumb", "index", "mrl"]])
        self.vel_ub = np.array([vel_limits[finger + "_vel_range"][1] for finger in ["thumb", "index", "mrl"]])
        
        # We set the reward range (not compulsory)
        self.reward_range = (-np.inf, np.inf)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self._action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self._obs_space))
        
        # TODO: Define these in setup
        self.end_episode_points = 0.0
        self.cumulated_steps = 0.0
        self._object_lift = 0.0
        self._finger_object_dist = np.zeros(3)
    

    # def step(self, action):
    #     self.rl_step(action)
    #     self.update_cached_state()
    #     self.update_imagination(reset_goal=False)
    #     obs = self.get_observation()
    #     reward = self.get_reward(action)
    #     done = self.is_done()
    #     info = self.get_info()

    #     if self.current_step >= self.horizon:
    #         info["TimeLimit.truncated"] = not done
    #         done = True
    #     return obs, reward, done, info

    
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
        velocities = self.joints_vel + action
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
            "joints" : self.joints,
            "joints_vel" : self.joints_vel,
            "points" : self.point_cloud_handler.points
        }
        
        rospy.logdebug("Observations==>"+str(observation))
        rospy.logdebug("END Get Observation ==>")
        return observation
        

    def is_done(self):
        # Now we check if it has crashed based on the imu
        if self._object_lift > OBJECT_LIFT_LOWER_LIMIT:
            self._episode_done = True

        return self._episode_done


    def compute_reward(self):
        """
        Compute the reward for the given rl step
        :return: reward
        """
        # Check if episode is done
        if self.is_done():
            return self.end_episode_points
        
        # Obtain the shortest distance between finger and object
        finger_object_dist = np.min(self._finger_object_dist)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        
        # Obtain the combined joint velocity
        clipped_vel = np.clip(self.joints_vel, self.vel_lb, self.vel_ub)
        combined_joint_vel = np.sum(np.abs(clipped_vel))
        
        # Check if at least three fingers are in contact with object
        fingers_in_contact = np.sum(self._finger_object_dist < 0.03)
        
        # Reward for energy expenditure (based on distance to object)
        reward = -(finger_object_dist * combined_joint_vel) * 0.01
        if fingers_in_contact >= 2:
            # Reward for contact
            reward += 0.5 * fingers_in_contact
            
            # Reward for lifting object
            lift = np.clip(self._object_lift, 0, 0.2)
            reward += 10 * lift

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        
        return reward


    @cached_property
    def _action_space(self):
        return spaces.Box(self.vel_lb, self.vel_ub, dtype = np.float32)


    @cached_property
    def _obs_space(self):
        state_space = spaces.Box(self.pos_lb, self.pos_ub, dtype = np.float32)
        obs_dict = {"state": state_space}
        
        for cam_name, cam_config in self.config_cameras.items():
            for modality_name in cam_config.keys():
                key_name = f"{cam_name}-{modality_name}" 
                
                if modality_name == 'rgb':
                    resolution = cam_config[modality_name]["resolution"]
                    spec = spaces.Box(low=0, high=1, shape=resolution + (3,))
                
                elif modality_name == 'depth':
                    max_depth = cam_config[modality_name]["max_depth"]
                    resolution = cam_config[modality_name]["resolution"]
                    spec = spaces.Box(low=0, high=max_depth, shape=resolution + (1,))
                
                elif modality_name == 'point_cloud':
                    spec = spaces.Box(low=-np.inf, high=np.inf, shape=((cam_config[modality_name]["num_points"],) + (3,)))
                    
                else:
                    raise RuntimeError("Modality not supported")              
            
            obs_dict[key_name] = spec
            
        if self.config_imagined is not None:
            self.update_imagination()
            obs_dict["imagination"] = spaces.Box(low=-np.inf, high=np.inf, shape=((self.config_imagined["num_points"],) + (3,)))
            
        return spaces.Dict(obs_dict)
    
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
        for cam_name, cam_config in self.config_cameras.items():
            for modality_name, modality_config in cam_config.items():
                key_name = f"{cam_name}-{modality_name}"
                
                if modality_name == 'rgb':
                    raise NotImplementedError()
                
                elif modality_name == 'depth':
                    raise NotImplementedError()
                
                elif modality_name == 'point_cloud':
                    # Remove table and enforce cardinality
                    self.pc_cam_handler.remove_plane()
                    self.pc_cam_handler.update_cardinality(modality_config["num_points"])

                    # Transform point cloud to reference frame
                    transform = self.tf_handler.get_transform_matrix(cam_name, modality_config["ref_frame"])
                    self.pc_cam_handler.transform(transform)
                    obs_dict[key_name] = self.pc_cam_handler.points[0]
                    
                else:
                    raise RuntimeError("Modality not supported")
        
        # Get the observations from the imagination
        if self.config_imagined is not None:
            self.update_imagination()
            obs_dict["imagined"] = self.pc_imagine_handler.points[0]
        
        return obs_dict
        
    def setup_imagination(self):
        # config has: "stl_files", "ref_frame", "groups", "num_points"
        # Define the imagined groups where each group corresponds to a movable joint in the hand (along with base palm)
        free_joints = self.urdf_handler.get_free_joints()
        self.imagined_groups = {free_joint : self.urdf_handler.get_free_joint_group(free_joint) for free_joint in free_joints}
        self.imagined_groups["j_" + self.config_imagined["ref_frame"]] = self.urdf_handler.get_link_group(self.config_imagined["ref_frame"])
        
        mesh_dict = {}  # Initialize an empty dictionary
        for stl_file in self.config_imagined["stl_files"]:
            # Get the origin and scale for the mesh
            visual_origin, scale = self.urdf_handler.get_visual_origin_and_scale(Path(stl_file).stem)
            
            # Get the link name assoicated with the mesh and obtain the origin (transformation) from the reference
            link_name = self.urdf_handler.get_link_name(Path(stl_file).stem)
            link_origin = self.urdf_handler.get_link_transform(self.config_imagined["ref_frame"], link_name)
            
            # Get the group index of the link
            group_index = None
            for group_name, links in self.imagined_groups.items():
                if not any(link in link_name for link in links):
                    continue
                # Save the group index and the group
                group_index = list(self.imagined_groups.keys()).index(group_name)
                group = self.imagined_groups[group_name]
                break
            
            # Fix the transformation to have both visual and link origin at the group parent frame
            # (necessary since urdf is not consistent with the link tree)
            group_parent = group[0]
            while link_name != group_parent:
                index = group.index(link_name)
                
                visual_origin = self.urdf_handler.get_link_transform(group[index - 1], link_name) @ visual_origin
                link_origin = link_origin @ np.linalg.inv(self.urdf_handler.get_link_transform(group[index - 1], link_name))
                
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
        self.pc_imagine_handler.sample_from_meshes(mesh_dict, 10*self.config_imagined["num_points"])
        
        # Update the hand point cloud, which is the combination of the individual groups transformed to the palm frame
        # We here downsample the point cloud to the desired number of points (for even distribution of points)
        self.pc_imagine_handler.update_hand(self.config_imagined["num_points"])
        
    
    def update_imagination(self):
        # Go through each point cloud group and update the transform
        for index, group_links in enumerate(self.imagined_groups.values()):
            # Set frame and group index
            frame = group_links[0]
            group_index = index + 1
            
            # Get the transform to the reference frame and convert it to a transformation matrix
            transform = self.tf_handler.get_transform_matrix(frame, self.config_imagined["ref_frame"])            
            
            # Get the relative transform from the initial transform (describes relative finger movement)
            rel_transform = np.linalg.inv(self.pc_imagine_handler.initial_transforms[group_index]) @ transform
            
            # Update the transform of the group
            self.pc_imagine_handler.transforms[group_index] = self.pc_imagine_handler.initial_transforms[group_index] @ rel_transform

            # rospy.logdebug(f"URDF From {frame} to {self.config_imagined['ref_frame']}:\n {self.pc_imagine_handler.transforms[group_index]}")
            # rospy.logdebug(f"TF2 From {frame} to {self.config_imagined['ref_frame']}:\n {transform}")
            # rospy.logdebug(f"Relative transform:\n {rel_transform}")

        # Update the overall hand point cloud based on the updated group point clouds
        self.pc_imagine_handler.update_hand()