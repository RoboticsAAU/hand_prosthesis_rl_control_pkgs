import rospy
import numpy as np
from pathlib import Path
from gym import spaces
from gym.envs.registration import register
from functools import cached_property
from rl_env.robot_envs import mia_hand_env
from rl_env.utilities.robot_utils import generate_free_robot_hand_info

OBJECT_LIFT_LOWER_LIMIT = -0.03

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MiaHandWorld-v0',
        entry_point='task_envs.mia_hand_task_env:MiaHandWorldEnv',
        #timestep_limit=timestep_limit_per_episode,
    )

class MiaHandWorldEnv(mia_hand_env.MiaHandEnv):
    def __init__(self, config):
        """
        This Task Env is designed for having the Mia hand in the hand grasping world.
        It will learn how to move around without crashing.
        """
        
        self.config_imagined = config["config_imagined"]
        self.config_cameras = config["config_cameras"]
        self.group_frames = {}
        
        # Define the upper and lower bounds for velocity (Params are loaded in launch file)
        
        self.index_vel_lb = rospy.get_param('/mia_hand_camera/index_vel_lb')
        self.index_vel_ub = rospy.get_param('/mia_hand_camera/index_vel_ub')
        self.thumb_vel_lb = rospy.get_param('/mia_hand_camera/thumb_vel_lb')
        self.thumb_vel_ub = rospy.get_param('/mia_hand_camera/thumb_vel_ub')
        self.mrl_vel_lb = rospy.get_param('/mia_hand_camera/mrl_vel_lb')
        self.mrl_vel_ub = rospy.get_param('/mia_hand_camera/mrl_vel_ub')
        
        # Bounds for joint velocities
        self.vel_lb = np.array([self.index_vel_lb, self.thumb_vel_lb, self.mrl_vel_lb])
        self.vel_ub = np.array([self.index_vel_ub, self.thumb_vel_ub, self.mrl_vel_ub])
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
                
        # Initial velocities
        self.init_index_vel = 0
        self.init_thumb_vel = 0
        self.init_mrl_vel = 0
        
        # Define the upper and lower bounds for positions
        self.index_pos_lb = rospy.get_param('/mia_hand_camera/index_pos_lb')
        self.index_pos_ub = rospy.get_param('/mia_hand_camera/index_pos_ub')
        self.thumb_pos_lb = rospy.get_param('/mia_hand_camera/thumb_pos_lb')
        self.thumb_pos_ub = rospy.get_param('/mia_hand_camera/thumb_pos_ub')
        self.mrl_pos_lb = rospy.get_param('/mia_hand_camera/mrl_pos_lb')
        self.mrl_pos_ub = rospy.get_param('/mia_hand_camera/mrl_pos_ub')
        
        # Bounds for joint positions        
        self.pos_lb = np.array([self.index_pos_lb, self.thumb_pos_lb, self.mrl_pos_lb])
        self.pos_ub = np.array([self.index_pos_ub, self.thumb_pos_ub, self.mrl_pos_ub])
                
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self._action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self._obs_space))
        
        # Rewards
        self.end_episode_points = rospy.get_param("/mia_hand_camera/end_episode_points")

        self.cumulated_steps = 0.0
        

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MiaHandWorldEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        
        velocities = np.array([self.init_index_vel,
                                self.init_thumb_vel,
                                self.init_mrl_vel])
        
        self.move_fingers(velocities)

        return True


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
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        
        # Get the action
        action = np.array([action[0], action[1], action[2]])
        
        # Get the new joint velocities
        velocities = self.joints_vel + action
        velocities.clip(self._as_low, self._as_high)
        
        self.move_fingers(velocities)
        
        rospy.logdebug("END Set Action ==>"+str(action))


    def _get_obs(self):
        """
        Fetch observations from the Mia Hand
        :return:
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
        

    def _is_done(self):
        # Now we check if it has crashed based on the imu
        if self._object_lift > OBJECT_LIFT_LOWER_LIMIT:
            self._episode_done = True

        return self._episode_done


    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
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
        camera_obs = self.get_camera_obs()
        state_obs = self.get_state_obs()
        camera_obs.update(state_obs)
        return camera_obs
    
    def get_state_obs(self):
        raise NotImplementedError()
            
    def get_camera_obs(self):
        obs_dict = {}
        
        for cam_name, cam_config in self.config_cameras.items():
            for modality_name in cam_config.keys():
                key_name = f"{cam_name}-{modality_name}"
                
                if modality_name == 'rgb':
                    raise NotImplementedError()
                
                elif modality_name == 'depth':
                    raise NotImplementedError()
                
                elif modality_name == 'point_cloud':
                    # Remove table and enforce cardinality
                    self.point_cloud_handler.remove_plane()
                    self.point_cloud_handler.update_cardinality(512)

                    # Transform point cloud to palm frame
                    transform = self.tf_handler.get_transform_matrix(cam_name, "palm")
                    self.point_cloud_handler.transform(transform)
                
                else:
                    raise RuntimeError("Modality not supported")
                
                obs_dict[key_name] = self.point_cloud_handler.points
                
        if self.config_imagined is not None:
            self.update_imagination()
            obs_dict.update(self.imagination)
        
        return obs_dict
        
    def setup_imagination(self):
        # config has: "stl_files", "ref_frame", "groups", "num_points"
        self.group_frames = {group_index: None for group_index in range(len(self.config_imagined["groups"].keys()))}
        mesh_dict = {}  # Initialize an empty dictionary
        for stl_file in self.config_imagined["stl_files"]:
            visual_origin, scale = self.urdf_handler.get_visual_origin_and_scale(Path(stl_file).stem)
            
            link_name = self.urdf_handler.get_link_name(Path(stl_file).stem)
            link_origin = self.urdf_handler.get_link_transform(self.config_imagined["ref_frame"], link_name)
            
            group_index = None
            for group_name, links in self.config_imagined["groups"].items():
                if not any(link in link_name for link in links):
                    continue
                group_index = list(self.config_imagined["groups"].keys()).index(group_name)
            
            # Create a dictionary for each mesh file
            mesh_dict[Path(stl_file).stem] = {
                'path': stl_file,  # Construct the path for the file
                'scale_factors': scale,  # Assign scale factors
                'visual_origin': visual_origin,  # Assign origin
                'link_origin' : link_origin,
                'group_index' : group_index
            }
            
            if self.group_frames[group_index] is None:
                self.group_frames[group_index] = link_name
        
        self.pc_imagine_handler.sample_from_meshes(mesh_dict, 10*self.config_imagined["num_points"])
        self.pc_imagine_handler.update_cardinality(self.config_imagined["num_points"]//len(self.config_imagined["groups"]), voxel_size=0.001)
        
    def update_imagination(self):
        for group_index, frame in self.group_frames.items():
            # Get the transform to the reference frame
            tf_transform = self.tf_handler.get_transform(self.config_imagined["ref_frame"], frame)            
            transform = self.tf_handler.convert_tf_to_matrix(tf_transform)
            # Get the relative transform and update the transform
            rel_transform = np.linalg.inv(self.pc_imagine_handler.transforms[group_index]) @ transform
            np.set_printoptions(precision=3)
            tmp = self.config_imagined["ref_frame"]
            print("------------------------------")
            print(f"URDF From {tmp} to {frame}:\n {self.pc_imagine_handler.transforms[group_index]}")
            print(f"TF2 From {tmp} to {frame}:\n {transform}")
            print(f"Relative transform:\n{rel_transform}")
            print("------------------------------")
            
            self.pc_imagine_handler.transform(self.pc_imagine_handler.pc[group_index], rel_transform)
            
            self.pc_imagine_handler._transforms[group_index] = transform