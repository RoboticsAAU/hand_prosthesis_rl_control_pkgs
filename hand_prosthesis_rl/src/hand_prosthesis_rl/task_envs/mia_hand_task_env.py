import rospy
import numpy as np
from gym import spaces
from hand_prosthesis_rl.robot_envs import mia_hand_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from functools import cached_property

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='MiaHandWorld-v0',
        entry_point='task_envs.mia_hand_task_env:MiaHandWorldEnv',
        timestep_limit=timestep_limit_per_episode,
    )

class MiaHandWorldEnv(mia_hand_env.MiaHandEnv):
    def __init__(self):
        """
        This Task Env is designed for having the Mia hand in the hand grasping world.
        It will learn how to move around without crashing.
        """
        
        # Define the upper and lower bounds for velocity (Params are loaded in launch file)
        self.index_vel_lb = rospy.get_param('/mia_hand/index_vel_lb')
        self.index_vel_ub = rospy.get_param('/mia_hand/index_vel_ub')
        self.thumb_vel_lb = rospy.get_param('/mia_hand/thumb_vel_lb')
        self.thumb_vel_ub = rospy.get_param('/mia_hand/thumb_vel_ub')
        self.mrl_vel_lb = rospy.get_param('/mia_hand/mrl_vel_lb')
        self.mrl_vel_ub = rospy.get_param('/mia_hand/mrl_vel_ub')
        
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
        self.index_pos_lb = rospy.get_param('/mia_hand/index_pos_lb')
        self.index_pos_ub = rospy.get_param('/mia_hand/index_pos_ub')
        self.thumb_pos_lb = rospy.get_param('/mia_hand/thumb_pos_lb')
        self.thumb_pos_ub = rospy.get_param('/mia_hand/thumb_pos_ub')
        self.mrl_pos_lb = rospy.get_param('/mia_hand/mrl_pos_lb')
        self.mrl_pos_ub = rospy.get_param('/mia_hand/mrl_pos_ub')
        
        # Bounds for joint positions        
        self.pos_lb = np.array([self.index_pos_lb, self.thumb_pos_lb, self.mrl_pos_lb])
        self.pos_ub = np.array([self.index_pos_ub, self.thumb_pos_ub, self.mrl_pos_ub])
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self._action_space()))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self._obs_space()))
        
        # Rewards
        self.end_episode_points = rospy.get_param("/mia_hand/end_episode_points")

        self.cumulated_steps = 0.0

        self.camera_infos = {
            "wrist_cam": {
                "point_cloud": {
                    "num_points": 512,
                    "fov_x": 60,
                    "fov_y": 60,
                    "max_range": 1.5,
                    "min_range": 0.5
                }
            }
        }
        
        self.use_imagined = True

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
        
        # Current and previous actions are stored before sending it to the parent class MiaHandEnv
        index_vel = action[0]       
        self.last_index_vel = index_vel
        
        thumb_vel = action[1]
        self.last_thumb_vel = thumb_vel
        
        mrl_vel = action[2]
        self.last_mrl_vel = mrl_vel
        
        #index_vel, thumb_vel, mrl_vel = action[0], action[1], action[2]
        
        # Mia hand is set to execute the speeds
        velocities = np.array([index_vel, thumb_vel, mrl_vel]) 
        
        self.move_fingers(velocities)
        
        rospy.logdebug("END Set Action ==>"+str(action))


    def _get_obs(self):
        """
        Fetch observations from the Mia Hand
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations


    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("Mia hand is Too Close to wall==>")
        else:
            rospy.logwarn("Mia hand is NOT close to a wall ==>")
            
        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("Mia hand Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        else:
            rospy.logerr("DIDNT crash Mia hand ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        

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
        
        for cam_name, cam_config in self.camera_infos.items():
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
            
        if self.use_imagined:
    
        