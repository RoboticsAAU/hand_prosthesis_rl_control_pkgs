import rospy
import numpy
from gym import spaces
from hand_prosthesis_rl.robot_envs import mia_hand_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3

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
        
        as_low = numpy.array([self.index_vel_lb,
                            self.thumb_vel_lb,
                            self.mrl_vel_lb])
        
        as_high = numpy.array([self.index_vel_ub, 
                            self.thumb_vel_ub,
                            self.mrl_vel_ub])
        
        self.action_space = spaces.Box(as_low, as_high, dtype = numpy.float32)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
                
        # Initial velocities
        self.init_index_vel = 0
        self.init_thumb_vel = 0
        self.init_mrl_vel = 0
        
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        # laser_scan = self._check_laser_scan_ready()
        # num_laser_readings = len(laser_scan.ranges)/self.new_ranges
        # high = numpy.full((num_laser_readings), self.max_laser_value)
        # low = numpy.full((num_laser_readings), self.min_laser_value)
        
        # Define the upper and lower bounds for positions
        self.index_pos_lb = rospy.get_param('/mia_hand/index_pos_lb')
        self.index_pos_ub = rospy.get_param('/mia_hand/index_pos_ub')
        self.thumb_pos_lb = rospy.get_param('/mia_hand/thumb_pos_lb')
        self.thumb_pos_ub = rospy.get_param('/mia_hand/thumb_pos_ub')
        self.mrl_pos_lb = rospy.get_param('/mia_hand/mrl_pos_lb')
        self.mrl_pos_ub = rospy.get_param('/mia_hand/mrl_pos_ub')
        
        os_low = numpy.array([self.index_pos_lb,
                            self.thumb_pos_lb,
                            self.mrl_pos_lb])
        
        os_high = numpy.array([self.index_pos_ub, 
                            self.thumb_pos_ub,
                            self.mrl_pos_ub])
        
        # We only use two integers
        self.observation_space = spaces.Box(os_low, os_high, dtype = numpy.float32)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.end_episode_points = rospy.get_param("/mia_hand/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MiaHandWorldEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        
        velocities = numpy.array([self.init_index_vel,
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
        velocities = numpy.array([index_vel, thumb_vel, mrl_vel]) 
        
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


    # Internal TaskEnv Methods
    
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))
                    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    

        return discretized_ranges
        
        
    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been 
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude