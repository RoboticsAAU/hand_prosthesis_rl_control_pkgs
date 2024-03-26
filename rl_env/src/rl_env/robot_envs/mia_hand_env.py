import numpy
import rospy
import rospkg
import open3d as o3d
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
#from mia_hand_msgs.msg import FingersData
#from mia_hand_msgs.msg import FingersStrainGauges
from rl_env.gazebo.robot_gazebo_env import RobotGazeboEnv
import rl_env.utils.addons.lib_cloud_conversion_Open3D_ROS as o3d_ros
from rl_env.utils.tf_handler import TFHandler
from rl_env.utils.point_cloud_handler import PointCloudHandler, ImaginedPointCloudHandler
from rl_env.utils.urdf_handler import URDFHandler

class MiaHandEnv(RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        rospy.loginfo("Start MiaHanEnv INIT...")
                
        # Internal Vars
        self.controllers_list = []
        self.robot_name_space = ""
        # Get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        
        # Initialise handlers
        urdf_path = rospack.get_path("sim_world") + "/urdf/hands/mia_hand_default.urdf"
        self.urdf_handler = URDFHandler(urdf_path)
        self.pc_cam_handler = PointCloudHandler(point_clouds=[o3d.geometry.PointCloud()])
        self.pc_imagine_handler = ImaginedPointCloudHandler()
        self.tf_handler = TFHandler()
        
        # We launch the init function of the Parent Class RobotGazeboEnv
        super(MiaHandEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False,
                                                start_init_physics_parameters=False,
                                                reset_world_or_sim="WORLD")
        
        self.gazebo.unpauseSim()

        self._check_all_sensors_ready()
        
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/mia_hand_camera/joint_states", JointState, self._joints_callback)
        rospy.Subscriber("/mia_hand_camera/camera/depth_registered/points", PointCloud2, self._camera_point_cloud_callback)
        
        self._thumb_vel_pub = rospy.Publisher('/mia_hand_camera/j_thumb_fle_velocity_controller/command', Float64, queue_size=1)
        self._index_vel_pub = rospy.Publisher('/mia_hand_camera/j_index_fle_velocity_controller/command', Float64, queue_size=1)
        self._mrl_vel_pub = rospy.Publisher('/mia_hand_camera/j_mrl_fle_velocity_controller/command', Float64, queue_size=1)
        
        self._wait_for_publishers_connection()
        
        self.gazebo.pauseSim()
        
        rospy.loginfo("Finished MiaHanEnv INIT...")
    

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True
    
    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        
        self.hand_joints_data = None
        while self.hand_joints_data is None and not rospy.is_shutdown():
            try:
                self.hand_joints_data = rospy.wait_for_message("/mia_hand_camera/joint_states", JointState, timeout=1.0)
                #rospy.loginfo("Current mia_hand_camera/joint_states READY=>" + str(self.hand_joints_data))

            except:
                rospy.logerr("Current mia_hand_camera/joint_states not ready yet, retrying for getting joint_states")
        
        self.camera_pc_data = None
        while self.camera_pc_data is None and not rospy.is_shutdown():
            try:
                self.camera_pc_data = rospy.wait_for_message("/mia_hand_camera/camera/depth_registered/points", PointCloud2, timeout=1.0)
                #rospy.loginfo("Current /mia_hand_camera/camera/depth_registered/points READY=>" + str(self.camera_pc_data))

            except:
                rospy.logerr("Current /mia_hand_camera/camera/depth_registered/points not ready yet, retrying for getting joint_states")

        rospy.loginfo("ALL SENSORS READY")
    
    def _wait_for_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._thumb_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _thumb_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_thumb_vel_pub Publisher Connected")
        while self._mrl_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _mrl_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_mrl_vel_pub Publisher Connected")
        while self._index_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _index_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_index_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    def _joints_callback(self, data : JointState):
        self.joints = data.position
        self.joints_vel = data.velocity
        self.joints_effort = data.effort
        
    def _camera_point_cloud_callback(self, data : PointCloud2):
        self.pc_cam_handler.pc[0] = o3d_ros.convertCloudFromRosToOpen3d(data)
        
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    
    def move_finger(self, speed : float, finger_id : str):
        """
        It will move the finger based on the speed given.
        :param speed: Speed in the positive axis of the finger
        :return:
        """
        vel_value = Float64()
        vel_value.data = speed
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        self._wait_for_publishers_connection()
        if finger_id == "thumb":
            self._thumb_vel_pub.publish(vel_value)
        elif finger_id == "index":
            self._index_vel_pub.publish(vel_value)
        elif finger_id == "mrl":
            self._mrl_vel_pub.publish(vel_value)
        else:
            raise ValueError("The finger_id specified is not valid")
        
    def move_fingers(self, speeds : numpy.float64):
        """
        It will move the fingers based on the speeds given.
        :param speed: Speeds in the positive axis of the fingers
        :return:
        """
        vel_value = Float64()
        
        self._wait_for_publishers_connection()
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        
        vel_value.data = speeds[0]
        self._thumb_vel_pub.publish(vel_value)
        
        vel_value.data = speeds[1]
        self._index_vel_pub.publish(vel_value)
        
        vel_value.data = speeds[2]
        self._mrl_vel_pub.publish(vel_value)