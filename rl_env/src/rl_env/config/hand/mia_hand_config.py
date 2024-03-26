import rospy
import yaml
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from rl_env.config.config import HandConfig
import rl_env.utilities.addons.lib_cloud_conversion_Open3D_ROS as o3d_ros

class MiaHandConfig(HandConfig):
    def __init__(self):
        
        super(MiaHandConfig, self).__init__()

        hand_config_file = self.rospack.get_path("rl_env") + "/params/hand/mia_hand_params.yaml"
        with open(hand_config_file, 'r') as file:
            hand_config = yaml.load_safe(file)
        
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.hand_name + hand_config["subscriptions"]["joint_state_topic"], JointState, self._joints_callback)
        rospy.Subscriber(self.hand_name + hand_config["subscriptions"]["camera_points_topic"], PointCloud2, self._camera_point_cloud_callback)
        
        self._thumb_vel_pub = rospy.Publisher(self.hand_name + hand_config["publications"]["thumb_velocity_topic"], Float64, queue_size=1)
        self._index_vel_pub = rospy.Publisher(self.hand_name + hand_config["publications"]["index_velocity_topic"], Float64, queue_size=1)
        self._mrl_vel_pub = rospy.Publisher(self.hand_name + hand_config["publications"]["mrl_velocity_topic"], Float64, queue_size=1)
        

    def _joints_callback(self, data : JointState):
        self.joints = data.position
        self.joints_vel = data.velocity
        self.joints_effort = data.effort
        
    def _camera_point_cloud_callback(self, data : PointCloud2):
        self.points = o3d_ros.convertCloudFromRosToOpen3d(data)
    
    
    def move_finger(self, speed : float, finger_id : str):
        """
        It will move the finger based on the speed given.
        :param speed: Speed in the positive axis of the finger
        :return:
        """
        vel_value = Float64(data=speed)
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        if finger_id == "thumb":
            self._thumb_vel_pub.publish(vel_value)
        elif finger_id == "index":
            self._index_vel_pub.publish(vel_value)
        elif finger_id == "mrl":
            self._mrl_vel_pub.publish(vel_value)
        else:
            raise ValueError("The finger_id specified is not valid")
        
    def move_fingers(self, speeds : np.float64):
        """
        It will move the fingers based on the speeds given.
        :param speed: Speeds in the positive axis of the fingers
        :return:
        """
        for index, finger_id in enumerate(["thumb", "index", "mrl"]):
            self.move_finger(speeds[index], finger_id)