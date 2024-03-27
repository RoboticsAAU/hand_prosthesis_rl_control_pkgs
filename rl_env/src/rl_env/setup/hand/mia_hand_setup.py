import rospy
import yaml
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from rl_env.setup.hand.hand_setup import HandSetup
import rl_env.utils.addons.lib_cloud_conversion_Open3D_ROS as o3d_ros
from typing import Dict, List, Any

class MiaHandSetup(HandSetup):
    def __init__(self):
        
        super(MiaHandSetup, self).__init__()
        
        hand_config_file = self.rospack.get_path("rl_env") + "/params/hand/mia_hand_params.yaml"
        with open(hand_config_file, 'r') as file:
            self._config = yaml.safe_load(file)
        
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self._name + self._config["subscriptions"]["joint_state_topic"], JointState, self._joints_callback)
        rospy.Subscriber(self._name + self._config["subscriptions"]["camera_points_topic"], PointCloud2, self._camera_point_cloud_callback)
        
        self._thumb_vel_pub = rospy.Publisher(self._name + self._config["publications"]["thumb_velocity_topic"], Float64, queue_size=1)
        self._index_vel_pub = rospy.Publisher(self._name + self._config["publications"]["index_velocity_topic"], Float64, queue_size=1)
        self._mrl_vel_pub = rospy.Publisher(self._name + self._config["publications"]["mrl_velocity_topic"], Float64, queue_size=1)
        
    
    def _joints_callback(self, data : JointState):
        self.joints = data.position
        self.joints_vel = data.velocity
        self.joints_effort = data.effort
        
    def _camera_point_cloud_callback(self, data : PointCloud2):
        self.points = o3d_ros.convertCloudFromRosToOpen3d(data)
    
    def _get_subscribers_info(self) -> List[Dict[str, Any]]:
        return [
            {"topic": self._name + self._config["subscriptions"]["joint_state_topic"], "message_type": JointState},
            {"topic": self._name + self._config["subscriptions"]["camera_points_topic"], "message_type": PointCloud2}
        ]
    
    def _get_publishers(self) -> dict:
        return [
            self._thumb_vel_pub,
            self._index_vel_pub,
            self._mrl_vel_pub
        ]
    
    def set_finger_vel(self, vel : float, finger_id : str) -> None:
        """
        It will move the finger based on the velocity given.
        :param vel: Speed in the positive axis of the finger
        :return:
        """
        vel_value = Float64(data=vel)
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        if finger_id == "thumb":
            self._thumb_vel_pub.publish(vel_value)
        elif finger_id == "index":
            self._index_vel_pub.publish(vel_value)
        elif finger_id == "mrl":
            self._mrl_vel_pub.publish(vel_value)
        else:
            raise ValueError("The finger_id specified is not valid")
    
    def set_action(self, vels : np.array) -> None:
        """
        It will move the fingers based on the velocities given.
        :param vels: Velocities in the positive axis of the fingers
        :return:
        """
        for index, finger_id in enumerate(["thumb", "index", "mrl"]):
            self.set_finger_vel(vels[index], finger_id)