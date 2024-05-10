import rospy
import yaml
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, PointCloud2
from gazebo_msgs.srv import SetModelConfiguration
from typing import Dict, List, Any, Union

import rl_env.utils.addons.lib_cloud_conversion_Open3D_ROS as o3d_ros
from rl_env.setup.hand.hand_setup import HandSetup

class MiaHandSetup(HandSetup):
    def __init__(self, hand_config: Dict[str, Dict], joint_limits: Dict[str, Any]):
        
        super(MiaHandSetup, self).__init__()
        
        self._topic_config = hand_config['topics']
        self._general_config = hand_config['general']
        
        # Save the joint names and limits
        self._joint_velocity_limits = hand_config["general"]["joint_velocity_limits"]
        self._joint_names = [joint_name for joint_name in joint_limits["joint_limits"].keys() if joint_name in list(self._joint_velocity_limits.keys())]
        self._joint_limits = {joint_name : joint_limit for joint_name, joint_limit in joint_limits["joint_limits"].items()
                              if joint_name in list(self._joint_velocity_limits.keys())}
        
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self._name + self._topic_config["subscriptions"]["joint_state_topic"], JointState, self._joints_callback)
        rospy.Subscriber(self._name + self._topic_config["subscriptions"]["camera_points_topic"], PointCloud2, self._camera_point_cloud_callback)
        
        # Store publishers in a list
        self._publishers = [rospy.Publisher(self._name + topic_name, Float64, queue_size=1) for topic_name in self._topic_config["publications"].values() 
                            if any(joint_name in topic_name for joint_name in self._joint_velocity_limits.keys())]

        self._set_configuration_srv = rospy.ServiceProxy("/gazebo/set_model_configuration", SetModelConfiguration)

        # Initialise the subscribed data variables
        self.joints_pos = [Float64()]
        self.joints_vel = [Float64()]
        self.joints_effort = [Float64()]
        self.point_cloud = PointCloud2()
        
        # Specifying the rotation of the hand frame "palm" w.r.t. the frame orientation assumed for starting position in move_hand_controller
        # z-axis points along the hand (points along middle finger) and x-axis points out of the palm (for both right and left hand). 
        if self._general_config["right_hand"] == True:
            self._hand_rotation = np.array([[0, 0, 1],
                                           [1, 0, 0],
                                           [0, 1, 0]], dtype=np.float32)
        else:
            self._hand_rotation = np.array([[0, 0, 1],
                                           [-1, 0, 0],
                                           [0, -1, 0]], dtype=np.float32)
        
    def get_subscriber_data(self) -> Dict[str, Any]:
        """
        Get all the subscriber data and return it in a dictionary.
        """
        subscriber_data = {
            "hand_data": {
                "joints_pos": self.joints_pos,
                "joints_vel": self.joints_vel,
                "joints_effort": self.joints_effort,
                "point_cloud": self.point_cloud,
                "contacts": self._get_hand_contact()
            }
        }
        return subscriber_data
    
    def set_vel(self, vel : float, hand_id : str) -> None:
        """
        It will move the finger/wrist based on the velocity given.
        :param vel: Speed in the positive axis of the hand
        :return:
        """
        vel_value = Float64(data=vel)
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        publisher = self._get_publisher(hand_id)
        
        if publisher is None:
            raise Exception("Publisher for given hand_id does not exist")
        
        publisher.publish(vel_value)


    def set_finger_pos(self, pos : np.array) -> None:
        """
        It will move the fingers to a given position.
        :param pos: Positions in the positive axis of the fingers
        """
                   
        self._set_configuration_srv(
            model_name=self._name,
            urdf_param_name="mia_hand_description",
            joint_names=self._joint_names,
            joint_positions=pos
        )
    

    def set_action(self, vels : np.array) -> None:
        """
        It will move the fingers and wrist joints based on the velocities given.
        :param vels: Velocities in the positive axis of the fingers, and wrist velocities in both directions
        """
        # TODO: Make this general (instead of specific to velocity) to work with effort also
        
        # Fingers
        hand_ids = ["index", "mrl", "thumb", "rot", "exfle", "ulra"]
        for index, vel in enumerate(vels):
            self.set_vel(vel, hand_ids[index])
    
    def _joints_callback(self, data : JointState):
        self.joints_pos = data.position
        self.joints_vel = data.velocity
        self.joints_effort = data.effort
        
    def _camera_point_cloud_callback(self, data : PointCloud2):
        # Code to compute the delay of the pointcloud. 
        # callback_time = rospy.Time.now().to_sec()
        # message_stamp = data.header.stamp.to_sec()
        # difference = callback_time - message_stamp
        # rospy.logwarn("Camera point cloud callback timestamp: {}, and post pipeline message header timestamp: {} and difference: {}".format(callback_time, message_stamp, difference ))
    
        self.point_cloud = o3d_ros.convertCloudFromRosToOpen3d(data)
    
    def _get_subscribers_info(self) -> List[Dict[str, Any]]:
        return [
            {"topic": self._name + self._topic_config["subscriptions"]["joint_state_topic"], "message_type": JointState},
            {"topic": self._name + self._topic_config["subscriptions"]["camera_points_topic"], "message_type": PointCloud2}
        ]
    
    def _get_publishers(self) -> List:        
        return self._publishers
    
    def _get_publisher(self, topic_subname : str) -> Union[rospy.Publisher, None]:
        publisher = None
        for pub in self._publishers:
            if topic_subname in pub.name:
                if publisher is not None: raise ValueError("Multiple publishers found for the same topic")
                publisher = pub
        
        return publisher
        
        