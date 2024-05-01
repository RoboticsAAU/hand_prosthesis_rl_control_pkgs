import rospy
import yaml
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, PointCloud2
from gazebo_msgs.srv import SetModelConfiguration
from typing import Dict, List, Any

import rl_env.utils.addons.lib_cloud_conversion_Open3D_ROS as o3d_ros
from rl_env.setup.hand.hand_setup import HandSetup

class MiaHandSetup(HandSetup):
    def __init__(self, hand_config: Dict[str, Dict], joint_limits: Dict[str, Any]):
        
        super(MiaHandSetup, self).__init__()
        
        self._topic_config = hand_config['topics']
        self._general_config = hand_config['general']
        
        # Save the joint names and limits
        self._joint_names = [joint_name for joint_name in joint_limits["joint_limits"].keys()]
        self._joint_limits = joint_limits["joint_limits"]
        self._joint_velocity_limits = hand_config["general"]["joint_velocity_limits"]
        
        # hand_config_file = self.rospack.get_path("rl_env") + "/params/hand/mia_hand_params.yaml"
        # with open(hand_config_file, 'r') as file:
        #     self._config = yaml.safe_load(file)
        
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self._name + self._topic_config["subscriptions"]["joint_state_topic"], JointState, self._joints_callback)
        rospy.Subscriber(self._name + self._topic_config["subscriptions"]["camera_points_topic"], PointCloud2, self._camera_point_cloud_callback)
        
        # Finger publishers
        self._thumb_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["thumb_controller_topic"], Float64, queue_size=1)
        self._index_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["index_controller_topic"], Float64, queue_size=1)
        self._mrl_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["mrl_controller_topic"], Float64, queue_size=1)

        # Wrist publishers
        self._wrist_rot_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["wrist_rot_controller_topic"], Float64, queue_size=1)
        self._wrist_exfle_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["wrist_exfle_controller_topic"], Float64, queue_size=1)
        self._wrist_ulra_controller_pub = rospy.Publisher(self._name + self._topic_config["publications"]["wrist_ulra_controller_topic"], Float64, queue_size=1)

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
    
    def set_finger_vel(self, vel : float, finger_id : str) -> None:
        """
        It will move the finger based on the velocity given.
        :param vel: Speed in the positive axis of the finger
        :return:
        """
        vel_value = Float64(data=vel)
        rospy.logdebug("MiaHand Float64 Cmd>>" + str(vel_value))
        if finger_id == "thumb":
            self._thumb_controller_pub.publish(vel_value)
        elif finger_id == "index":
            self._index_controller_pub.publish(vel_value)
        elif finger_id == "mrl":
            self._mrl_controller_pub.publish(vel_value)
        else:
            raise ValueError("The finger_id specified is not valid")
    

    def set_wrist_vel(self, vel: float, wrist_id: str) -> None:
        """
        It will move the wrist based on the velocity given.
        :param vel: Velocity of the wrist joint specified by wrist_id
        :param wrist_id: The wrist joint to move
        """
        vel_value = Float64(data=vel)
        rospy.logdebug("Setting velocity: " + str(vel_value) + " for wrist ID: " + wrist_id)
        if wrist_id == "rot":
            self._wrist_rot_controller_pub.publish(vel_value)
        elif wrist_id == "exfle":
            self._wrist_exfle_controller_pub.publish(vel_value)
        elif wrist_id == "ulra":
            self._wrist_ulra_controller_pub.publish(vel_value)
        else:
            raise ValueError("The wrist_id {} is not valid".format(wrist_id))

    

    def set_finger_pos(self, pos : np.array) -> None:
        """
        It will move the fingers to a given position.
        :param pos: Positions in the positive axis of the fingers
        """
        # Before finger position can be set, effort must be set to zero
        self.set_action(np.zeros(6))
            
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
        
        # Wrist
        for index, wrist_id in enumerate(["rot", "exfle", "ulra"]):
            self.set_wrist_vel(vels[index], wrist_id)
        
        # Fingers
        for index, finger_id in enumerate(["index", "mrl", "thumb"], start=3):
            self.set_finger_vel(vels[index], finger_id)

    
    def _joints_callback(self, data : JointState):
        self.joints_pos = data.position
        self.joints_vel = data.velocity
        self.joints_effort = data.effort
        
    def _camera_point_cloud_callback(self, data : PointCloud2):
        self.point_cloud = o3d_ros.convertCloudFromRosToOpen3d(data)
    
    def _get_subscribers_info(self) -> List[Dict[str, Any]]:
        return [
            {"topic": self._name + self._topic_config["subscriptions"]["joint_state_topic"], "message_type": JointState},
            {"topic": self._name + self._topic_config["subscriptions"]["camera_points_topic"], "message_type": PointCloud2}
        ]
    
    def _get_publishers(self) -> List:
        return [
            self._thumb_controller_pub,
            self._index_controller_pub,
            self._mrl_controller_pub,
            self._wrist_rot_controller_pub,
            self._wrist_exfle_controller_pub,
            self._wrist_ulra_controller_pub
        ]
