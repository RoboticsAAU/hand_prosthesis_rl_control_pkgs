import rospy
import rospkg
from abc import ABC, abstractmethod

class HandConfig(ABC):
    def __init__(self) -> None:
        self.hand_name = rospy.get_param('robot_namespace', None)
        assert self.hand_name is not None, rospy.logerr("Could not get robot namespace")
        
        self.rospack = rospkg.RosPack()
        

        # Controller types
        # COntroller list
        # Robot name space