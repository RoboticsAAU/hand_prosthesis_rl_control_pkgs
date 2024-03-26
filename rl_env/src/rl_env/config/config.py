import rospy
import rospkg
from abc import ABC, abstractmethod
import numpy as np

class HandConfig(ABC):
    def __init__(self):
        self.hand_name = rospy.get_param('robot_namespace', None)
        assert self.hand_name is not None, rospy.logerr("Could not get robot namespace")
        
        self.hand_config = {}
        
        self.rospack = rospkg.RosPack()
    
    @abstractmethod
    def set_action(self, action: np.array) -> None:
        pass