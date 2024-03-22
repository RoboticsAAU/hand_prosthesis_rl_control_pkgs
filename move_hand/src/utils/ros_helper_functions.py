import rospy
import numpy as np
from geometry_msgs.msg import Twist
from functools import wraps

def _is_connected(publisher: rospy.Publisher) -> bool:
    """ Check if the publisher is connected to any other nodes. Returns True if any node is connected to the publisher."""
    return publisher.get_num_connections() > 0

def _is_connected(subscriber: rospy.Subscriber) -> bool:
    """ Check if the subscriber is connected to any other nodes. Returns True if any node is connected to the subscriber."""
    return subscriber.get_num_connections() > 0



def velocity_decorator(func):
    @wraps(func)
    def wrapper(self, velocity):
        if velocity is None:
            raise ValueError("The velocity cannot be None")
        
        if isinstance(velocity, np.ndarray):
            if len(velocity) != 6:
                raise ValueError("The velocity must have 6 elements")
        elif not isinstance(velocity, Twist):
            raise ValueError("The velocity must be of type Twist or numpy.ndarray[float]")
        
        if not _is_connected(self._pub_state):
            raise rospy.ROSException("The publisher is not connected")

        if isinstance(velocity, np.ndarray):
            twist_velocity = Twist()
            twist_velocity.linear.x = velocity[0]
            twist_velocity.linear.y = velocity[1]
            twist_velocity.linear.z = velocity[2]
            twist_velocity.angular.x = velocity[3]
            twist_velocity.angular.y = velocity[4]
            twist_velocity.angular.z = velocity[5]
            return func(self, twist_velocity)
        else: 
            return func(self, velocity)
            
    return wrapper



