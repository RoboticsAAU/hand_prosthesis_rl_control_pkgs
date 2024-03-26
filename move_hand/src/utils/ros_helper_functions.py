import rospy
from typing import Union

def _is_connected(publisher: Union[rospy.Publisher, rospy.Subscriber]) -> bool:
    """ Check if the publisher or Publisher is connected to any other nodes. Returns True if any node is connected to the publisher."""
    return publisher.get_num_connections() > 0




