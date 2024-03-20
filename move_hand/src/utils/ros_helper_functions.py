import rospy

def _is_connected(publisher: rospy.Publisher) -> bool:
    """ Check if the publisher is connected to any other nodes. Returns True if any node is connected to the publisher."""
    return publisher.get_num_connections() > 0

def _is_connected(subscriber: rospy.Subscriber) -> bool:
    """ Check if the subscriber is connected to any other nodes. Returns True if any node is connected to the subscriber."""
    return subscriber.get_num_connections() > 0
