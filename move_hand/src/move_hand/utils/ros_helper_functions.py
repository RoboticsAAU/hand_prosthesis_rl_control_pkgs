import rospy
from typing import Union, List

def _is_connected(pubsub: Union[rospy.Publisher, rospy.Subscriber]) -> bool:
    """ Check if the publisher or Publisher is connected to any other nodes. Returns True if any node is connected to the publisher."""
    return pubsub.get_num_connections() > 0


def wait_for_connection(pubsubs: List[Union[rospy.Publisher, rospy.Subscriber]]):
    """ Wait for the connection to the publishers and subscribers. """

    if pubsubs is None:
        rospy.logwarn("No publishers or subscribers were supplied.")
        return

    for pubsub in pubsubs:
        while not _is_connected(pubsub) and not rospy.is_shutdown():
            rospy.logdebug("No connections to {} yet so we wait and try again.".format(pubsub.name))
            try:
                rospy.sleep(0.5)
            except rospy.ROSInterruptException:
                pass

