# !/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion

# Utils
from utils.ros_helper_functions import _is_connected


class GazeboInterface():
    def __init__(self, hand_name: str = 'mia_hand'):
        """ hand_name: str is the name of the model in the gazebo world."""
        rospy.init_node('gazebo_interface')
        
        # Publishers
        self._pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

        # Subscribers
        self._sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, self._state_callback)

        # Set the rate of the controller.
        self._rate = rospy.Rate(1000)  # 1000hz

        # Variables
        self.hand_name = hand_name

        # Wait for the publishers and subscribers to connect before returning from the constructor.
        self.wait_for_connection()

    def wait_for_connection(self):
        """ Wait for the connection to the publishers and subscribers. """
        while not _is_connected(self._pub_state) and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _pub_state yet so we wait and try again")
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass

        while not _is_connected(self._sub_state) and not rospy.is_shutdown():
            rospy.logdebug("No publishers to _sub_state yet so we wait and try again")
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass


    def reset(self):
        # Should reset the position of the hand, and all other objects in the world.
        pass

    def step(self, action: ModelState) -> bool:
        """ Step the environment by applying the given action and return if the step is done correctly."""
        # TODO set a new position of the hand in the world.
        pass

    def set_position(self, position: Pose, reference_frame: str = 'world'):
        """ Set the position of the hand to the given position. """
        try:
            self._publish_position(position, reference_frame)
        except Exception as e:
            rospy.logwarn(e)

    def set_velocity(self, velocity: Twist):
        """ Set the velocity of the hand to the given velocity. """
        try:
            self._publish_velocity(velocity)
        except Exception as e:
            rospy.logwarn(e)

    


    # ------------------- Private methods ------------------- #
    def _publish_velocity(self, velocity: Twist):
        """ Publish the velocity of the hand to the gazebo world. Includes metadata of the controller in the message."""

        # Verify that the position is not None
        if velocity is None:
            raise ValueError("The position cannot be None")
        
        # Verify that the position is of type Pose
        if not isinstance(velocity, Twist):
            raise ValueError("The position must be of type Pose")
        
        # Final check before publishing
        if not _is_connected(self._pub_state):
            raise rospy.ROSException("The publisher is not connected")
            
        # Set the data of the message
        modelstate = ModelState(model_name=self.hand_name,
                                twist=velocity,
                                )


        # Publish the message
        self._pub_state.publish(modelstate)



    def _publish_position(self, position: Pose, reference_frame: str = 'world'):
        """ Publish the position of the hand to the gazebo world. Includes metadata of the controller in the message."""

        # Verify that the position is not None
        if position is None:
            raise ValueError("The position cannot be None")
        
        # Verify that the position is of type Pose
        if not isinstance(position, Pose):
            raise ValueError("The position must be of type Pose")
        
        # Final check before publishing
        if not _is_connected(self._pub_state):
            raise rospy.ROSException("The publisher is not connected")
            
        # Set the data of the message
        modelstate = ModelState(model_name=self.hand_name,
                                pose=position,
                                reference_frame=reference_frame)


        # Publish the message
        self._pub_state.publish(modelstate)


    def _state_callback(self, msg):
        pass


def main():
    # Test move hand controller class
    gazebointerface = GazeboInterface()
    # pose = Pose(position=Point(x=1.0, y=1.0, z=1.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0))
    vel = Twist(linear=Point(x=0.0, y=0.0, z=0.0), angular=Point(x=0.0, y=0.0, z=0.0))
    while not rospy.is_shutdown():
        # pose.position.z += 0.001
        # gazebointerface.set_position(pose)
        vel.linear.z = 1.0
        gazebointerface.set_velocity(vel)
        gazebointerface._rate.sleep()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass    