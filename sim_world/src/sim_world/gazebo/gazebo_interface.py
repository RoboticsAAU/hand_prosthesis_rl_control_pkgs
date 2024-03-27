# !/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import numpy as np
from typing import Union, Type

# Utils
from move_hand.utils.ros_helper_functions import _is_connected, wait_for_connection
from move_hand.utils.movement import next_pose
from rl_env.setup.hand.hand_setup import HandSetup


class GazeboInterface():
    def __init__(self, hand_setup : Type[HandSetup]):
        """ hand_name: str is the name of the model in the gazebo world."""
        
        # Save the hand_config object
        self.hand_setup = hand_setup
        
        # Publishers
        self._pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

        # Subscribers
        self._sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, self._state_callback, buff_size=1000)

        # Set the rate of the controller.
        self.hz = 1000
        self._rate = rospy.Rate(self.hz)  # 1000hz


        # Wait for the publishers and subscribers to connect before returning from the constructor. Supply them in a list.
        wait_for_connection(
            [self._pub_state,
             self._sub_state]
        )

        # Initialize the current state of the hand
        self.current_state = None
        # Wait for the first state to be received before returning from the constructor.
        while self.current_state is None:
            rospy.logdebug("No state received yet so we wait and try again")
            try:
                self._rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass


    
    def reset_to(self, pose: Union[Pose, np.ndarray], model_name: None):
        """ Reset a gazebo_model to the given pose. If no model_name is passed, the hand is used."""

        if model_name is None:
            model_name = self.hand_setup.name

        # TODO: Use the model_name to reset the model to the given pose.
        self.set_pose(pose)

    def step(self, action: ModelState) -> bool:
        """ Step the environment by applying the given action and return if the step is done correctly."""
        # TODO set a new position of the hand in the world.
        pass

    # Set Position
    def set_pose(self, pose: Union[Pose, np.ndarray], reference_frame: str = 'world'):
        """ Set the pose of the hand to the given position and orientation. Orientation is given in quaternions. """
        try:
            self._publish_pose(pose, reference_frame)
        except Exception as e:
            rospy.logwarn("Failed to set position because: ", e)


    # Set Velocity
    def set_velocity(self, velocity: Union[Twist, np.ndarray]):
        """ Set the velocity of the hand to the given velocity. The velocity can be of type Twist or numpy.ndarray with 6 dimensions."""
        try:
            self._publish_velocity(velocity)
        except Exception as e:
            rospy.logwarn("Failed to set velocity because: ", e)


    # ------------------- Private methods ------------------- #
    def _publish_velocity(self, velocity: Union[Twist, np.ndarray]):
        """Publish the velocity of the hand to the gazebo world. Includes metadata of the controller in the message."""
        
        if velocity is None:
            raise ValueError("The velocity cannot be None")
        
        if not _is_connected(self._pub_state):
            raise rospy.ROSException("The publisher is not connected")

        if isinstance(velocity, np.ndarray):
            if len(velocity) != 6:
                raise ValueError("The velocity must have 6 elements, [x, y, z, roll, pitch, yaw]")
            twist_velocity = Twist()
            twist_velocity.linear.x = velocity[0]
            twist_velocity.linear.y = velocity[1]
            twist_velocity.linear.z = velocity[2]
            twist_velocity.angular.x = velocity[3]
            twist_velocity.angular.y = velocity[4]
            twist_velocity.angular.z = velocity[5]
            velocity = twist_velocity

        elif not isinstance(velocity, Twist):
            raise ValueError("The velocity must be of type Twist or numpy.ndarray")

        # Initialize a new ModelState instance.
        modelstate = ModelState()

        # Set the metadata of the message
        modelstate.model_name = self.hand_setup.name
        modelstate.reference_frame = 'world'

        # Forward propagate the position of the hand in time and set the next position based on the velocity and delta time.
        modelstate.pose = next_pose(velocity, self.current_state.pose, 1.0 / float(self.hz))
        
        # Publish the message
        self._pub_state.publish(modelstate)
    

    

    def _publish_pose(self, pose, reference_frame: str = 'world'):
        """ Publish the position of the hand to the gazebo world. Includes metadata of the controller in the message. Overwrites the current position of the hand in the simulation."""

        # Verify that the position is not None
        if pose is None:
            raise ValueError("The position cannot be None")

        # Final check before publishing
        if not _is_connected(self._pub_state):
            raise rospy.ROSException("The publisher is not connected")
            
        
        # Verify that the position is of type Pose
        if not isinstance(pose, Pose) or not isinstance(pose, np.ndarray):
            raise ValueError("The position must be of type Pose or numpy.ndarray")

        if isinstance(pose, np.ndarray):
            if len(pose) != 7:
                raise ValueError("The pose must have define both position and orientation as a 7 element array")
            # Convert the numpy array to a Pose message
            pose = Pose(position=Point(x=pose[0], y=pose[1], z=pose[2]), orientation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))

        # Set the data of the message
        modelstate = ModelState(model_name=self.hand_setup.name,
                                pose=pose,
                                reference_frame=reference_frame)

        # Publish the message
        self._pub_state.publish(modelstate)


    def _state_callback(self, msg):
        """ Callback method for the state subscriber. """
        rospy.logdebug("Received state message")
        
        # Initialize a new ModelState instance.
        self.current_state = ModelState()
        
        # Get the state of the hand
        for i in range(len(msg.name)):
            if msg.name[i] == self.hand_setup.name:
                self.current_state.pose = msg.pose[i]
                return

        rospy.logwarn("The gazebo model: '", self.hand_setup.name, "', was not found in the list of list of model states in gazebo. Check the name of the desired model in the gazebo world")



if __name__ == '__main__':
    print("{} should not be run as a script...".format(__file__))