# !/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import numpy as np
from typing import Union, Type

# Utils
from move_hand.utils.ros_helper_functions import _is_connected, wait_for_connection
from move_hand.utils.movement import next_pose
from sim_world.world_interfaces.world_interface import WorldInterface

class SimulationInterface(WorldInterface):
    def __init__(self):
        """ hand_name: str is the name of the model in the gazebo world."""
        # Initialize the parent class
        super(SimulationInterface, self).__init__()

        # Model state publisher and subscriber
        self._pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self._sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, self._state_callback, buff_size=1000)

        # Set the rate of the controller.
        self.hz = 1000
        self._rate = rospy.Rate(self.hz)  # 1000hz
        
        # Initialize variables for saving state information
        self.model_states = ModelStates()

        # Wait for the publishers and subscribers to connect before returning from the constructor. Supply them in a list.
        wait_for_connection([self._pub_state, self._sub_state])
    
    def get_subscriber_data(self):
        """ Get all the subscriber data and return it in a dictionary. """
        subscriber_data = self.hand.get_subscriber_data()
        subscriber_data.update({"hand_pose": self.hand_pose})
        
        return subscriber_data
    
    def set_pose(self, model_name : str, pose: Union[Pose, np.ndarray], reference_frame: str = 'world'):
        """ Set the pose of the hand to the given position and orientation. Orientation is given in quaternions. """
        try:
            self._publish_pose(model_name, pose, reference_frame)
        except Exception as e:
            rospy.logwarn("Failed to set position because: ", e)

    def set_velocity(self, model_name : str, velocity: Union[Twist, np.ndarray]):
        """ Set the velocity of the hand to the given velocity. The velocity can be of type Twist or numpy.ndarray with 6 dimensions."""
        try:
            self._publish_velocity(model_name, velocity)
        except Exception as e:
            rospy.logwarn("Failed to set velocity because: ", e)

    def spawn_object(self, object: str, pose: Union[Pose, np.ndarray]):
        """ Spawn the object in the gazebo world. """
        raise NotImplementedError("Not implemented yet")


    # ------------------- Private methods ------------------- #
    def _publish_velocity(self, model_name : str, velocity: Union[Twist, np.ndarray], ref_frame: str = 'world'):
        """Publish a velocity to the given model in the gazebo world. Includes metadata of the controller in the message."""
        # If velocity is a numpy array, convert it to a Twist message.
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
        model_state = ModelState()
        
        # Set the metadata of the message
        model_state.model_name = model_name
        model_state.reference_frame = ref_frame

        # Forward propagate the position of the hand in time and set the next position based on the velocity and delta time.
        model_state.pose = next_pose(velocity, self.hand_pose, 1.0 / float(self.hz))
        
        # Publish the message
        self._pub_state.publish(model_state)

    def _publish_pose(self, model_name : str, pose : Union[Pose, np.ndarray], ref_frame: str = 'world'):
        """ Publish the position of the hand to the gazebo world. Includes metadata of the controller in the message. Overwrites the current position of the hand in the simulation."""            
        # Verify that the position is of type Pose
        if not isinstance(pose, Pose) or not isinstance(pose, np.ndarray):
            raise ValueError("The position must be of type Pose or numpy.ndarray")

        if isinstance(pose, np.ndarray):
            if len(pose) != 7:
                raise ValueError("The pose must have define both position and orientation as a 7 element array")
            # Convert the numpy array to a Pose message
            pose = Pose(position=Point(x=pose[0], y=pose[1], z=pose[2]), orientation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))

        # Set the data of the message
        model_state = ModelState(model_name=model_name,
                                pose=pose,
                                reference_frame=ref_frame)

        # Publish the message
        self._pub_state.publish(model_state)

    def _state_callback(self, msg : ModelStates):
        """ Callback method for the state subscriber. """
        rospy.logdebug("Received state message")
        self.model_states = msg

    
    @property
    def hand_pose(self) -> Pose:
        """ Return the current state of the hand. """
        try:
            index = self.model_states.name.index(self.hand.name)
            return self.model_states.pose[index]
        except ValueError:
            rospy.logwarn("The gazebo model: '", self.hand.name, "', was not found in the list of list of model states in gazebo")
            return None


if __name__ == '__main__':
    print("{} should not be run as a script...".format(__file__))