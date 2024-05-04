import numpy as np
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from typing import Union
from tf.transformations import quaternion_from_euler, quaternion_multiply

def next_pose(velocity: Twist, current_position: Pose, dt: float) -> Pose:
    """ Calculate the next position of the model given the current position, velocity and delta time. """
    # Initialize a new pose
    new_pose = Pose()

    # Propagate the position using velocity and delta time
    new_pose.position.x = current_position.position.x + velocity.linear.x * dt
    new_pose.position.y = current_position.position.y + velocity.linear.y * dt
    new_pose.position.z = current_position.position.z + velocity.linear.z * dt

    # # Compute new orientation using the angular velocity
    rotation = quaternion_from_euler(velocity.angular.x * dt, velocity.angular.y * dt, velocity.angular.z * dt)
    orientation_np = quaternion_multiply(rotation, np.array([current_position.orientation.x, current_position.orientation.y, current_position.orientation.z, current_position.orientation.w]))

    new_pose.orientation.x = orientation_np[0]
    new_pose.orientation.y = orientation_np[1]
    new_pose.orientation.z = orientation_np[2]
    new_pose.orientation.w = orientation_np[3]


    # Return the new pose   
    return new_pose


def convert_pose(pose : Union[np.array, Pose]) -> Pose:
    """
    Function to convert a np.array pose to a geometry_msg.msg.Pose.
    """
    
    # Verify that the position is of type Pose
    if not isinstance(pose, Pose) and not isinstance(pose, np.ndarray):
        raise ValueError(f"The position must be of type Pose or numpy.ndarray. Input given is of type {type(pose)}.")

    # If the pose is a numpy array, convert it to a Pose message
    if isinstance(pose, np.ndarray):
        if len(pose) != 7:
            raise ValueError(f"The pose must have define both position and orientation as a 7 element array. Pose is {pose}")
        return Pose(position=Point(x=pose[0], y=pose[1], z=pose[2]), orientation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))
    else:
        return pose


def convert_velocity(velocity : Union[np.array, Twist]) -> Twist:
    """
    Function to convert a np.array velocity to a geometry_msg.msg.Twist.
    """
    
    # Verify that the position is of type Velocity
    if not isinstance(velocity, Twist) and not isinstance(velocity, np.ndarray):
        raise ValueError(f"The velocity must be of type Twist or numpy.ndarray. Input given is of type {type(velocity)}.")

    # If the velocity is a numpy array, convert it to a Twist message
    if isinstance(velocity, np.ndarray):
        if len(velocity) != 6:
            raise ValueError("The velocity must have 6 elements, [x, y, z, roll, pitch, yaw]")
        return Twist(linear=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]), angular=Vector3(x=velocity[3], y=velocity[4], z=velocity[5]))
    else:
        return velocity


def convert_state(name : str, pose : Union[np.array, Pose], vel : Union[np.array, Twist]) -> ModelState:
    """
    Function to convert a np.array pose to a geometry_msg.msg.Pose.
    """
    return ModelState(model_name=name, pose=convert_pose(pose), twist=convert_velocity(vel))
    