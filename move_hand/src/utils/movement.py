from geometry_msgs.msg import Twist, Pose
from tf.transformations import quaternion_from_euler, quaternion_multiply
import numpy as np

def next_position(velocity: Twist, current_position: Pose, dt: float) -> Pose:
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


def sample_position_in_sphere(radius: float, inner_radius: float = 0.0) -> np.ndarray[float]:
    # Sample a random position within a sphere of radius r and subtracting a sphere of inner radius r
    phi = np.random.uniform(0,2*np.pi)
    costheta = np.random.uniform(-1,1)
    u = np.random.uniform(inner_radius/radius,1)
    
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)  

    return np.array([x, y, z])





'''
import numpy as np

def next_position(velocity: np.array, current_position: np.array, dt: float) -> np.array:
    """ calculate the next position of the hand given the current position and the velocity. """
    pass

'''
