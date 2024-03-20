import rospy
import numpy as np 
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

class TFHandler():
    def __init__(self):        
        # Initialize the TF2 buffer and listener. The buffer will store the latest transforms
        # and the listener will listen for new transforms and update the buffer.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
    
    def get_transform(self, child_frame_id : str, parent_frame_id : str = "world") -> TransformStamped:
        """
        Get the transform from the parent_frame_id to the child_frame_id.
        :param parent_frame_id: The parent frame id (default is "world")
        :param child_frame_id: The child frame id
        :return: The transform
        """
        try:
            return self._tf_buffer.lookup_transform(parent_frame_id, child_frame_id, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to get transform: %s", str(e))
            return None
    
    def transform_pose(self, input_pose, target_frame : str):
        """
        Transform the input_pose to the target_frame.
        :param input_pose: The input pose
        :param target_frame: The target frame
        :return: The transformed pose
        """
        try:
            return self._tf_buffer.transform(input_pose, target_frame, rospy.Duration(1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to transform pose: %s", str(e))
            return None
        
    def convert_tf_to_matrix(self, transform : TransformStamped):
        """
        Convert a tf transform to a numpy 4x4 transformation matrix.
        :param transform: The tf transform
        :return: The 4x4 transformation matrix
        """
        # Convert the quaternion to a numpy rotation matrix
        R = Rotation.from_quat([transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z,
                                transform.transform.rotation.w]).as_matrix()
        
        # Save the translation as a numpy array
        t = np.array([transform.transform.translation.x,
                      transform.transform.translation.y,
                      transform.transform.translation.z])
        
        # Create the 4x4 transformation matrix
        T = np.concatenate((R, t.reshape(-1,1)), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
        
        return T