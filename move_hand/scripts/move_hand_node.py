import rospy
import numpy as np
from gazebo_msgs.msg import ModelState, ModelStates
from move_hand.utils.ros_helper_functions import wait_for_connection


class MoveHandNode():
    def __init__(self):
        self._pose_buffer = []
        self._name = rospy.get_param('~robot_namespace', None)
        self._r = rospy.Rate(1000)
        
        self._sub_poses = rospy.Subscriber(self._name + "/poses", ModelStates, self._poses_callback, queue_size=1000)
        self._pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1000)
        self._sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, self._states_callback, queue_size=1)
        wait_for_connection([self._sub_poses, self._pub_state])
        
        self._hand_exists = False
        
    def run(self):
        
        while not rospy.is_shutdown():
            if not self._hand_exists or len(self._pose_buffer) == 0:
                continue
            
            try:
                pose = self._pose_buffer.pop(0)
                self._pub_state.publish(pose)
                
                if not self._pose_buffer:
                        self._pose_buffer.append(pose)
            except Exception as e:
                rospy.logwarn("Failed to set position because: ", e)
            
            
            self._r.sleep()

    def _poses_callback(self, data : ModelStates):
        for name, pose, twist in zip(data.name, data.pose, data.twist):
            state = ModelState(model_name=name, pose=pose, twist=twist)
            self._pose_buffer.append(state)
    
    def _states_callback(self, data : ModelStates) -> bool:
        self._hand_exists = any([name == self._name for name in data.name])

if __name__ == "__main__":
    rospy.init_node("move_hand_node", log_level=rospy.INFO)
    move_hand_node = MoveHandNode()
    
    try:
        move_hand_node.run()
    except rospy.ROSInterruptException:
        pass