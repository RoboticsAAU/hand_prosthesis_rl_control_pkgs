import rospy
from std_msgs.msg import Bool
from gazebo_msgs.msg import ModelState, ModelStates
from move_hand.utils.ros_helper_functions import wait_for_connection


class MoveHandNode():
    def __init__(self):
        self._pose_buffer = []
        self._episode_done = False # Flag used to indicate if the episode is done
        self._name = rospy.get_param('~robot_namespace', None)
        self._r = rospy.Rate(1000)
        
        self._sub_poses = rospy.Subscriber(self._name + "/poses", ModelStates, self._poses_cb, queue_size=1000)
        self._sub_episode_done = rospy.Subscriber(self._name + "/episode_done", Bool, self._episode_done_cb, queue_size=10)
        self._pub_model_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=None)
        wait_for_connection([self._sub_poses, self._sub_episode_done, self._pub_model_state])
     
    def run(self):
        
        while not rospy.is_shutdown():
            if self._episode_done or len(self._pose_buffer) == 0:
                continue
            try:
                pose = self._pose_buffer.pop(0)
                self._pub_model_state.publish(pose)
                
                if not self._pose_buffer:
                    self._pose_buffer.append(pose)
                
            except Exception as e:
                rospy.logwarn("Failed to set position because: ", e)
            
            self._r.sleep()

    def _poses_cb(self, data : ModelStates):
        for name, pose, twist in zip(data.name, data.pose, data.twist):
            state = ModelState(model_name=name, pose=pose, twist=twist)
            self._pose_buffer.append(state)
    
    def _episode_done_cb(self, data : Bool):
        self._episode_done = data.data
        if self._episode_done:
            # TODO: Ensure that no points remain, i.e. rl does not terminate episode before node_handler had the chance to complete trajectory
            self._pose_buffer.clear()
        

if __name__ == "__main__":
    rospy.init_node("move_hand_node", log_level=rospy.INFO)
    move_hand_node = MoveHandNode()
    
    try:
        move_hand_node.run()
    except rospy.ROSInterruptException:
        pass