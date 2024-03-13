#!/usr/bin/env python3

import rospy
import numpy as np
from hand_prosthesis_rl.robot_envs.mia_hand_env import MiaHandEnv

def main():
    mia_env = MiaHandEnv()
    
    rospy.sleep(5)
    
    while not rospy.is_shutdown():
        current_time = rospy.get_time()
        
        # Alternating between opening and closing hand
        alternating_time = 5
        if (int(current_time) % (2*alternating_time)) < alternating_time:
            speed = 0.5
        else:
            speed = -0.5
        mia_env.move_fingers(np.repeat(speed, 3))
        
        rospy.sleep(0.1)

if __name__ == "__main__":
    rospy.init_node("mia_hand_rl_env")
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass