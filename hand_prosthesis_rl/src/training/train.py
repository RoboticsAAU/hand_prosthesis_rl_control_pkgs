#!/usr/bin/env python3

import rospy
from robot_envs.mia_hand_env import MiaHandEnv


def main():
    mia_env = MiaHandEnv()
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node("mia_hand_rl_env")
    
    try:
        main()
        
    except rospy.ROSInterruptException:
        pass