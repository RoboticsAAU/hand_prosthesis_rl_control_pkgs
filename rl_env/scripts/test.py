#!/usr/bin/env python3

import rospy
import numpy as np
import rospkg
import glob
from pathlib import Path
from task_envs.mia_hand_task_env import MiaHandWorldEnv


def main():

    def create_env():
        # Get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        
        # Get the stl files from the mia description package
        ignore_files = ["1.001.stl", "UR_flange.stl"]
        stl_folder = rospack.get_path('mia_hand_description') + "/meshes/stl"
        stl_files = [file for file in glob.glob(stl_folder + "/*") if (Path(file).name not in ignore_files)]
        
        # Extract stl files for left and right hand respectively
        stl_files_left, stl_files_right = [], []
        for x in stl_files:
            (stl_files_right, stl_files_left)["mirrored" in x].append(x)
        
        config_imagined = {"stl_files" : stl_files_right,
                            "ref_frame" : "palm",
                            "num_points" : 512}
        
        config_cameras = {
            "realsense_d435": {
                "point_cloud": {
                    "ref_frame": "palm",
                    "num_points": 512,
                    "fov_x": 60,
                    "fov_y": 60,
                    "max_range": 1.5,
                    "min_range": 0.5
                }
            }
        }
        
        config = {"config_imagined" : config_imagined,
                  "config_cameras" : config_cameras}
        
        env = MiaHandWorldEnv(config)
        
        # Setup camera and imagination
        env.setup_imagination()
        
        return env
    
    mia_world_env = create_env()
    
    rospy.sleep(2)
    
    while not rospy.is_shutdown():
        current_time = rospy.get_time()
        
        # Alternating between opening and closing hand
        alternating_time = 5
        if (int(current_time) % (2*alternating_time)) < alternating_time:
            speed = 0.5
        else:
            speed = -0.5
        mia_world_env.move_fingers(np.repeat(speed, 3))
        rospy.sleep(0.1)
        #mia_world_env.pc_imagine_handler.visualize(index=0)
        mia_world_env.update_imagination()

if __name__ == "__main__":
    rospy.init_node("mia_hand_rl_env")
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass