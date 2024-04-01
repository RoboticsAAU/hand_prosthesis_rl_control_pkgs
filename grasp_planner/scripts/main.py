from grasp_planner.grasp_handler import GraspHandler
from geometry_msgs.msg import Pose

if __name__ == '__main__':
    
    config = {
        "robot_file": "mia_hand",
        "search_energy": "STRICT_AUTO_GRASP_ENERGY", # See graspit interface
        "num_obj_grasps": 20,
        "obj_dir": "/Development/graspit/models/objects/graspit_shapenet",
        "save_dir": "/grasps",
        "num_steps": 70000,
    }
    
    graspit_handler = GraspHandler(config, debug = True)
    
    hand_pose = Pose()
    hand_pose.position.x = 100.0
    hand_pose.position.y = 100.0
    hand_pose.position.z = 100.0
    
    graspit_handler.plan_all_grasps(hand_pose=hand_pose)
    
    