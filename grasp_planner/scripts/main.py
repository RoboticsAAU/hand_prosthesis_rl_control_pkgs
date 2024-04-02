from grasp_planner.grasp_handler import GraspHandler
from geometry_msgs.msg import Pose, Quaternion

if __name__ == '__main__':
    
    config = {
        "world_file": "mia_hand_world",
        "search_energy": "GUIDED_POTENTIAL_QUALITY_ENERGY", # See graspit interface
        "num_obj_grasps": 20,
        "obj_dir": "/Development/graspit/models/objects/graspit_shapenet",
        "save_dir": "/grasps",
        "num_steps": 100000,
    }
    
    graspit_handler = GraspHandler(config, debug = True)
    graspit_handler.plan_all_grasps()
    
    