from grasp_planner.src.grasp_handler import GraspitHandler


if __name__ == '__main__':
    
    config = {
        "world_file": "mia_hand_all_world",
        "search_energy": "STRICT_AUTO_GRASP_ENERGY", # See graspit interface
        "num_obj_grasps": 20,
        "obj_dir": "/objects",
        "save_dir": "/grasps",
    }
    
    graspit_handler = GraspitHandler(config, debug = True)