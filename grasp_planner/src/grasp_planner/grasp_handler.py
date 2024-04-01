from graspit_commander.graspit_commander import GraspitCommander
from geometry_msgs.msg import Pose 
from typing import Dict, Union, Optional
from pathlib import Path
from tqdm import tqdm
from graspit_interface.msg import SearchSpace
import glob
import time
import os
import json

class GraspHandler():
    def __init__(self, config : Dict[str, Union[int, str]], debug = False):
        """
        Initializes a GraspLogger object.

        Parameters:
        - config (Optional[Dict[str, Union[int, str]]]): A dictionary containing configuration options.
          It should include the following keys:
            * robot_file: (str) The path to the xml robot file.
            * obj_dir: (str) The directory that contains all objects. 
            * search_energy: (int) The search energy to use for planning grasps.
            * num_obj_grasps: (int) The number of grasps to plan for each object.
            * num_steps: (int) The number of steps used by planner for each grasp. Default is 50.
            * save_dir: (str) The directory where the JSON file with grasp candidates is stored.
          Default is None.
        - debug (bool): If True, debug messages will be printed. Default is False.
        """
        
        self.expected_keys = ["robot_file", "obj_dir", "save_dir", "search_energy", "num_obj_grasps", "num_steps"]
        
        # Check if the configuration dictionary is valid
        self._config = config
        self._check_config()
            
        self._debug = debug
    
    
    def _check_config(self) -> None:
        """
        Function to check the configuration dictionary.
        
        Parameters:
        - config (Dict[str, Union[int, str]]): The configuration dictionary.
        
        """
        
        if any(value == '' or value is None for value in self.config.values()):
            raise ValueError("Error: empty string in config. Either do not provide config or provide a valid value")
        
        elif set(self.config.keys()) != set(self.expected_keys):
            raise ValueError("Error: invalid keys in configuration dictionary.")
    
    
    def _store_grasps(self):
        pass


    def _plan_object_grasps(self, feedback_num_steps : int = 50):
        """
        Function to plan grasps for a single object.
        """
        
        grasps = GraspitCommander.planGrasps(max_steps=self._config["num_steps"], search_energy=self.config["search_energy"], feedback_num_steps=feedback_num_steps, feedback_cb=self._callback)
        
        return grasps
        
    
    def plan_all_grasps(self, hand_pose : Optional[Pose] = Pose(), obj_pose : Optional[Pose] = Pose()):
        '''
        Function to plan grasps for all objects for the current set configuration. Assumes objects and world files are are in same folder and have same name.
        
        Parameters:
        - obj_pose (Optional[Pose]): The pose of the object. Default is zero pose.
        
        '''
        
        self._check_config()
        num_grasps = 0
        
        # Get recursively the path of all object files
        obj_files = glob.glob(self.config["obj_dir"] + "/*/*.xml", recursive=True)         
        rel_obj_files = []
        
        # Because graspit expects relative paths
        for obj_file in obj_files:
            # Split the full path into parts
            path_parts = obj_file.split(os.sep)

            # Find the index of the known folder
            folder_index = path_parts.index(Path(self.config["obj_dir"]).name)

            # Construct the relative path from the known folder
            relative_path = os.path.join(*path_parts[folder_index:])
            
            rel_obj_files.append(relative_path)
        
        for rel_obj_file in tqdm(rel_obj_files):
            GraspitCommander.importRobot(self.config["robot_file"], hand_pose)
            GraspitCommander.importGraspableBody(os.path.splitext(rel_obj_file)[0], obj_pose)
            grasps = self._plan_object_grasps(feedback_num_steps=-1)
            num_grasps += len(grasps.grasps)
            
            # Store grasps in a JSON file
            # self._store_grasps(grasps)
            
            time.sleep(1)                    
        
        
    def _callback(self, data):
        self._curr_data = data
        if self._debug:
            os.system('clear')
            print(data)    
    
    
    @property
    def config(self):
        return self._config

    
    @config.setter
    def config(self, config : Dict[str, Union[int, str]]) -> None:
        """
        Function to set configuration dictionary.
        
        Parameters:
        - config (Dict[str, Union[int, str]]): The configuration dictionary.
        """
        
        self._config = config
    
    
    @property
    def curr_data(self):
        return self._curr_data


def callback(data):
    os.system('clear')
    print(data)


if __name__ == '__main__':
    
    GraspitCommander.clearWorld()
    GraspitCommander.loadWorld("mia_hand_world")
    GraspitCommander.importGraspableBody("graspit_shapenet/category_1/obj_1")
    # grasps = GraspitCommander.planGrasps(max_steps=30900, search_energy="STRICT_AUTO_GRASP_ENERGY", feedback_num_steps=50, feedback_cb=callback)
    # search_space = SearchSpace(1)
    
    grasps = GraspitCommander.planGrasps(max_steps=100000, search_energy="GUIDED_POTENTIAL_QUALITY_ENERGY", feedback_num_steps=-1, feedback_cb=callback)
    print(grasps.grasps)

    for i in range(20):
        print(grasps.grasps[i].pose)
        GraspitCommander.setRobotPose(grasps.grasps[i].pose)
        GraspitCommander.setRobotDesiredDOF(grasps.grasps[i].dofs,[0,0,0,0])
        input("Press Enter to continue...")
    
