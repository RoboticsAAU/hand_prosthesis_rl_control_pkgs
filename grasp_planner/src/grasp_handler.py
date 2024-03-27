from graspit_commander.graspit_commander import GraspitCommander
from geometry_msgs.msg import Pose 
from typing import Dict, Union, Optional
from pathlib import Path
from tqdm import tqdm
import glob
import os
import json

class GraspitHandler():
    def __init__(self, config : Optional[Dict[str, Union[int, str]]] = None, debug = False):
        """
        Initializes a GraspLogger object.

        Parameters:
        - config (Optional[Dict[str, Union[int, str]]]): A dictionary containing configuration options.
          It should include the following keys:
            * world_file: (str) The name of the world file to load.
            * search_energy: (int) The search energy to use for planning grasps.
            * num_obj_grasps: (int) The number of grasps to plan for each object.
            * obj_dir: (str) The directory containing all the object files.
            * save_dir: (str) The directory where the JSON file with grasp candidates is stored.
          Default is None.
        - debug (bool): If True, debug messages will be printed. Default is False.
        """
            
        if config is not None:
            # Check whether any strings are empty
            if any(value is '' for value in config.values()):
                raise ValueError("Error: empty string in config. Either do not provide config or provide a valid value")
            
            self.config_set = True
            
            try:
                self._world_file = config["world_file"]
                self._search_energy = config["search_energy"]
                self._num_obj_grasps = config["num_obj_grasps"]
                self._obj_dir = config["obj_dir"]
                self._save_dir = config["save_dir"]
            
            except KeyError as e:
                raise KeyError(f"Error occurred while processing config: {e}")
            
        self._debug = debug
        self.config_set = False
        
    
    def _store_grasp(self):
        pass
    
    
    def load_obj(self, obj_path : str, pose : Optional[Pose] = None) -> None:
        """
        Function to load an object into the environment.
        
        Parameters:
        - obj_path (str): The path to the object file.
        - pose (Optional[Pose]): The pose of the object. Default is None.
        """
        
        GraspitCommander.importObstacle(obj_path, pose)
    
    
    def _clear_world(self) -> None:
        """
        Function to clear the environment.
        """
        GraspitCommander.clearWorld()  
    
    
    def _load_world(self, file : str) -> None:
        """
        Function to load a world file into the environment.
        
        Parameters:
        - file (str): The name of the world file to load.
        """
        GraspitCommander.loadWorld(file)
    
    def _load_object(self):
        pass

    # TODO: Return type
    def plan_object_grasps(self, obj_file : str) -> grasps:
        """
        Function to plan grasps for a single object.
        """
        self._clear_world()
        self._load_world(self._world_file)
        self._load_object(obj_file)
        
        # TODO: Make following user configurable
        return self.planGrasps(max_steps=30900, search_energy="STRICT_AUTO_GRASP_ENERGY", feedback_num_steps=50, feedback_cb=self._callback)
        
    
    def plan_all_grasps(self):
        '''
        Function to plan grasps for all objects for the current set configuration.
        '''
        
        num_grasps = 0
        obj_folders = [f.path for f in os.scandir(self._obj_dir) if f.is_dir()]
        
        for obj_folder in tqdm(obj_folders):            
            xml_file = glob.glob(obj_folder + "*.xml") 
            
            if len(xml_file) != 1:
                raise ValueError(f"Error: {obj_folder} does not contain exactly one xml file.")
            else:
                xml_file = xml_file[0]
            
            self.plan_object_grasps(xml_file)
            
            
            num_grasps += len(grasps)
            
            # Store grasps in a JSON file
            self._store_grasp(grasps)
            
            # Clear the world
            self.clear_world()
        
        
    def _callback(self, data):
        self._curr_data = data
        if self._debug:
            os.system('clear')
            print(data)
            
    @property
    def curr_data(self):
        return self._curr_data

def callback(data):
    os.system('clear')
    print(data)


if __name__ == '__main__':
    
    GraspitCommander.clearWorld()
    GraspitCommander.loadWorld("mia_hand_all_world")
    grasps = GraspitCommander.planGrasps(max_steps=30900, search_energy="STRICT_AUTO_GRASP_ENERGY", feedback_num_steps=50, feedback_cb=callback)
    print(grasps.grasps)

    for i in range(20):
        print(grasps.grasps[i].pose)
        GraspitCommander.setRobotPose(grasps.grasps[i].pose)
        GraspitCommander.setRobotDesiredDOF(grasps.grasps[i].dofs,[0,0,0,0])
        input("Press Enter to continue...")
    
