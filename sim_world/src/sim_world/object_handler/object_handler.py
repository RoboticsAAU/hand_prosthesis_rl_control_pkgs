import rospy
import numpy.random as random
import xml.etree.ElementTree as ET
from stl import mesh
import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any
import rospkg
from tqdm import tqdm

class ObjectHandler():
    def __init__(self, object_config : Dict[str, Any]):
        self._config = object_config
        self._objects = self._load_objects()
        self._curr_obj = None
        self._obj_idx = 0
    
    
    def _load_objects(self) -> Dict[str, str]:
        """
        Function to load objects as a dict, where key is the object name (e.g. category_1/obj_1) and value is the .sdf as a string.
        """
        
        objects = {}
        
        # Get objects path
        rospack = rospkg.RosPack()
        object_dataset_path = rospack.get_path("assets") + "/" + self._config["object_dataset"]
        if not Path(object_dataset_path).is_dir():
            raise ValueError("The object path is not a valid directory: "+str(object_dataset_path))
        
        # Load the objects into the objects variable
        for category_folder in tqdm(glob.glob(object_dataset_path + "/*"), desc="Loading objects"):
            for object_folder in tqdm(glob.glob(category_folder + "/*"), leave=False):
                # Load sdf string
                path_sdf = glob.glob(object_folder + '/mesh_new.sdf')[0]
                tree = ET.parse(path_sdf)
                xml_string = ET.tostring(tree.getroot(), encoding='utf8', method='xml').decode('utf-8')
                
                # Load mesh object
                path_mesh = glob.glob(object_folder + '/mesh.stl')[0]
                cuboid = mesh.Mesh.from_file(path_mesh, remove_empty_areas=True)
                # TODO: Hardcoded scale. Should be read from the sdf file.
                cuboid.vectors *= 0.15
                
                objects[Path(category_folder).name + "/" + Path(object_folder).name] = {
                    "sdf": xml_string,
                    "mesh": cuboid
                }
        
        self._config["num_objects"] = self.config["num_objects"] if self.config["num_objects"] != -1 else len(objects)
    
        if len(objects) < self._config["num_objects"]:
            raise ValueError("The number of objects is less than the number of objects requested.")
        
        random_keys = random.choice(list(objects.keys()), size=self._config["num_objects"], replace=False)
        rospy.loginfo("Loaded objects: " + str(random_keys))
        return OrderedDict((key, objects[key]) for key in random_keys)
    
    
    def update_current_object(self) -> None:
        try:
            if self._config["selection_mode"] == "random":
                self._curr_obj = random.choice(list(self._objects.keys()))
                self._obj_idx = list(self._objects.keys()).index(self._curr_obj)
                
            elif self._config["selection_mode"] == "sequential":
                self._curr_obj = list(self.objects.keys())[self._obj_idx]
                self._obj_idx = (self._obj_idx + 1) % self._config["num_objects"]
            
            else:
                raise ValueError("The object handler selection_mode is not valid.")
            
        except (ValueError, IndexError) as e:
            rospy.logwarn("Could not update current object: ", e)
    
    
    @property
    def curr_obj(self):
        return self._curr_obj
    
    
    @property
    def objects(self):
        return self._objects
    
    @property
    def config(self):
        return self._config