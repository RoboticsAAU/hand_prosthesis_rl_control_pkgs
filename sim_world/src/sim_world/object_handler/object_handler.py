import rospy
import numpy.random as random
import xml.etree.ElementTree as ET
import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any

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
        
        if not Path(self._config["objects_path"]).is_dir():
            raise ValueError("The object path is not a valid directory.")
        
        # Load the objects into the objects variable
        for category_folder in glob.glob(self._config["objects_path"] + "/*"):
            for object_folder in glob.glob(category_folder + "/*"):
                path = glob.glob(object_folder + '/mesh_new.sdf')[0]
                tree = ET.parse(path)
                xml_string = ET.tostring(tree.getroot(), encoding='utf8', method='xml').decode('utf-8')
                rospy.logwarn_once(type(xml_string))
                rospy.logwarn_once(xml_string)
                
                objects[Path(category_folder).name + "/" + Path(object_folder).name] = xml_string
        
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
    
        rospy.logwarn("Current object: " + self._curr_obj)
    
    @property
    def curr_obj(self):
        return self._curr_obj
    
    
    @property
    def objects(self):
        return self._objects
    
    @property
    def config(self):
        return self._config