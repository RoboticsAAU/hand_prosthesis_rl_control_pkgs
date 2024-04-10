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
        objects = {}
        
        try:
            if not Path(self._config["object_path"]).is_dir():
                raise ValueError("The object path is not a valid directory.")
            
            # Load the objects into the objects variable
            for category_folder in glob.glob(self._config["object_path"] ):
                for object_folder in glob.glob(category_folder):
                    path = glob.glob(object_folder + '/*.xml')[0]
                    tree = ET.parse(path)
                    xml_string = ET.tostring(tree.getroot(), encoding='utf8', method='xml')
                    
                    objects[Path(category_folder).name + "/" + Path(object_folder).name] = xml_string
            
            self._config["num_objects"] = self.config["num_objects"] if self.config["num_objects"] != -1 else len(objects)
        
            if len(objects) < self._config["num_objects"]:
                raise ValueError("The number of objects is less than the number of objects requested.")
            
            random_keys = random.sample(list(objects.keys()), k=self._config["num_objects"])
            return OrderedDict((key, objects[key]) for key in random_keys)
        
        except ValueError as e:
            rospy.logerr("Could not load the objects from the object path: ", e)

            
    
    def update_current_object(self) -> None:
        try:
            if self._config["mode"] == "random":
                self._curr_obj = random.choice(list(self._objects.keys()))
                self._obj_idx = list(self._objects.keys()).index(self._curr_obj)
                
            elif self._config["mode"] == "sequential":
                self._curr_obj = list(self.objects.keys())[self._obj_idx]
                self._obj_idx = (self._obj_idx + 1) % self._config["num_objects"]
            
            else:
                raise ValueError("The object handler mode is not valid.")
            
        except ValueError as e:
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