from typing import Dict, Any

class ObjectHandler():
    def __init__(self, folder_path : str):
        self._objects = self.load_objects(folder_path)
    
    def load_objects(self, folder_path : str):
        # Load the objects into the objects variable
        pass
    
    def get_random_objects(self, num_objects):
        # Get a random set of objects from the objects variable
        pass
    
    @property
    def objects(self):
        return self._objects