import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
import urdfpy
import rospkg
import os

rospack = rospkg.RosPack()
DEFAULT_PATH = rospack.get_path("simulation_world") + "/urdf/hands/mia_hand_default.urdf"

class URDFHandler():
    def __init__(self, urdf_file_path = DEFAULT_PATH):
        self._urdf_model = urdfpy.URDF.load(urdf_file_path)
        self._transform_relations = {}
        self._transform_dict = {}
        for joint in self._urdf_model.joints:
            self._transform_relations[joint.child] = joint.parent
            self._transform_dict[(joint.parent, joint.child)] = joint.origin
        self._joints = {joint.name : joint for joint in self._urdf_model.joints}
        self._links = {link.name : link for link in self._urdf_model.links}

    def get_visual_origin_and_scale(self, mesh_name):
        # Search link name by mesh name
        for link in self._urdf_model.links:
            for visual in link.visuals:
                if mesh_name in visual.geometry.mesh.filename:
                    origin = visual.origin
                    scale = visual.geometry.mesh.scale
                    break
        return origin, scale    
    
    def get_link_name(self, mesh_name):
        for link in self._urdf_model.links:
            for visual in link.visuals:
                if mesh_name in visual.geometry.mesh.filename:
                    return link.name
        return None
    
    def get_link_transform(self, parent, child):
        if parent == child:
            return np.eye(4)
        
        if child not in self._transform_relations:
            raise ValueError("The child link is not in the transform relations")
        
        tmp_parent = self._transform_relations[child]
        
        transform = self._transform_dict[(tmp_parent, child)]
        while tmp_parent != parent:
            child = tmp_parent
            tmp_parent = self._transform_relations[tmp_parent]
            transform = self._transform_dict[(tmp_parent, child)] @ transform
        
        return transform
    
    def get_free_joints(self):
        return [joint.name for joint in self._urdf_model.joints if joint.joint_type in ["revolute", "continuous"]]
    
    def get_link_child(self, link_name):
        # Get the joint as the first entry in the list that has the link as parent
        joint = next((joint for joint in self._urdf_model.joints if joint.parent == link_name), None)
        
        if joint is None or (joint.name in self.get_free_joints() and joint.name != joint):
            return None
        
        return joint.child
    
    def get_link_group(self, link):
        link_group = []
        while link != None:
            link_group.append(link)
            link = self.get_link_child(link)
            
        return link_group
    
    def get_free_joint_group(self, free_joint):
        child_link = self._joints[free_joint].child
        return self.get_link_group(child_link)
    
    def is_group_parent(self, link):
        test = self.get_link_group(link)
        return link == self.get_link_group(link)[0]
    
    # CAN BE USED IF WE FIX THE MESH FILE PATHS
    def get_mesh_files(self):
        mesh_files = []
        for link in self._urdf_model.links:
            for visual in link.visuals:
                package_name, completion = visual.geometry.mesh.filename.split('//')[1].split('/', 1)
                file_path = rospack.get_path(package_name) + "/" + completion
                mesh_files.append(file_path)
        return mesh_files


if __name__ == "__main__":
    # # Get an instance of RosPack with the default search paths
    # rospack = rospkg.RosPack()

    # # Get the stl files from the mia description package
    # urdf_file = rospack.get_path('mia_hand_description') + "/urdf/mia_hand_default.urdf"
    
    urdf_handler = URDFHandler(DEFAULT_PATH)
    
    origin, scale = urdf_handler.get_visual_origin_and_scale("little_finger")
    