import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
import urdfpy
import rospkg

rospack = rospkg.RosPack()
DEFAULT_PATH = rospack.get_path("sim_world") + "/urdf/hands/mia_hand_default.urdf"

class URDFHandler():
    def __init__(self, urdf_file_path = DEFAULT_PATH):
        # Load the urdf file
        self._urdf_model = urdfpy.URDF.load(urdf_file_path)
        
        # Compute the transform relations between all the links
        self._transform_relations = {}
        self._transform_dict = {}
        for joint in self._urdf_model.joints:
            self._transform_relations[joint.child] = joint.parent
            self._transform_dict[(joint.parent, joint.child)] = joint.origin
        
        # Save the joints and links in dictionaries
        self._joints = {joint.name : joint for joint in self._urdf_model.joints}
        self._links = {link.name : link for link in self._urdf_model.links}

    def get_visual_origin_and_scale(self, mesh_name):
        """
        Returns the origin and scale of the visual geometry of the link that contains the given mesh
        :param mesh_name: The name of the mesh
        """
        # Search link name by mesh name
        for link in self._urdf_model.links:
            for visual in link.visuals:
                if mesh_name not in visual.geometry.mesh.filename:
                    continue
                
                return visual.origin, visual.geometry.mesh.scale
        # The mesh was not found in any link
        return None
    
    def get_link_name(self, mesh_name):
        """
        Returns the name of the link that contains the given mesh
        :param mesh_name: The name of the mesh
        """
        # Search link name by mesh name
        for link in self._urdf_model.links:
            for visual in link.visuals:
                if mesh_name not in visual.geometry.mesh.filename:
                    continue
                
                return link.name
        # The mesh was not found in any link
        return None
    
    def get_link_transform(self, parent, child):
        """
        Returns the transform from the parent link to the child link
        :param parent: The name of the parent link
        :param child: The name of the child link
        """
        # If the parent and child are the same, return the identity matrix
        if parent == child:
            return np.eye(4)
        
        # Check if the child is in the transform relations
        if child not in self._transform_relations.keys():
            raise ValueError("The child link is not in the transform relations")
        
        # Iteratively compute the transform from the parent to the child
        tmp_parent = self._transform_relations[child]
        transform = self._transform_dict[(tmp_parent, child)]
        while tmp_parent != parent:
            child = tmp_parent
            tmp_parent = self._transform_relations[tmp_parent]
            transform = self._transform_dict[(tmp_parent, child)] @ transform
        
        return transform.copy()
    
    def get_free_joints(self):
        """
        Returns the names of the free joints as a list, i.e. those joints that are not fixed
        """
        return [joint.name for joint in self._urdf_model.joints if joint.joint_type in ["revolute", "continuous"]]
    
    def get_link_group(self, link):
        """
        Returns the group/tree of links that are connected to the given link (stops at next free joint or end of tree)
        :param link: The name of the link
        """
        def get_link_child(link_name):
            # Get the joint as the first entry in the list that has the link as parent
            joint = next((joint for joint in self._urdf_model.joints if joint.parent == link_name), None)
            
            # Return none if we are at the end of the tree or if the new joint is free (part of a new group)
            if joint is None or (joint.name in self.get_free_joints() and joint.name != joint):
                return None
            
            # Return the child link of the joint
            return joint.child
        
        # Iteratively get the links in the group
        link_group = []
        while link != None:
            link_group.append(link)
            link = get_link_child(link)
        
        return link_group
    
    def get_free_joint_group(self, free_joint):
        '''
        Returns the group of links that are connected to the free joint
        :param free_joint: The name of the free joint
        '''
        # Get the child link of the free joint and return the group of links connected to it
        child_link = self._joints[free_joint].child
        return self.get_link_group(child_link)
    
    # CANNOT BE USED UNTIL WE FIX THE MESH FILE PATHS
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
    