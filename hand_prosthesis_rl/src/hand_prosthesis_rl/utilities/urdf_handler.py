import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
import urdfpy
import rospkg
import os

rospack = rospkg.RosPack()
DEFAULT_PATH = rospack.get_path("hand_prosthesis_env") + "/urdf/hands/mia_hand_default.urdf"

class URDFHandler():
    def __init__(self, urdf_file_path = DEFAULT_PATH):
        self._urdf_model = urdfpy.URDF.load(urdf_file_path)
        
    def get_link_names(self):
        link_names = []
        for link in self._urdf_model.links:
            link_names.append(link.name)
        return link_names

    def get_joint_names(self):
        joint_names = []
        for joint in self._urdf_model.joints:
            joint_names.append(joint.name)
        return joint_names

    def get_joint_position(self, joint_name):
        for joint in self._urdf_model.joints:
            if joint.name == joint_name:
                return joint.origin.xyz[2]

    def get_joint_positions(self):
        joint_positions = {}
        for joint in self._urdf_model.joints:
            joint_positions[joint.name] = joint.origin.xyz[2]
        return joint_positions

    def get_origin_and_scale(self, mesh_name):
        # Search link name by mesh name
        for link in self._urdf_model.links:
            for visual in link.visuals:
                if mesh_name in visual.geometry.mesh.filename:
                    origin = visual.origin
                    scale = visual.geometry.mesh.scale
                    break
        return origin, scale
    
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
    print(urdf_handler.get_link_names())
    
    origin, scale = urdf_handler.get_origin_and_scale("little_finger")
    