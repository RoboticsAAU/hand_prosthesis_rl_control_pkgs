import open3d as o3d
import numpy as np
from typing import List
import rospkg
import glob
import ntpath

class PointCloudHandler():
    def __init__(self):
        self._pc = o3d.geometry.PointCloud()
    
    def visualize(self):
        """
        Visualize its own point cloud.
        :param point_cloud: Open3D PointCloud object
        :return:
        """
        if self._pc is None:
            raise ValueError("The point cloud is not set.")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self._pc)
        vis.run()
        vis.destroy_window()
    
    
    def sample_from_meshes(self, mesh_files : List[str], total_sample_points : int = 1000):
        """
        It will sample the mesh files (either .stl or .obj).
        :param stl_files: The path to the stl files
        :param total_sample_points: The total number of points to sample
        :return:
        """
        
        sample_points = total_sample_points // len(mesh_files)
        
        for mesh_file in mesh_files:
            point_cloud = self.sample_from_mesh(mesh_file, sample_points)
            self.combine(point_cloud)
    
    def isolate_object(self):
        raise NotImplementedError("This method is not implemented yet.")
    
    
    def combine(self, point_cloud):
        self._pc += point_cloud
    
    def clear(self):
        self._pc.clear()
        
    
    @staticmethod
    def sample_from_mesh(mesh_file : str, sample_points : int = 1000):
        """
        It will sample the mesh file (either .stl or .obj).
        :param stl_file: The path to the stl file
        :param sample_points: The number of points to sample
        :return: The point cloud
        """
        # Read the stl file
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        # Sample the stl file
        return mesh.sample_points_uniformly(number_of_points=sample_points)

    @property
    def pc(self):
        return self._pc



if __name__ == "__main__":
    # Create the point cloud handler
    point_cloud_handler = PointCloudHandler()
    
    # Get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # Get the stl files from the mia description package
    ignore_files = ["1.001.stl", "UR_flange.stl"]
    stl_folder = rospack.get_path('mia_hand_description') + "/meshes/stl"
    stl_files = [file for file in glob.glob(stl_folder + "/*") if (ntpath.basename(file) not in ignore_files)]
    
    # Extract stl files for left and right hand respectively
    stl_files_left, stl_files_right = [], []
    for x in stl_files:
        (stl_files_right, stl_files_left)["mirrored" in x].append(x)
    
    # Load the stl file
    point_cloud_handler.sample_from_meshes(stl_files_right)
    
    # Visualize the point cloud
    point_cloud_handler.visualize()