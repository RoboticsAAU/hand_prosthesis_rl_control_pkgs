import open3d as o3d
import numpy as np
from typing import List
import rospkg
import glob
from pathlib import Path
from hand_prosthesis_rl.utilities.urdf_handler import URDFHandler

class PointCloudHandler():
    def __init__(self, point_cloud : o3d.geometry.PointCloud = None):
        self._pc = point_cloud if point_cloud is not None else o3d.geometry.PointCloud()
    
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
    
    def sample_from_meshes(self, 
                           mesh_dict : dict, 
                           total_sample_points : int = 1000):
        """
        It will sample the mesh files (either .stl or .obj).
        :param stl_files: The path to the stl files
        :param total_sample_points: The total number of points to sample
        :return:
        """
        sample_points = total_sample_points // len(mesh_dict)
        
        for mesh_values in mesh_dict.values():
            point_cloud = self.sample_from_mesh(mesh_values["path"], sample_points)
            
            if mesh_values["scale_factors"] is not None:
                self.scale(point_cloud, mesh_values["scale_factors"])
            
            if mesh_values["origin"] is not None:
                self.transform(point_cloud, mesh_values["origin"])
            
            self.combine(point_cloud)
    
    def remove_plane(self):
        """
        It will remove the plane from the point cloud.
        :return:
        """
        # Isolate the plane in the point cloud
        (plane_coeffs, plane_indices) = self._pc.segment_plane(distance_threshold=2, ransac_n=3, num_iterations=1000)
        
        # Remove the plane from the point cloud
        self._pc = self._pc.select_by_index(plane_indices, invert=True)        
    
    def combine(self, 
                point_cloud : o3d.geometry.PointCloud):
        """
        It will combine the point cloud with the current point cloud.
        :param point_cloud: The point cloud to combine
        """
        self._pc += point_cloud
    
    def clear(self):
        self._pc.clear()
        
    
    @staticmethod
    def sample_from_mesh(mesh_file : str, 
                         sample_points : int = 1000, 
                         transformation : np.ndarray = None):
        """
        It will sample the mesh file (either .stl or .obj).
        :param stl_file: The path to the stl file
        :param sample_points: The number of points to sample
        :return: The point cloud
        """
        # Read the stl file
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # Sample the stl file
        pc =  mesh.sample_points_uniformly(number_of_points=sample_points)
        
        # Apply transformation if it is provided
        if transformation is not None:
            pc.transform(transformation)
        
        return pc

    @staticmethod
    def scale(pc : o3d.geometry.PointCloud,
              scale_factors : np.array, 
              transform : np.array = None):
        """
        It will scale the point cloud by the scale factors.
        :param scale_factor: The scale factors in each direction
        :param transform: The coordinate frame to scale the point cloud about
        :return:
        """
        if transform is not None:
            pc.transform(np.linalg.inv(transform))
        
        scaling_matrix = np.diag(np.append(scale_factors, 1))
        
        pc.transform(scaling_matrix)
        
        if transform is not None:
            pc.transform(transform)
        
    @staticmethod
    def transform(pc : o3d.geometry.PointCloud,
                  transform : np.ndarray):
        """
        It will transform the point cloud.
        :param pc: The point cloud
        :return:
        """
        pc.transform(transform)
        
        
    @property
    def pc(self):
        return self._pc
    
    @pc.setter
    def pc(self, point_cloud : o3d.geometry.PointCloud):
        self._pc = point_cloud

    @property
    def points(self):
        return np.asarray(self.pc.points)


if __name__ == "__main__":
    # Create the point cloud handler
    point_cloud_handler = PointCloudHandler()
    urdf_handler = URDFHandler()
    
    # Get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # Get the stl files from the mia description package
    ignore_files = ["1.001.stl", "UR_flange.stl"]
    stl_folder = rospack.get_path('mia_hand_description') + "/meshes/stl"
    stl_files = [file for file in glob.glob(stl_folder + "/*") if (Path(file).name not in ignore_files)]
    
    # Extract stl files for left and right hand respectively
    stl_files_left, stl_files_right = [], []
    for x in stl_files:
        (stl_files_right, stl_files_left)["mirrored" in x].append(x)
    
    
    # Create a dictionary for the stl files
    mesh_dict = {}  # Initialize an empty dictionary
    for stl_file in stl_files_right:
        origin, scale = urdf_handler.get_origin_and_scale(Path(stl_file).stem)
        
        link_name = urdf_handler.get_link_name(Path(stl_file).stem)
        origin = urdf_handler.get_link_transform("palm", link_name) @ origin        
        
        # Create a dictionary for each mesh file
        mesh_dict[Path(stl_file).stem] = {
            'path': stl_file,  # Construct the path for the file
            'scale_factors': scale,  # Assign scale factors
            'origin': origin  # Assign origin
        }
    
    # Load the stl file
    point_cloud_handler.sample_from_meshes(mesh_dict, 10000)
    #pc_plane = PointCloudHandler.sample_from_mesh("./plane.stl", 1000)
    #point_cloud_handler.combine(pc_plane)
    
    # Visualize the point cloud
    point_cloud_handler.visualize()
    
    #point_cloud_handler.remove_plane()
    #point_cloud_handler.visualize()