import rospy
import numpy as np
import numpy.random as random
from stl import mesh
from geometry_msgs.msg import Pose
from typing import Dict, Any, Union
from scipy.spatial.transform import Rotation as R
from move_hand.path_planners import bezier_paths, navigation_function, path_planner

# TODO: Compute trajectories for the hand
# TODO: The hand orientation could always point towards some point, or the hand could be oriented tangent to the trajectory


class HandController:
    def __init__(self, move_hand_config : Dict[str, Any], hand_rotation : np.array):
        # Create the gazebo interface
        self._config = move_hand_config
        # To store the planned path
        self._pose_buffer = np.array([])
        
        # Assign the path planner
        try:
            if self._config["path_planner"] == "bezier":
                self._path_planner = bezier_paths.BezierPlanner()
            elif self._config["path_planner"] == "navigation_function":
                self._path_planner = navigation_function.NavFuncPlanner(world_dim=3, world_sphere_rad=self._config["outer_radius"]+0.5)
            else:
                raise ValueError("Path planner not recognized")
        except ValueError as e:
            rospy.logwarn("Failed to set path planner: ", e)
            rospy.logwarn("Defaulting to straight line path planner")
            self._path_planner = path_planner.PathPlanner()
        
        # Variable to store the pose of the hand's palm frame
        self._pose = Pose()
        
        # Store hand rotation
        self._hand_rotation = hand_rotation


    def update(self, hand_state : Dict[str, Any]) -> None:
        # Update the hand controller state
        self._pose = hand_state["pose"]


    def step(self) -> np.array:
        
        if self._pose_buffer.shape[1] == 0:
            raise IndexError("Empty pose buffer")
        
        first_pose, self._pose_buffer = self._pose_buffer[:,0], self._pose_buffer[:,1:]
        return first_pose
    
    
    def plan_trajectory(self, obj_center : np.array, obj_mesh) -> None:
        # Obtain the start and goal pose
        #TODO: z-offset should be a parameter in yaml
        start_pose = self._sample_start_pose(obj_center, 0.1)
        goal_pose = self._sample_goal_pose(obj_center, start_pose, obj_mesh)
                
        # Plan trajectory with the given path planner and parameters
        if self._config["path_planner"] == "bezier":
            path_params = {
                "num_way_points": random.randint(1, 5),
                "sample_type": "constant",
                "num_points": 1000,
                "dt" : 0.01,
            }
        elif self._config["path_planner"] == "navigation_function":
            path_params = {
                "num_rand_obs": random.randint(1, 5),
                "obs_rad": random.uniform(0.1, 0.5),
                "kappa": 5,
                "step_size": 0.1,
                "dt" : 0.01,
            }
        
        self._pose_buffer = self._path_planner.plan_path(start_pose, goal_pose, path_params)
        
        
        # to_append = np.vstack([start_pose[3:].copy()] * self._pose_buffer.shape[1]).T
        # self._pose_buffer = np.append(self._pose_buffer, to_append, axis=0)
        

    def reset(self):
        self._pose_buffer = []
    
    
    def _sample_start_pose(self, obj_center : np.array, z_offset : float) -> np.array:
        """
        Sample a start pose for the hand. The start pose is sampled on the upper hemisphere.
        obj_center: The center of the object
        z_offset: Hemisphere offset from the object center in the z-direction
        :return: The start pose for the hand
        """
        
        # Sample starting point on upper-half of unit sphere (i.e., z>0)
        def sample_spherical() -> np.array:
            vec = np.random.uniform(-1, 1, (3,))
            # Convert the point to upper hemisphere
            vec[2] = abs(vec[2])
            # Add offset
            vec /= np.linalg.norm(vec, axis=0)
            return vec
        
        # Compute the start position
        rel_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        while rel_pos[2] < z_offset/self._config["outer_radius"]:
            rel_pos = sample_spherical()
        
        start_position = rel_pos*self._config["outer_radius"] + obj_center

        # Compute the start orientation
        auxiliary_vec = np.array([0, 0, 1.])
        z_axis = -rel_pos

        y_axis = np.cross(auxiliary_vec, z_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Create the rotation matrix and account for hand frame rotation  
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T @ self._hand_rotation

        # Convert it to quaternion
        start_orientation = R.from_matrix(rotation_matrix).as_quat()

        # Obtain the start pose as the combined position and orientation
        return np.concatenate([start_position, start_orientation])

    def _sample_goal_pose(self, obj_center : np.array, start_pose : np.array, obj_mesh : mesh.Mesh, normal_dist : float = 0.05) -> np.array:
        """ Sample a goal pose for the hand. The goal pose is sampled on the as some orthogonal distance to the object surface
        in the intersection point between the start pose and object center.
        obj_center: The center of the object
        start_pose: The start pose of the hand
        obj_mesh: The mesh of the object
        normal_dist: The orthogonal distance from the object surface
        :return: The goal pose for the hand
        """
        
        # Extract the triangles from the object mesh and shift them to the object center
        triangles = obj_mesh.vectors + obj_center[:3].reshape(1,-1)
        triangle_normals = obj_mesh.normals
        
        # Function to check if a line intersects a triangle and returns the intersection point 
        # and surface normal if it does
        def intersect_line_triangle(line_point1, line_point2, triangle, normal):
            # Define the line as a vector (from start position to object center) and a point on the line (start position)
            line_vector = line_point2 - line_point1
            line_point = line_point1
            
            # Define the triangle's vertices (extracted as rows in "triangle") and the triangle's normal
            v0, v1, v2 = triangle
            triangle_normal = normal / np.linalg.norm(normal)
            
            # Return early if the triangle normal from the mesh is pointing in the same direction as the line vector
            if np.dot(triangle_normal, line_vector) > 0:
                return False, None, None
            
            # Calculate the intersection point between the line and the plane. We do this by solving the equation of the line
            # and the plane simultaneously. The intersection point is given by the point on the line that satisfies the equation
            # of the plane. The value t describes how far along the line the intersection point is.
            # https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
            t = -(np.dot(triangle_normal, line_point - v0) / np.dot(triangle_normal, line_vector))
            intersection_point = line_point + t * line_vector
            
            # Return early if the intersection point is outside the line segment
            if t < 0 or t > 1:
                return False, None, None
            
            # Check if the intersection point is inside the triangle. We do this by checking if the intersection point is on 
            # the same side of each edge. Using the cross product between a vector on the triangle side and a vector from the 
            # triangle corner to the intersection point, we either get a vector pointing in the same direction or 
            # opposite direction of the triangle normal. If the vectors point in the same direction, the dot product will be positive,
            # otherwise it will be negative.
            u = np.dot(np.cross(v1 - v0, intersection_point - v0), triangle_normal)
            v = np.dot(np.cross(v2 - v1, intersection_point - v1), triangle_normal)
            w = np.dot(np.cross(v0 - v2, intersection_point - v2), triangle_normal)
            
            # Return early if the intersection point is outside the triangle
            if u < 0 or v < 0 or w < 0:
                return False, None, None
            
            # Return the intersection point and the triangle normal
            return True, intersection_point, triangle_normal
        
        # Find all the triangles that the line between start position and object center intersects with 
        intersections = []
        for triangle, normal in zip(triangles,  triangle_normals):
            intersection, intersection_point, triangle_normal = intersect_line_triangle(start_pose[:3], obj_center, triangle, normal)
            if intersection:
                intersections.append((intersection_point, triangle_normal))

        # Find the intersection point that is closest to the start position
        intersection_point, triangle_normal = min(intersections, key=lambda x: np.linalg.norm(x[0] - start_pose[:3]))
        
        # Compute the goal position as some orthogonal offset to the intersection point
        goal_position = intersection_point + normal_dist * triangle_normal
        
        def visualize(triangles, intersection_point, goal_position, surface_normal_vector):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d import art3d
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each triangle as a mesh
            # https://stackoverflow.com/questions/57948678/how-to-properly-plot-collection-of-polygons-stl-file
            # collection = art3d.Poly3DCollection(triangles, linewidths=0.5, edgecolors='k')
            # collection.set_facecolor('gray')  # Set face color to gray
            ax.add_collection3d(art3d.Poly3DCollection(triangles, facecolors='gray', edgecolors='k', linewidths=0.5, alpha=0.3))
            
            # Plot the intersection point
            ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], c='r', marker='o', label='Intersection Point')
            
            # Plot goal position
            ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='g', marker='o', label='Goal Position')
            
            ax.quiver(intersection_point[0], intersection_point[1], intersection_point[2],
              surface_normal_vector[0], surface_normal_vector[1], surface_normal_vector[2],
              length=0.05, color='m', normalize=True, label='Surface Normal')
            
            # Plot line from intersection point to goal position
            ax.plot([obj_center[0], start_pose[0]], 
                    [obj_center[1], start_pose[1]], 
                    [obj_center[2], start_pose[2]], 
                    color='b', linestyle='-', linewidth=1, label='Line from Object Center to Start Position')
            
            # Set axes limits
            lim = 0.2
            ax.set_xlim([-lim, lim])  # Set appropriate values for xmin and xmax
            ax.set_ylim([-lim, lim])  # Set appropriate values for ymin and ymax
            ax.set_zlim([-lim, lim])  # Set appropriate values for zmin and zmax
            
            # Set labels and show plot
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        
        visualize(triangles, intersection_point, goal_position, triangle_normal)
        
        
        # Store the rotation matrix from the start pose
        start_rot_matrix = R.from_quat(start_pose[3:]).as_matrix()
        
        def project_vector_onto_plane(v, plane_v1, plane_v2):
            # Calculate the normal vector of the plane
            n = np.cross(plane_v1, plane_v2)
            
            # Compute the projection of v onto the normal vector n
            projection = np.dot(v, n) / np.dot(n, n) * n
            
            # Subtract the projection from v to obtain the projected vector onto the plane
            projected_vector = v - projection
            
            return projected_vector
        
        # Compute the z-axis from the triangle normal
        z_axis = -triangle_normal
        
        # Compute the plane vectors as the orthogonal vectors to the z-axis and two random vectors 
        # (need not be orthogonal to z-axis)
        plane_vec1, plane_vec2 = np.cross(random.rand(3), z_axis), np.cross(random.rand(3), z_axis)

        # Project the start x- and y-axes onto this new plane
        projected_x = project_vector_onto_plane(start_rot_matrix[:, 0], plane_vec1, plane_vec2)
        projected_y = project_vector_onto_plane(start_rot_matrix[:, 1], plane_vec1, plane_vec2)
        
        # Choose the axis with the largest projection onto the plane (and normalise it), and select the other axis
        # with the right hand rule (this will ensure that the new rotation is similar to the start rotation)
        if np.linalg.norm(projected_x) > np.linalg.norm(projected_y):
            x_axis = projected_x / np.linalg.norm(projected_x)
            y_axis = np.cross(z_axis, x_axis)
        else:
            y_axis = projected_y / np.linalg.norm(projected_y)
            x_axis = np.cross(y_axis, z_axis)
        
        # Combine the rotation axes into a rotation matrix
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        
        # Convert it to quaternion
        goal_orientation = R.from_matrix(rotation_matrix).as_quat()
        
        return np.concatenate([goal_position, goal_orientation])

if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController({"path_planner": "bezier"}, np.eye(3))
    
    import rospkg
    from time import time
    rospack = rospkg.RosPack()
    stl_file = rospack.get_path('assets') + '/shapenet_sdf/category_1/obj_2/mesh.stl'
    obj_mesh = mesh.Mesh.from_file(stl_file)
    obj_mesh.vectors = obj_mesh.vectors * 0.15
    
    # start_pose = hand_controller._sample_start_pose(np.array([0,0,0]), 0.1)
    start_pose = np.array([1,2,1,0,0,0,1], dtype=np.float32)
    start = time()
    goal_pose = hand_controller._sample_goal_pose(np.array([0,0,0], dtype=np.float32), start_pose,  obj_mesh)
    
    from move_hand.path_planners.path_visualiser import plot_path
    
    path = np.concatenate([start_pose.reshape(-1,1), goal_pose.reshape(-1,1)], axis=1)
    plot_path(path)
    