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
        
        if len(self._pose_buffer) == 0:
            raise IndexError("Empty pose buffer")
        
        first_pose, self._pose_buffer = self._pose_buffer[:,0], self._pose_buffer[:,1:]
        return first_pose
    
    
    def plan_trajectory(self, obj_center : np.array) -> None:
        # Obtain the start and goal pose
        start_pose = self._sample_start_pose(obj_center)
        goal_pose = self._sample_goal_pose(obj_center, start_pose)
        
        # Plan trajectory with the given path planner and parameters
        if self._config["path_planner"] == "bezier":
            path_params = {
                "num_way_points": random.randint(1, 5),
                "sample_type": "constant",
                "num_points": 1000,
            }
        elif self._config["path_planner"] == "navigation_function":
            path_params = {
                "num_rand_obs": random.randint(1, 5),
                "obs_rad": random.uniform(0.1, 0.5),
                "kappa": 5,
                "step_size": 0.1,
            }
        
        #TODO: Implement orientation in the path planner
        self._pose_buffer = self._path_planner.plan_path(start_pose[:3], goal_pose[:3], path_params)
        to_append = np.vstack([np.array([0,0,0,1])] * self._pose_buffer.shape[1]).T
        self._pose_buffer = np.append(self._pose_buffer, to_append, axis=0)
        rospy.logwarn("Planned trajectory shape: " + str(self._pose_buffer.shape))
        

    def reset(self):
        self._pose_buffer = []
    
    
    def _sample_start_pose(self, obj_center : np.array) -> np.array:
                # Sample starting point on outer sphere
        def sample_spherical() -> np.array:
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec, axis=0)
            vec *= self._config["outer_radius"]
            return vec
        
        # Compute the start position
        rel_pos = sample_spherical()
        start_position = rel_pos + obj_center

        # Compute the start orientation
        auxiliary_vec = np.array([0, 0, 1.])
        z_axis = -rel_pos

        y_axis = np.cross(auxiliary_vec, z_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # If x-axis is not pointing down, flip the orientation 180 degrees around the z-axis
        if x_axis[2] > 0:
            x_axis, y_axis = -x_axis, -y_axis

        # Create the rotation matrix and account for hand frame rotation  
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T @ self._hand_rotation
        # Convert it to quaternion
        start_orientation = R.from_matrix(rotation_matrix).as_quat()

        # Obtain the start pose as the combined position and orientation
        return np.concatenate([start_position, start_orientation])

    # def _sample_goal_pose(self, obj_center : np.array, start_pose : np.array) -> np.array:
    #     import rospkg
    #     rospack = rospkg.RosPack()
    #     stl_file = rospack.get_path('assets') + '/shapenet_sdf/category_1/obj_1/mesh.stl'
    #     cuboid = mesh.Mesh.from_file(stl_file)
    #     points = np.around(np.unique(cuboid.vectors.reshape([cuboid.vectors.size//3, 3]), axis=0), 4)
    #     # print("Points are", points.tolist())
        
    #     def closest_point_on_line(point, line_point1, line_point2):
    #         line_vector = line_point2 - line_point1
    #         line_length_squared = np.dot(line_vector, line_vector)
            
    #         if line_length_squared == 0:
    #             return line_point1, np.linalg.norm(point - line_point1)  # Return distance to the only point
            
    #         # Vector from line_point1 to the given point
    #         point_vector = point - line_point1

    #         # Projection factor
    #         t = np.dot(point_vector, line_vector) / line_length_squared
            
    #         # Clamp t to the range [0, 1]
    #         t = np.clip(t, 0, 1)
            
    #         # Closest point on the line
    #         closest_point = line_point1 + t * line_vector
            
    #         # Distance between original point and closest point
    #         distance = np.linalg.norm((point - closest_point))
            
    #         return closest_point, distance

    #     closest_point, distance_to_closest_point = min(
    #         ((closest_point_on_line(point, start_pose[:3], obj_center)) 
    #         for point in points),
    #         key=lambda x: x[1]
    #     )
    #     print("Closest point is", closest_point, "and closest distance is", distance_to_closest_point)
        
    #     # Calculate distances to the closest point
    #     distances_to_closest_point = np.linalg.norm(points - closest_point, axis=1)
        
    #     # Get the indices of the 3 closest points
    #     closest_points_idx = np.argpartition(distances_to_closest_point, 3)
    #     closest_points = points[closest_points_idx[:3]]
        
    #     def surface_normal(points):
    #         # Calculate vectors between points
    #         vectors = points[1:] - points[0]
            
    #         # Calculate cross product of two vectors to find the normal vector
    #         normal_vector = np.cross(vectors[0], vectors[1])
            
    #         # Normalize the normal vector
    #         normal_vector /= np.linalg.norm(normal_vector)
            
    #         return normal_vector
        
    #     surface_normal_vector = surface_normal(closest_points)
    #     # Ensure the direction is correct
    #     if np.dot(surface_normal_vector, start_pose[:3] - obj_center) < 0:
    #         surface_normal_vector = -surface_normal_vector
        
    #     goal_position = closest_point + 0.05 * surface_normal_vector
        

    #     def visualize_points(points, closest_points, goal_position, surface_normal_vector):
    #         import matplotlib.pyplot as plt
    #         from mpl_toolkits.mplot3d import Axes3D
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
            
    #         # Plot all points
    #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')
            
    #         # Plot closest points
    #         ax.scatter(closest_points[:, 0], closest_points[:, 1], closest_points[:, 2], c='r', marker='o', label='Closest Points')
            
    #         # Plot goal position
    #         ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='g', marker='o', label='Goal Position')
            
    #         ax.quiver(closest_point[0], closest_point[1], closest_point[2],
    #           surface_normal_vector[0], surface_normal_vector[1], surface_normal_vector[2],
    #           length=0.05, color='k', normalize=True, label='Surface Normal')
            
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.legend()
            
    #         plt.show()
        
    #     visualize_points(points, closest_points, goal_position, surface_normal_vector)
        
    #     return np.concatenate([goal_position, start_pose[3:]])

    def _sample_goal_pose(self, obj_center : np.array, start_pose : np.array) -> np.array:
        import rospkg
        from time import time
        start = time()
        rospack = rospkg.RosPack()
        stl_file = rospack.get_path('assets') + '/shapenet_sdf/category_1/obj_1/mesh.stl'
        cuboid = mesh.Mesh.from_file(stl_file)
        triangles = cuboid.vectors * 0.15
        
        def intersect_line_triangle(line_point1, line_point2, triangle):
            # Define the line as a vector and a point on the line
            line_vector = line_point2 - line_point1
            line_point = line_point1
            
            # Define the triangle's vertices (extracted as rows in "triangle")
            v0, v1, v2 = triangle
            
            # Define the triangle's plane
            triangle_normal = np.cross(v1 - v0, v2 - v0)
            triangle_normal /= np.linalg.norm(triangle_normal)
            triangle_d = -np.dot(triangle_normal, v0)
            
            # Calculate the intersection point between the line and the plane
            t = -(np.dot(triangle_normal, line_point) + triangle_d) / np.dot(triangle_normal, line_vector)
            intersection_point = line_point + t * line_vector
            
            # Check if the intersection point is inside the triangle
            u = np.dot(np.cross(v1 - v0, intersection_point - v0), triangle_normal)
            v = np.dot(np.cross(v2 - v1, intersection_point - v1), triangle_normal)
            w = np.dot(np.cross(v0 - v2, intersection_point - v2), triangle_normal)
            
            if u >= 0 and v >= 0 and w >= 0 and 0 <= t <= 1:
                return True, intersection_point, triangle_normal
            else:
                return False, None, None
        
        intersections = []
        for triangle in triangles:
            intersection, intersection_point, triangle_normal = intersect_line_triangle(obj_center, start_pose[:3], triangle)
            if intersection:
                intersections.append((intersection_point, triangle_normal))

        intersection_point, triangle_normal = min(intersections, key=lambda x: np.linalg.norm(x[0] - start_pose[:3]))
        
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
                    color='b', linestyle='-', linewidth=1, label='Line from Intersection to Goal')
            
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
        
        goal_position = intersection_point + 0.05 * triangle_normal
        print(f"Duration: {time() - start:.2f} s")
        
        #visualize(triangles, intersection_point, goal_position, triangle_normal)
        
        return np.concatenate([goal_position, start_pose[3:]])

if __name__ == '__main__':
    # Test move hand controller class
    hand_controller = HandController({"path_planner": "bezier"}, np.eye(3))
    
    goal_pose = hand_controller._sample_goal_pose(np.array([0,0,0]), np.array([1,-1,1]))
    
    