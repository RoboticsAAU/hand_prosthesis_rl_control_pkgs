import numpy as np
import bezier
from move_hand.path_planners.path_visualiser import animate_path
from move_hand.path_planners.path_planner import PathPlanner
from typing import Dict, Any
from move_hand.path_planners.orientation_planners.interpolation import interpolate_rotation
import rospy

class BezierPlanner(PathPlanner):
    def __init__(self):
        self._curves = []
    
    
    def plan_path(self, start_pose : np.array, goal_pose : np.array, parameters : Dict[str, Any]) -> np.ndarray:
        # Reset the planner
        self._reset()
        
        # Generate the bezier curve
        self.generate_bezier_curves_random(start_pose[:3], goal_pose[:3], parameters["num_way_points"])
        
        # Sample the bezier curve
        if parameters["sample_type"] == "constant":
            path = self.sample_bezier_curve_constant(parameters["num_points"])
        elif parameters["sample_type"] == "velocity":
            raise NotImplementedError
        
        # Combine the rotation into the path
        rotation = interpolate_rotation(start_pose[3:], goal_pose[3:], parameters["num_points"])
        path = np.concatenate((path, rotation), axis=0)
        
        return path
    
    
    def generate_bezier_curve(self, control_points: np.ndarray):
        """
        Generate a bezier curve from start to end
        :param control_points: Control points for the bezier curve (including start and end points)
        :return:
        """
        # Create the curve
        nodes = np.asfortranarray(control_points)
        # Append the curve to the list of curves
        self._curves.append(bezier.Curve(nodes, degree=control_points.shape[1]-1))
    
    
    def generate_bezier_curves_random(self, start_point : np.ndarray, end_point : np.ndarray, num_way_points : int, num_curves : int = 1, seed : int = None):
        """
        Generate a random bezier curve
        :param num_control_points: Number of control points to use
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        
        if num_way_points < 0 or num_curves < 1:
            raise ValueError("Invalid input for number of way points or number of curves")
        
        # Generate all the way points
        control_points = np.zeros((3, 2 + num_way_points))
        control_points[:, 0] = start_point
        control_points[:, -1] = end_point
        for _ in range(num_curves):
            for i in range(num_way_points):
                control_points[:, i + 1] = np.random.uniform(start_point, end_point, 3)

            self.generate_bezier_curve(control_points)
        
        return control_points
    
    
    def sample_bezier_curve_constant(self, num_points : float, index : int = 0) -> np.ndarray:
        """
        Sample the curve based on the number of points
        :param num_points: Number of points to sample on the curve
        :return: Sampled points on the curve
        """
        return self._curves[index].evaluate_multi(np.linspace(0, 1, num_points))
    
    
    def sample_bezier_curve_velocity(self, velocity_profile : np.ndarray, index : int = 0) -> np.ndarray:
        """
        Sample the curve based on a normalised velocity profile
        :param velocity_profile: Normalised velocity profile
        :return: Sampled points on the curve
        """
        
        # Integrate over velocity profile to get distance
        dt = 1 / len(velocity_profile)
        distance_profile = np.cumsum((velocity_profile[:-1] + velocity_profile[1:]) / 2) * dt 
        distance_profile = np.append([0.0], distance_profile)
        
        # Normalise distance
        distance_profile /= distance_profile[-1]
        
        return self._curves[index].evaluate_multi(distance_profile)
    
    
    def _reset(self):
        """
        Reset the planner
        :return:
        """
        self._curves = []
    
    
    @property
    def curves(self):
        return self._curves

    
if __name__ == "__main__":
    # Test
    bp = BezierPlanner()
    control_points = np.array([[0, 1, 9],
                               [0, 3, 9],
                               [0, 5, 9]])
    #bp.generate_bezier_curve(control_points)
    # bp.generate_bezier_curves_random(np.array([0, 0, 0]), np.array([9, 9, 9]), 2)
    
    # path = bp.sample_bezier_curve_constant(100)
    # path2 = bp.sample_bezier_curve_velocity(np.linspace(0, 1, 100))
    # animate_path(path, final_time=10)
    # animate_path(path2, final_time=10)
    
    # Create a figure with subplots
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    
    np.random.seed(2)
    control_points_1 = bp.generate_bezier_curves_random(np.array([0, 0, 0]), np.array([10, 10, 10]), 1)
    path_1 = bp.sample_bezier_curve_constant(100, 0)
    np.random.seed(5)
    control_points_2 = bp.generate_bezier_curves_random(np.array([0, 0, 0]), np.array([10, 10, 10]), 2)
    path_2 = bp.sample_bezier_curve_constant(100, 1)
    np.random.seed(7)
    control_points_5 = bp.generate_bezier_curves_random(np.array([0, 0, 0]), np.array([10, 10, 10]), 5)
    path_5 = bp.sample_bezier_curve_constant(100, 2)
    
    # Plot for 1 control point
    axs[0].plot(path_1[0], path_1[1], color='blue', linestyle='-', label='Path')  # Plot the path
    axs[0].plot(control_points_1[0,:], control_points_1[1,:], color='black', linestyle="--", marker="o", alpha=0.5)  # Plot the control points
    axs[0].scatter(control_points_1[0,[0, -1]], control_points_1[1,[0, -1]], color='black', marker='o', zorder=5, label='Int. Control Points')  # Plot the control points
    axs[0].set_xlabel('X-axis', fontsize=14)  # Label for the x-axis with increased font size
    axs[0].set_ylabel('Y-axis', fontsize=14)  # Label for the y-axis with increased font size
    axs[0].set_title('Bezier Path - 1 Int. Control Point', fontsize=16)  # Title of the plot with increased font size
    axs[0].legend(fontsize=12)  # Show legend with increased font size
    axs[0].grid(True)  # Show grid

    # Plot for 2 control points
    axs[1].plot(path_2[0], path_2[1], color='blue', linestyle='-', label='Path')  # Plot the path
    axs[1].plot(control_points_2[0,:], control_points_2[1,:], color='black', linestyle="--", marker="o", alpha=0.5)  # Plot the control points
    axs[1].scatter(control_points_2[0,[0, -1]], control_points_2[1,[0, -1]], color='black', marker='o', zorder=5, label='Int. Control Points')  # Plot the control points
    axs[1].set_xlabel('X-axis', fontsize=14)  # Label for the x-axis with increased font size
    axs[1].set_ylabel('Y-axis', fontsize=14)  # Label for the y-axis with increased font size
    axs[1].set_title('Bezier Path - 2 Int. Control Points', fontsize=16)  # Title of the plot with increased font size
    axs[1].legend(fontsize=12)  # Show legend with increased font size
    axs[1].grid(True)  # Show grid

    # Plot for 5 control points
    axs[2].plot(path_5[0], path_5[1], color='blue', linestyle='-', label='Path')  # Plot the path
    axs[2].plot(control_points_5[0,:], control_points_5[1,:], color='black', linestyle="--", marker="o", alpha=0.5)  # Plot the control points
    axs[2].scatter(control_points_5[0,[0, -1]], control_points_5[1,[0, -1]], color='black', marker='o', zorder=5, label='Int. Control Points')  # Plot the control points
    axs[2].set_xlabel('X-axis', fontsize=14)  # Label for the x-axis with increased font size
    axs[2].set_ylabel('Y-axis', fontsize=14)  # Label for the y-axis with increased font size
    axs[2].set_title('Bezier Path - 5 Int. Control Points', fontsize=16)  # Title of the plot with increased font size
    axs[2].legend(fontsize=12)  # Show legend with increased font size
    axs[2].grid(True)  # Show grid

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the figure as PDF
    plt.savefig('bezier_plots_2d.pdf', bbox_inches='tight')

    plt.show()  # Display the plot
    
