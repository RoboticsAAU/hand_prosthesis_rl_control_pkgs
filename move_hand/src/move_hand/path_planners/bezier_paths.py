import numpy as np
import bezier
from move_hand.path_planners.path_visualiser import animate_path
from move_hand.path_planners.path_planner import PathPlanner
from typing import Dict, Any
from move_hand.path_planners.orientation_planners.interpolation import interpolate_rotation

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
        path = np.concatenate((path, rotation), axis=1)
        
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
    bp.generate_bezier_curves_random(np.array([0, 0, 0]), np.array([9, 9, 9]), 2)
    
    path = bp.sample_bezier_curve_constant(100)
    path2 = bp.sample_bezier_curve_velocity(np.linspace(0, 1, 100))
    animate_path(path, final_time=10)
    animate_path(path2, final_time=10)