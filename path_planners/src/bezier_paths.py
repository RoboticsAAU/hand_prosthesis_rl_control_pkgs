import numpy as np
import bezier
from path_visualiser import animate_path

class BezierPlanner():
    def __init__(self):
        pass
    
    def generate_bezier_path(self, control_points: np.ndarray, num_points : float) -> np.ndarray:
        """
        Generate a bezier path from start to end
        :param control_points: Control points for the bezier curve (including start and end points)
        :param num_points: Number of points to sample on the curve
        """
        # Create the curve
        nodes = np.asfortranarray(control_points)
        curve = bezier.Curve(nodes, degree=len(control_points)-1)
        
        # Sample the curve based on the number of points
        return curve.evaluate_multi(np.linspace(0, 1, num_points))
    

if __name__ == "__main__":
    # Test
    bp = BezierPlanner()
    control_points = np.array([[0, 1, 9],
                               [0, 3, 9],
                               [0, 5, 9]])
    path = bp.generate_bezier_path(control_points, 100)
    animate_path(path, sample_rate=10, save_file="test.gif")