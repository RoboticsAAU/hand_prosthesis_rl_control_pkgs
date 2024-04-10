import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class PathPlanner(ABC):
    def __init__(self) -> None:
        pass
    
    def plan_path(self, start_pos : np.array, goal_pos : np.array, parameters : Dict[str, Any]) -> None:
        """ Plan a path between start_pos and goal_pos with the given parameters. Here, a default implementation is provided for linear interpolation. 
        """
        # Generate parameter values (t) for linear interpolation
        t_values = np.linspace(0, 1, parameters["num_points"])
        
        # Perform linear interpolation between start_pos and goal_pos
        path = np.array([(1 - t) * start_pos + t * goal_pos for t in t_values])
        
        return path
    
    @abstractmethod
    def _reset(self) -> None:
        pass