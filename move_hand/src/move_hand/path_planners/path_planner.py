import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class PathPlanner(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def plan_path(self, start_pos : np.array, goal_pos : np.array, parameters : Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def _reset(self) -> None:
        pass