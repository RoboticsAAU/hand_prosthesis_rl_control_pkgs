from pathlib import Path
from typing import NamedTuple, Dict
import numpy as np
import rospkg

rospack = rospkg.RosPack()
hand_path = rospack.get_path("simulation_world") + "/urdf/hands"

class FreeRobotInfo(NamedTuple):
    path: str
    dof: int
    palm_name: str


def generate_free_robot_hand_info() -> Dict[str, FreeRobotInfo]:
    mia_hand_free_info = FreeRobotInfo(path=(hand_path + "/mia_hand_camera_launch.urdf.xacro"), dof=3,
                                          palm_name="palm_center")

    info_dict = dict(shadow_hand_free=mia_hand_free_info)
    return info_dict