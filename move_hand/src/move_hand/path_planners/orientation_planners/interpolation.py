import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_rotation(start_quat : np.array, goal_quat : np.array, num_points : int):
    # Use scipy's Slerp to interpolate between two quaternions
    start_rot = R.from_quat(start_quat)
    goal_rot = R.from_quat(goal_quat)
    rots = R.concatenate([start_rot, goal_rot])
    slerp = Slerp([0, 1], rots)
    
    # Interpolate between the two quaternions
    t = np.linspace(0, 1, num_points)
    interp_rot = slerp(t)
    
    return interp_rot.as_quat().T
