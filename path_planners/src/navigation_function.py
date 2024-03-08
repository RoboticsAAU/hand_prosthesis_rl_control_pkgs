import rospy
import numpy as np


class Fake3DWorld():
    
    def __init__(self, start_pos: np.array, goal_pos: np.array, world_sphere_rad: float, num_rand_obs : int = 0):
        
        # Set users configuration
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.num_obstacles = num_rand_obs
        
        # Generate random obstacles if specified
        if num_rand_obs > 0:
            obstacles = np.empty([num_rand_obs, 3], dtype=np.float32)
            
            # Generate random obstacles. Based on https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            for i in range(num_rand_obs):
                phi = np.random(0,2*np.pi)
                costheta = np.random(-1,1)
                u = np.random(0,1)

                theta = np.arccos(costheta)
                r = world_sphere_rad * np.cuberoot(u)
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                np.append(obstacles, np.array([x,y,z]), axis=0)
    
    def step():
        pass
        
    def reset():
        pass
    def render():
        pass


class NavPlanner():
    def __ini__(self):
        pass        
    