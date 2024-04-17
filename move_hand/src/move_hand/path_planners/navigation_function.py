import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from move_hand.path_planners.path_visualiser import animate_path
from move_hand.path_planners.path_planner import PathPlanner
from typing import Dict, Any
from move_hand.path_planners.orientation_planners.interpolation import interpolate_rotation

# TODO: Check if goal/tart collide with obstacle. If so, raise error
class NavFuncPlanner(PathPlanner):
    
    def __init__(self, world_dim : int, world_sphere_rad : float) -> None:
        
        sp.init_printing()
        
        self.world = self.NavFuncWorld(world_dim, world_sphere_rad)        
        self.tmp_curr_pos = np.zeros((self.world.dimension,1), dtype=np.float64)

        self.path = np.empty((self.world.dimension,0), dtype=np.float64)
    
    
    def setup(self, num_rand_obs : int, obs_rad : float, goal_pos : np.array, kappa : float):
        # Set the parameters for the planner
        self.world.setup(num_rand_obs, obs_rad)
        self.goal_pos = goal_pos
        self.kappa = kappa
    
        # Sympy definitions
        self.nav_func = sp.Function('f')
        self.nabla_nav_func = sp.Function('df') # Gradient of nav function
        
        self.q = sp.zeros(self.world.dimension, 1) # Current agent position (for symbolic use)
        
        self._express_nav_func()
    
    
    def plan_path(self, start_pose : np.array, goal_pose : np.array, parameters : Dict[str, Any]) -> np.ndarray:
        # Reset the planner
        self._reset()
        
        # Setup the planner
        self.setup(parameters["num_rand_obs"], parameters["obs_rad"], goal_pose[:3], parameters["kappa"])

        # Compute the path
        self.compute_path(start_pose[:3], parameters["step_size"], plot=False)
        
        # Combine the rotation into the path
        rotation = interpolate_rotation(start_pose[3:], goal_pose[3:], self.path.shape[1])
        path = np.concatenate((self.path, rotation), axis=1)

        return path
    
    
    def compute_path(self, start_pos : np.array, step_size : float, plot : bool = True) -> None:
        
        self.tmp_curr_pos = start_pos.T.copy()
        
        if self.world.obstacle_num == 0:
            
            warnings.warn("No obstacles have been defined!", UserWarning)
        
        if start_pos.shape[0] != self.world.dimension or self.goal_pos.shape[0] != self.world.dimension:
            
            raise ValueError(f"Both start and goal positions must be {self.world.dimension}D vectors")
        
        for _ in tqdm(range(1000), desc="Computing path"):                
            # Computing navigation gradient vector
            grad_nav = self._compute_nav_grad(self.tmp_curr_pos)

            
            # Apply step
            step = grad_nav*step_size
            self.tmp_curr_pos -= step.copy()

            # Update path
            self.path = np.append(self.path, np.array([self.tmp_curr_pos]).T, axis=1)
            
            if np.isclose(self.tmp_curr_pos, self.goal_pos, atol=0.1).all():
                break
        
        if plot:
            if self.world.dimension == 2:
                self.plot_2d(start_pos, self.goal_pos)
            elif self.world.dimension == 3:
                self.plot_3d(start_pos, self.goal_pos)
    
    
    def _express_nav_func(self) -> None:

        # NavFuncWorld assumes sphere boundary is centered at origin (for simplicity)
        qw = sp.Matrix([0]*self.world.dimension)
            
        if self.world.dimension == 3:
            # Defining current pos
            q1, q2, q3 = sp.symbols('q1, q2, q3', real=True)            
            self.q = sp.Matrix([q1, q2, q3])
            
        else: # 2D
            q1, q2 = sp.symbols('q1, q2', real=True)
            self.q = sp.Matrix([q1, q2])
        
        d_qw = self.q - qw
        # Distance between agent and world (sphere) boundary
        beta_0 = -d_qw.dot(d_qw) + self.world.radius**2
        # Initial expression for repulsive function beta
        beta = beta_0

        # Completing expression for repulsive function, beta
        for idx in range(self.world.obstacle_num):
            obs = sp.Matrix(self.world.obstacles[idx,:].T)
            d_qo = self.q - obs
            # Dist between obstacles and agent
            beta_i = d_qo.dot(d_qo) - self.world.obstacle_rad**2
            beta *= beta_i.copy()
        
        qg = sp.Matrix(self.goal_pos.T)
        
        d_qg = qg - self.q
        
        # Expressing the attraction function
        gamma = sp.Pow(sp.sqrt(d_qg.dot(d_qg)), self.kappa)
        
        # Express final navigation function
        self.nav_func = d_qg.dot(d_qg) / sp.Pow(gamma + beta, 1.0/self.kappa)

        # Express gradient
        self.nabla_nav_func = sp.diff(self.nav_func, self.q)
        
        # print("Computed navigation function:")
        # sp.pprint(self.nav_func)
        # print("Computed gradient:")
        # sp.pprint(self.nabla_nav_func)
        
    
    def _compute_nav_grad(self, curr_pos : np.array, lambda_ : int = 1) -> np.array:
        
        if lambda_ <= 0:
            raise ValueError("Lambda must be a positive value")

        elif lambda_ != 1:
            raise NotImplementedError()
        
        if self.world.dimension == 3:
            nabla = self.nabla_nav_func.subs({self.q[0]: curr_pos[0], self.q[1]: curr_pos[1], self.q[2]: curr_pos[2]})
        else:
            nabla = self.nabla_nav_func.subs({self.q[0]: curr_pos[0], self.q[1]: curr_pos[1]})
        
        return np.array(nabla).astype(np.float64)[:,0].T
    
    
    def plot_2d(self, start_pos : np.array, goal_pos : np.array) -> None:
        # Create a figure
        fig, axs = plt.subplots()
        
        axs.grid(True)
        axs.set_aspect('equal')
        
        # Plot only x,y for simple visualisation
        plt.plot(self.path[0,:], self.path[1,:], 'b-', label='Path')
        
        # plot obstacles as circles
        for obstacle in self.world.obstacles:
            circle = plt.Circle(obstacle, self.world.obstacle_rad, facecolor='red', edgecolor='red')
            axs.add_artist(circle)
        
        # Plot world sphere
        world = plt.Circle((0,0), self.world.radius, fill=False, edgecolor='black')
        axs.add_artist(world)
        
        # Plot goal and start
        start = plt.Circle(start_pos, 0.2, facecolor='orange', edgecolor='orange')
        axs.add_artist(start)
        goal = plt.Circle(goal_pos, 0.2, facecolor='green', edgecolor='green')
        axs.add_artist(goal)
        
        plt.xlim(-self.world.radius, self.world.radius)
        plt.ylim(-self.world.radius, self.world.radius)
        
        # Label the axes
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Show the legend
        axs.legend()

        if self.world.dimension == 2:
            # Plot the cost function
            sp.plotting.plot3d(self.nav_func, (self.q[0], -10, 10), (self.q[1], -10, 10))
        
        # Show the plot
        plt.show()
    
    
    def plot_3d(self, start_pos : np.array, goal_pos : np.array) -> None:
        if self.world.dimension == 2:
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        ax.grid(True)
        
        # Plot path
        plt.plot(self.path[0,:], self.path[1,:], self.path[2,:], 'b-', label='Path')
        
        # plot obstacles as spheres
        for center in self.world.obstacles:
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = center[0] + self.world.obstacle_rad * np.outer(np.cos(u), np.sin(v))
            y = center[1] + self.world.obstacle_rad * np.outer(np.sin(u), np.sin(v))
            z = center[2] + self.world.obstacle_rad * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='red')
        
        # Plot goal and start
        ax.scatter(*start_pos, color='orange', label='Start')
        ax.scatter(*goal_pos, color='green', label='Goal')
        
        plt.show()
    
    
    def _reset(self):
        self.tmp_curr_pos = np.zeros((self.world.dimension,1), dtype=np.float64)
        self.path = np.empty((self.world.dimension,0), dtype=np.float64)
    
    # TODO: implement unique obstacle radii
    class NavFuncWorld():
        
        def __init__(self, world_dim : int = 3, world_sphere_rad : float = 5):
            if world_dim not in [2, 3]:
                raise ValueError("Specified world dimension not implemented! Must be 2 or 3.")
            
            # Set users configuration
            self.radius = world_sphere_rad # World radius
            self.dimension = world_dim # Dimension of world (e.g. 2D or 3D)
            
            # Other variables
            self.obstacles = np.empty((0, self.dimension), dtype=np.float64)
        
        
        def setup(self, num_rand_obs : int = 0, obs_rad : float = 0.5) -> None:
            self.obstacle_num = num_rand_obs
            self.obstacle_rad = obs_rad
            
            # Generate random obstacles if specified
            if num_rand_obs > 0:
                self._gen_rand_obs()


        def _gen_rand_obs(self) -> None:
            
            # Generate random obstacles. Based on https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            for _ in range(self.obstacle_num):
                phi = np.random.uniform(0,2*np.pi)
                costheta = np.random.uniform(-1,1)
                u = np.random.uniform(0,1)
                
                theta = np.arccos(costheta)
                r = (self.radius - self.obstacle_rad) * np.cbrt(u)
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                
                if self.dimension == 3:
                    z = r * np.cos(theta)  
                    self.obstacles = np.append(self.obstacles, np.array([[x,y,z]]), axis=0)
                else: # 2D
                    self.obstacles = np.append(self.obstacles, np.array([[x,y]]), axis=0)    
                
                
if __name__ == "__main__":
    # For testing
    np.random.seed(1)
    planner = NavFuncPlanner(world_dim=3, world_sphere_rad=10)
    planner.setup(num_rand_obs=5, obs_rad=0.2, goal_pos=np.array([0.0,7.5,3.0]), kappa=7)
    # planner.compute_path(start_pos=np.array([-1.0,-3.0, -3.0]), step_size=0.2, plot = True)
    planner.compute_path(start_pos=np.array([-1.0,-3.0,0.0]), step_size=0.3, plot=True)
    
    animate_path(planner.path, final_time=10)
    exit(0)
    