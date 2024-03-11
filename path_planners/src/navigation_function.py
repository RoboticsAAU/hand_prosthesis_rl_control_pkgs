import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


# TODO: Check if goal/tart collide with obstacle. If so, raise error
class NavFuncPlanner():
    
    def __init__(self, world_dim = int):
        
        sp.init_printing() 
        
        self.world = self.NavFuncWorld(world_dim=world_dim, world_sphere_rad=10, num_rand_obs=4, obs_rad=0.5)        
        self.tmp_curr_pos = np.zeros((self.world.dimension,1), dtype=np.float64)
        self.path = np.empty((self.world.dimension,0), dtype=np.float64)
        
        # Sympy definitions
        self.nav_func = sp.Function('f')
        self.nabla_nav_func = sp.Function('df') # Gradient of nav function
        
        self.q = sp.zeros(self.world.dimension, 1) # Current agent position (for symbolic use)
        self.qg = sp.zeros(self.world.dimension, 1) # Goal position (for symbolic use)
        self.k = sp.Symbol('k', real=True, positive=True) # Kappa
        
        self._express_nav_func()
    
    
    def compute_path(self, start_pos : np.array, goal_pos : np.array, kappa : float, step_size : float, plot : bool = True) -> None:
        
        self.tmp_curr_pos = start_pos.T.copy()
        
        if self.world.obstacle_num == 0:
            
            warnings.warn("No obstacles have been defined!", UserWarning)
        
        if start_pos.shape[0] != self.world.dimension or goal_pos.shape[0] != self.world.dimension:
            
            raise ValueError(f"Both start and goal positions must be {self.world.dimension}D vectors")
        
        #while not np.isclose(self.tmp_curr_pos, goal_pos, atol=1).all():
        for _ in tqdm(range(200), desc="Computing path"):                
            # Computing navigation gradient vector
            grad_nav = self._compute_nav_grad(self.tmp_curr_pos, goal_pos, kappa)
            print(grad_nav)
            print(self.tmp_curr_pos)
            print(goal_pos, flush=True)
            
            # Apply step
            step = grad_nav*step_size
            self.tmp_curr_pos += step[0] # Because it's nested numpy array

            # Update path
            self.path = np.append(self.path, np.array([self.tmp_curr_pos]).T, axis=1)
        
        if plot:
            self.plot_2d(start_pos, goal_pos)    
    
    
    def _express_nav_func(self) -> None:

        # NavFuncWorld assumes sphere boundary is centered at origin (for simplicity)
        qw = sp.Matrix([0]*self.world.dimension)
            
        if self.world.dimension == 3:
            # Defining current pos
            q1, q2, q3 = sp.symbols('q1, q2, q3', real=True)            
            self.q = sp.Matrix([q1, q2, q3])
            
            # Expressing goal
            qg1, qg2, qg3 = sp.symbols('qg1, qg2, qg3', real=True)
            self.qg = sp.Matrix([qg1, qg2, qg3])
            
        else: # 2D
            q1, q2 = sp.symbols('q1, q2', real=True)
            self.q = sp.Matrix([q1, q2])
            
            # Expressing goal
            qg1, qg2 = sp.symbols('qg1, qg2', real=True)
            self.qg = sp.Matrix([qg1, qg2])
        
        
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
        
        
        d_qg = self.qg - self.q
        
        # Expressing the attraction function
        gamma = sp.Pow(d_qg.dot(d_qg), 2.0*self.k) 
        
        # Express final navigation function
        # self.nav_func = sp.simplify(d_qg.dot(d_qg) / sp.Pow(gamma + beta, 1/self.k))
        self.nav_func = d_qg.dot(d_qg) / sp.Pow(gamma + beta, 1.0/self.k)
        
        # Express gradient
        # self.nabla_nav_func = sp.simplify(sp.diff(self.nav_func, self.q))
        self.nabla_nav_func = sp.diff(self.nav_func, self.q)
        
        # print("Computed navigation function:")
        # sp.pprint(self.nav_func)
        # print("Computed gradient:")
        # sp.pprint(self.nabla_nav_func)
        
        return
        
    
    def _compute_nav_grad(self, curr_pos : np.array, goal_pos : np.array, kappa : float, lambda_ : int = 1) -> np.array:
        
        if lambda_ <= 0:
            raise ValueError("Lambda must be a positive value")

        elif lambda_ != 1:
            raise NotImplementedError()
        
        if self.world.dimension == 3:
            nabla = self.nabla_nav_func.subs({self.q[0]: curr_pos[0], self.q[1]: curr_pos[1], self.q[2]: curr_pos[2], self.qg[0]: goal_pos[0], self.qg[1]: goal_pos[1], self.qg[2]: goal_pos[2], self.k: kappa})
        else:
            nabla = self.nabla_nav_func.subs({self.q[0]: curr_pos[0], self.q[1]: curr_pos[1], self.qg[0]: goal_pos[0], self.qg[1]: goal_pos[1], self.k: kappa})
        
        return np.array(nabla).astype(np.float64)
    
    
    def plot_2d(self, start_pos : np.array, goal_pos : np.array) -> None:
            
        # Create a figure
        fig, axs = plt.subplots()
        
        # Add grid and make axes equal
        axs.grid(True)
        axs.set_aspect('equal')
        
        # Plot only x,y for simple visualisation
        plt.plot(self.path[0,:], self.path[1,:], 'b-', label='Path')
        
        # plot obstacles as circles
        for obstacle in self.world.obstacles:
            circle = plt.Circle(obstacle, self.world.obstacle_rad, facecolor='red', edgecolor='red')
            axs.legend([circle], ['Obstacle'])
            axs.add_artist(circle)
        
        # Plot world sphere
        world = plt.Circle((0,0), self.world.radius, fill=False, edgecolor='black')
        axs.add_artist(world)
        axs.legend([world], ['World Boundary'])
        
        # Plot goal and start
        start = plt.Circle(start_pos, 0.2, facecolor='orange', edgecolor='orange')
        axs.add_artist(start)
        axs.legend([start], ['Start'])
        goal = plt.Circle(goal_pos, 0.2, facecolor='green', edgecolor='green')
        axs.add_artist(goal)
        axs.legend([goal], ['Goal'])
        
        plt.xlim(-self.world.radius, self.world.radius)
        plt.ylim(-self.world.radius, self.world.radius)
        
        # Label the axes
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Show the legend
        axs.legend()

        # Show the plot
        plt.show()
    
    
    # TODO: implement unique obstacle radii
    class NavFuncWorld():
        
        def __init__(self, world_sphere_rad: float, world_dim : int = 3, num_rand_obs : int = 0, obs_rad : float = 0.7):
            
            if world_dim not in [2, 3]:
                raise ValueError("Specified world dimension not implemented! Must be 2 or 3.")
            
            # Set users configuration        
            self.radius = world_sphere_rad # World radius
            self.dimension = world_dim # Dimension of world (e.g. 2D or 3D)
            self.obstacle_num = num_rand_obs # Number of random obstacles
            self.obstacle_rad = obs_rad# Radius of obstacles
            
            # Other variables
            self.obstacles = np.empty((0, self.dimension), dtype=np.float64)
            
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
    planner = NavFuncPlanner(world_dim=2)
    # planner.compute_path(start_pos=np.array([-1.0,-3.0, -3.0]), goal_pos=np.array([5.0,5.0,5.0]), kappa=5.0, step_size=5.0, plot = True)
    planner.compute_path(start_pos=np.array([-5.0,-1.0]), goal_pos=np.array([5.0,5.0]), kappa=5.0, step_size=15.0, plot = True)
    exit(0)
            






    