import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from typing import List

def animate_path(path: np.ndarray, final_time : float, save_file : str = None):
    '''
    Animate a 3D path (only positions, no orientation)
    :param path: Path as a sequence of points with shape (3, num_points)
    :param final_time: Duration of trajectory in seconds (DOES NOT WORK FOR SMALL TRAJECTORY TIMES)
    :param save_file: File to save the animation to if it should be saved
    :return:
    '''
    # Update the shape since the animation expects (num_points, 3)
    path = path.T
    
    def update_line(num, path, line):
        line.set_data(path[:num, :2].T)
        line.set_3d_properties(path[:num, 2])

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create lines initially without data
    line = ax.plot([], [], [])[0]

    # Setting the axes properties
    ax.set_xlim3d([np.min(path[:, 0]), np.max(path[:, 0])])
    ax.set_xlabel('X')

    ax.set_ylim3d([np.min(path[:, 1]), np.max(path[:, 1])])
    ax.set_ylabel('Y')

    ax.set_zlim3d([np.min(path[:, 2]), np.max(path[:, 2])])
    ax.set_zlabel('Z')

    # Creating the Animation object
    sample_rate = len(path) / final_time
    ani = FuncAnimation(
        fig, update_line, len(path), fargs=(path, line), interval=int(1000*(1/sample_rate)), blit=False)

    plt.show()
    
    if save_file:
        ani.save(save_file, writer="pillow")

def plot_path(path: np.ndarray, orientation_axes: List[bool] = [1, 1, 1]):
    '''
    Plot a 3D path (positions and orientations)
    :param path: Path as a sequence of points with shape (7, num_points)
    '''
    
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Setting the axes properties
    lim_min = min(np.min(path[i, :]) for i in range(3))
    lim_max = max(np.max(path[i, :]) for i in range(3))
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_zlim([lim_min, lim_max])
    
    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Extract position and orientation data
    positions = path[:3, :].T
    orientations = R.from_quat(path[3:, :].T).as_matrix()
    
    # Plot position
    ax.plot(*(positions.T),
            linestyle='-', marker='o', color='k')

    # Plot orientation axes
    axis_colors = ["red", "green", "blue"]
    for j, display_axis in enumerate(orientation_axes):
        if not display_axis:
            continue
        
        ax.quiver(*(positions.T), *(orientations[:, :, j].T),
            color=axis_colors[j],
            length=(lim_max-lim_min)/10.0,
            normalize=True
        )
    
    plt.show()
    
if __name__ == "__main__":
    # Example usage
    positions = np.array([[0, .1, .2, .3, .4, .5, .6, .7, .8, .9], 
                          [0, .1, .2, .3, .4, .5, .6, .7, .8, .9], 
                          [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]])
    orientations = np.array([0,0,0,1]).repeat(10).reshape(4, 10)
    
    path = np.concatenate((positions, orientations), axis=0)
    
    plot_path(path)