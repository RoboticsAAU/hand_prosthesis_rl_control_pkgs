import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def animate_path(path: np.ndarray, final_time : float, save_file : str = None):
    '''
    Animate a 3D path
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
