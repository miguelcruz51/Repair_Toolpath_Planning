"""Deposition simulation: animated preview of a trajectory being deposited.

Relocated from ``Simulation.py``. The three public functions keep their original
signatures and behaviour; only the interactive ``__main__`` block was dropped,
so every input (trajectory array, CSV path, material radius, frame interval)
arrives as a function argument instead of an ``input()`` prompt.

Inputs
    ``trajectory``: an ``(N, 3)`` array of 3D points, or a CSV file whose first
    row is a header and whose remaining rows hold X, Y, Z coordinates.

Outputs
    A Matplotlib 3D animation shown on screen (``simulate_deposition``,
    ``simulate_material_deposition``), or the parsed ``(N, 3)`` trajectory array
    (``import_trajectory_from_csv``).

Dependencies
    numpy, matplotlib.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

__all__ = [
    "simulate_deposition",
    "simulate_material_deposition",
    "import_trajectory_from_csv",
]


def simulate_deposition(trajectory, interval=0.01):
    """
    Simulates the deposition process in 3D space.

    Parameters:
        trajectory (ndarray): Array of 3D points representing the trajectory.
        interval (float): Time interval (in seconds) between each deposition step.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Deposition Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits
    x_lim = [np.min(trajectory[:, 0]), np.max(trajectory[:, 0])]
    y_lim = [np.min(trajectory[:, 1]), np.max(trajectory[:, 1])]
    z_lim = [np.min(trajectory[:, 2]), np.max(trajectory[:, 2])]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    # Incrementally simulate deposition
    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]

        # Plot the line segment
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="b")
        plt.draw()
        plt.pause(interval)  # Pause to simulate time delay

    plt.show()


def simulate_material_deposition(trajectory, material_radius=0.1, interval=0.05):
    """
    Simulates material deposition in 3D by adding spheres along the trajectory.

    Parameters:
        trajectory (ndarray): Array of 3D points representing the trajectory.
        material_radius (float): Radius of the deposited material at each point.
        interval (float): Time interval (in seconds) between each deposition step.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Material Deposition Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits
    x_lim = [np.min(trajectory[:, 0]), np.max(trajectory[:, 0])]
    y_lim = [np.min(trajectory[:, 1]), np.max(trajectory[:, 1])]
    z_lim = [np.min(trajectory[:, 2]), np.max(trajectory[:, 2])]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    # Plot spheres at each trajectory point to simulate material
    for point in trajectory:
        # Add a sphere at the current point
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = material_radius * np.cos(u) * np.sin(v) + point[0]
        y = material_radius * np.sin(u) * np.sin(v) + point[1]
        z = material_radius * np.cos(v) + point[2]
        ax.plot_surface(x, y, z, color='r', alpha=0.6)

        plt.draw()
        plt.pause(interval)  # Pause to simulate time delay

    plt.show()


def import_trajectory_from_csv(filename):
    """
    Imports a 3D trajectory from a CSV file.

    Parameters:
        filename (str): The name of the input CSV file.

    Returns:
        ndarray: Array of 3D points representing the trajectory.
    """
    trajectory = []
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            trajectory.append([float(coord) for coord in row])
    return np.array(trajectory)
