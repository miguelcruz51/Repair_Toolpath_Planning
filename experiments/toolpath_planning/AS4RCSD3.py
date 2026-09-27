import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from shapely.affinity import rotate
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, Point
import trimesh
import pyvista as pv
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.fftpack import fft, ifft
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random
from scipy.spatial import cKDTree, distance
from scipy.interpolate import splprep, splev
from geomdl.fitting import approximate_curve
from geomdl import fitting
import csv
from shapely.ops import unary_union, nearest_points
import time
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
import scipy.interpolate as interp
from scipy.optimize import fsolve, curve_fit, least_squares, minimize
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
import sympy as sp
from scipy.interpolate import splprep, splev
from itertools import combinations
from matplotlib.collections import LineCollection

import matplotlib
matplotlib.use("TkAgg")  # or "QtAgg", depending on your system

import matplotlib.pyplot as plt



# --------------------- Slice -------------------------------

def slice_mesh(mesh, layer_height=0.5, add_layers=0):
    """
    Slices the mesh along the specified axis based on deposition layer height.
    Also allows adding extra layers with the same boundary as the penultimate layer.
    
    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        slicing_axis (str): The axis to slice along ('x', 'y', or 'z').
        layer_height (float): The layer thickness for additive manufacturing.
        add_layers (int): The number of extra layers to add.
    """

    axis_idx = 2
   
    # Get bounding box limits
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound, max_bound = bbox.min_bound[axis_idx], bbox.max_bound[axis_idx]
    
    # Compute number of slices based on layer height
    num_slices = int((max_bound - min_bound) / layer_height)
    print(f"Total slices: {num_slices}")
    print(f"Total layers: {num_slices - 1 + add_layers}")
    # Store slice cross-sections
    slices = []

    # Loop through each slice height and generate contours
    for i in range(num_slices):  # Avoid slicing the last layer
        slice_position = min_bound + i * layer_height  # Compute the slicing plane position
        # print(f"Generating slice at Z = {slice_position:.2f}...")
        # Create a slicing plane
        plane = np.zeros((4,))
        plane[axis_idx] = 1  # Normal to slicing axis
        plane[3] = -slice_position  # Plane equation

        # Compute the slice boundary for the mesh at the current height
        slice_boundary = extract_slice(mesh, plane)

        if slice_boundary is not None:
            slices.append((slice_position, slice_boundary))  # Append slice position and its contour

    print(f"Slicing complete. {len(slices)} valid cross-sections generated.")

    if add_layers > 0 and len(slices) > 1:
        print(f"Adding {add_layers} extra layers with the same boundary as the penultimate layer.")
        last_layer_position, last_layer_boundary = slices[-1]  # Penultimate layer
    
    # Start adding layers above the penultimate, spaced by layer height
    for i in range(add_layers):
        new_layer_position = last_layer_position + (i + 1) * layer_height  # Add layer height increment for each new layer
        slices.append((new_layer_position, last_layer_boundary))  # Add the duplicate boundary
    
    # Ask user if they want to view a specific layer or 3D visualization
    while True:
        user_input = input("Enter a layer number to view or 'q' to view all 3D layers: ").strip().lower()

        if user_input == 'q':
            plot_3d_layers(slices, 'z')  # Show all layers in 3D
            break
        else:
            try:
                layer_number = int(user_input)
                if 0 <= layer_number < len(slices):
                    plot_layer(slices[layer_number])  # Show specific layer
                else:
                    print("Invalid layer number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid layer number or 'q'.")

    

    return slices


def extract_slice(mesh, plane):
    """
    Extracts the slice of a mesh at a specific height using a slicing plane.
    This function performs the intersection of the mesh triangles with the slicing plane.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        plane (np.array): The slicing plane equation (a, b, c, d for ax + by + cz + d = 0).

    Returns:
        List: List of 2D points representing the slice boundary.
    """
    # Get the vertices and triangles from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Normalize the plane equation
    plane_normal = plane[:3]
    plane_d = plane[3]

    # Initialize the list to hold the intersection points
    intersection_points = []

    # Loop through each triangle in the mesh
    for triangle in triangles:
        # Get the vertices of the triangle
        v0, v1, v2 = vertices[triangle]
        
        # Calculate the signed distances of the triangle vertices from the plane
        d0 = np.dot(plane_normal, v0) + plane_d
        d1 = np.dot(plane_normal, v1) + plane_d
        d2 = np.dot(plane_normal, v2) + plane_d

        # Check for intersection (if any two points are on opposite sides of the plane)
        if (d0 * d1 < 0) or (d1 * d2 < 0) or (d2 * d0 < 0):
            # Calculate the intersection points on the edges of the triangle
            for i, (p1, p2) in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
                d1 = np.dot(plane_normal, p1) + plane_d
                d2 = np.dot(plane_normal, p2) + plane_d

                # If the two points are on opposite sides of the plane
                if d1 * d2 < 0:
                    # Compute the intersection point on this edge
                    t = -d1 / (d2 - d1)
                    intersection_point = p1 + t * (p2 - p1)
                    intersection_points.append(intersection_point)

    # Convert intersection points to 2D
    if len(intersection_points) == 0:
        print("No intersection points found.")
        return None  # No intersections detected, return None

    intersection_points = np.array(intersection_points)

    # Project points onto the 2D plane (remove the slicing axis)
    slicing_axis_idx = np.argmax(np.abs(plane_normal))
    
    if slicing_axis_idx == 0:
        # Remove x-axis
        intersection_points = intersection_points[:, [1, 2]]
    elif slicing_axis_idx == 1:
        # Remove y-axis
        intersection_points = intersection_points[:, [0, 2]]
    elif slicing_axis_idx == 2:
        # Remove z-axis
        intersection_points = intersection_points[:, [0, 1]]

    # Create a polygon from the 2D intersection points (Convex Hull)
    if len(intersection_points) > 2:
        try:
            hull = ConvexHull(intersection_points)
            hull_points = intersection_points[hull.vertices]
            return hull_points
        except:
            print("Error in computing convex hull.")
            return intersection_points
    else:
        print("Not enough intersection points to form a polygon.")
        return intersection_points
    
def plot_layer(slice_data):
    """
    Plots a single layer slice in 2D using Matplotlib.
    
    Parameters:
        slice_data (tuple): Tuple containing the slice position and 2D slice points.
    """
    slice_position, slice_points = slice_data
    plt.figure(figsize=(8, 6))
    plt.plot(slice_points[:, 0], slice_points[:, 1], label=f"Layer at {slice_position:.2f}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Slice at Z = {slice_position:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d_layers(slices, slicing_axis):
    """
    Plots all layers in 3D using Matplotlib with boundary lines.

    Parameters:
        slices (list): List of slices with layer positions and 2D points.
        slicing_axis (str): The axis along which the mesh was sliced.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for slice_position, slice_points in slices:
        if len(slice_points) > 1:
            slice_points = np.vstack([slice_points, slice_points[0]])  # Close the loop
            if slicing_axis == "x":
                ax.plot(np.full(len(slice_points), slice_position), slice_points[:, 0], slice_points[:, 1], label=f"Layer {slice_position:.2f}")
            elif slicing_axis == "y":
                ax.plot(slice_points[:, 0], np.full(len(slice_points), slice_position), slice_points[:, 1], label=f"Layer {slice_position:.2f}")
            elif slicing_axis == "z":
                ax.plot(slice_points[:, 0], slice_points[:, 1], np.full(len(slice_points), slice_position), label=f"Layer {slice_position:.2f}")

    # # Set axis labels with increased font size and padding
    # ax.set_xlabel('X-Axis (mm)', fontsize=18, labelpad=15)
    # ax.set_ylabel('Y-Axis (mm)', fontsize=18, labelpad=15)
    # ax.set_zlabel('Z-Axis (mm)', fontsize=18, labelpad=15)

    # # Set title with increased font size
    # ax.set_title("Model A", fontsize=20)

    # # Adjust tick label sizes
    # ax.tick_params(axis='both', labelsize=16)

    # # Optional: enable legend and set font size
    # ax.legend(fontsize=10)


    # Keep grid ON by leaving ticks but hiding labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Title only
    ax.set_title("Model C", fontsize=20)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig("3D Model slice.png") 
    plt.show()


# def rotate_mesh(mesh, axis, angle):
#     """Rotate mesh around a given axis (X, Y, or Z) by the specified angle in degrees."""
#     radians = np.deg2rad(angle)
#     axis_map = {"x": (radians, 0, 0), "y": (0, radians, 0), "z": (0, 0, radians)}

#     if axis in axis_map:
#         R = mesh.get_rotation_matrix_from_xyz(axis_map[axis])
#         mesh.rotate(R, center=mesh.get_center())
#     else:
#         print("Invalid axis. No rotation applied.")

#     return mesh

def rotate_mesh(mesh, axis, angle, center=None):
    """Rotate mesh around a given axis (X, Y, or Z) by the specified angle in degrees."""
    radians = np.deg2rad(angle)
    axis_map = {"x": (radians, 0, 0), "y": (0, radians, 0), "z": (0, 0, radians)}

    if axis in axis_map:
        R = mesh.get_rotation_matrix_from_xyz(axis_map[axis])
        mesh.rotate(R, center=center if center is not None else mesh.get_center())
    else:
        print("Invalid axis. No rotation applied.")

    return mesh

# ----------------------- Strategies ------------------------------
def generate_raster_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle):
    """
    Generates a raster toolpath for a single boundary with optional rotation for cross-scanning.
    Adds fly-in and fly-out segments if fly_in_out is True.
    """

    # Ensure boundary points are in 2D array format
    boundary_points = np.asarray(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    try:
        slice_polygon = Polygon(boundary_points)
    except ValueError as e:
        print(f"Error creating polygon: {e}")
        return np.array([]), np.array([])

    # Determine the rotation angle based on the layer index
    effective_angle = raster_angle if layer_index % 2 != 0 else 0

    # Rotate the polygon by the effective_angle
    rotated_polygon = rotate(slice_polygon, angle=effective_angle, origin='center', use_radians=False)

    # Compute the bounding box and generate raster lines
    min_x, min_y, max_x, max_y = rotated_polygon.bounds

    deposition_lines = []
    tool_moves = []
    fly_lines = []  # Fly-in and fly-out lines
    y = min_y
    while y <= max_y:
        line_start = (min_x, y)
        line_end = (max_x, y)
        line = LineString([line_start, line_end])
        clipped_line = rotated_polygon.intersection(line)

        if not clipped_line.is_empty:
            if isinstance(clipped_line, LineString):
                coords = np.array(clipped_line.coords)
                if np.allclose(coords[:, 1], coords[0, 1]):  # Ensure y-values are the same (horizontal line)
                    deposition_lines.append(coords)  
            elif isinstance(clipped_line, MultiLineString):
                for subline in clipped_line.geoms:
                    coords = np.array(subline.coords)
                    if np.allclose(coords[:, 1], coords[0, 1]):  
                        deposition_lines.append(coords)

        y += hatch_distance

    # Fly-in and fly-out logic based on the flag
    if fly_in_out:
        # Add fly-in, fly-out, and diagonal linking
        for i, line in enumerate(deposition_lines):
            start_point = line[0]
            end_point = line[-1]

            # Calculate direction vectors
            start_dir = (line[1] - start_point) / np.linalg.norm(line[1] - start_point)
            end_dir = (end_point - line[-2]) / np.linalg.norm(end_point - line[-2])

            # Add fly-in and fly-out points
            fly_in_point = start_point - fly_distance * start_dir
            fly_out_point = end_point + fly_distance * end_dir
            fly_lines.append(np.array([fly_in_point, start_point]))
            fly_lines.append(np.array([end_point, fly_out_point]))

            # Add diagonal tool move between lines
            if i < len(deposition_lines) - 1:  # Ensure not the last line
                current_line = deposition_lines[i]
                next_line = deposition_lines[i + 1]

                fly_out_point = current_line[-1] + fly_distance * (current_line[-1] - current_line[0]) / np.linalg.norm(current_line[-1] - current_line[0])
                fly_in_point = next_line[0] - fly_distance * (next_line[1] - next_line[0]) / np.linalg.norm(next_line[1] - next_line[0])

                tool_moves.append(np.array([fly_out_point, fly_in_point]))
    else:
        # Add simple straight movements between consecutive deposition lines
        for i in range(len(deposition_lines) - 1):
            tool_moves.append(np.array([deposition_lines[i][-1], deposition_lines[i + 1][0]]))

    # Combine deposition lines and tool moves into separate arrays
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # No tool moves for zig-zag pattern

    # Rotate the toolpath back to the original orientation
    if effective_angle != 0:
        deposition_array = rotate_points(deposition_array, -effective_angle, slice_polygon.centroid)
        tool_moves_array = rotate_points(tool_moves_array, -effective_angle, slice_polygon.centroid)
        fly_lines_array = rotate_points(fly_lines_array, -effective_angle, slice_polygon.centroid)

    # 2D Visualization of Deposition and Travel Paths
    plt.figure(figsize=(8, 8))

    # Plot deposition lines (as connected path)
    if deposition_array.size > 0:
        plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # Plot connecting travel moves
    if tool_moves_array.size > 0:
        for i in range(0, len(tool_moves_array), 2):
            segment = tool_moves_array[i:i+2]
            if len(segment) == 2:
                plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # Plot fly-in/fly-out lines
    # if fly_lines_array.size > 0:
    #     for i in range(0, len(fly_lines_array), 2):
    #         segment = fly_lines_array[i:i+2]
    #         if len(segment) == 2:
    #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # Optional: plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')

    plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

def generate_zigzag_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a zig-zag raster toolpath with optional fly-in and fly-out.
    Includes tangent corner smoothing with adjustable fillet radius.
    """
    # Ensure boundary points are in 2D array format
    boundary_points = np.asarray(boundary_points)
    # print(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Determine the rotation angle based on the layer index
    effective_angle = raster_angle if layer_index % 2 != 0 else 0

    # Rotate the polygon by the effective_angle
    rotated_polygon = rotate(slice_polygon, angle=effective_angle, origin='center', use_radians=False)

   

    # Compute the bounding box and generate raster lines
    min_x, min_y, max_x, max_y = rotated_polygon.bounds

    deposition_lines = []
    y = min_y
    while y <= max_y:
        line_start = (min_x, y)
        line_end = (max_x, y)
        line = LineString([line_start, line_end])
        clipped_line = rotated_polygon.intersection(line)

        if not clipped_line.is_empty:
            if isinstance(clipped_line, LineString):
                deposition_lines.append(np.array(clipped_line.coords))
            elif isinstance(clipped_line, MultiLineString):
                for subline in clipped_line.geoms:
                    deposition_lines.append(np.array(subline.coords))

        y += hatch_distance

    # Generate the zig-zag toolpath
    fillets = []
    zig_zag_lines = []

    for i, line in enumerate(deposition_lines):
        if i % 2 == 1:  # Reverse direction for every other line
            line = line[::-1]

        # Add straight lines or fillets between racks
        if i > 0:
            prev_end = zig_zag_lines[-1][-1]
            curr_start = line[0]

            if corner_radius > 0:
                # Add fillet arcs and connecting line
                arc1, connecting_line, arc2 = create_fillet_between_racks(prev_end, curr_start, corner_radius)
                fillets.append(arc1)
                fillets.append(arc2)
                zig_zag_lines.append(connecting_line)
            else:
                # Straight connection
                straight_connection = np.array([prev_end, curr_start])
                zig_zag_lines.append(straight_connection)

        zig_zag_lines.append(line)
        

    # Add fly-in and fly-out only for the starting and ending points
    fly_lines = []
    if fly_in_out and zig_zag_lines:
        start_point = zig_zag_lines[0][0]
        end_point = zig_zag_lines[-1][-1]

        # Calculate direction vectors for fly-in and fly-out
        start_dir = (zig_zag_lines[0][1] - start_point) / np.linalg.norm(zig_zag_lines[0][1] - start_point)
        end_dir = (end_point - zig_zag_lines[-1][-2]) / np.linalg.norm(end_point - zig_zag_lines[-1][-2])

        fly_in_point = start_point - fly_distance * start_dir
        fly_out_point = end_point + fly_distance * end_dir

        fly_lines.append(np.array([fly_in_point, start_point]))
        fly_lines.append(np.array([end_point, fly_out_point]))

    # Combine results
    deposition_array = np.vstack(zig_zag_lines) if zig_zag_lines else np.array([])
    tool_moves_array = np.array([])  # No tool moves for zig-zag pattern
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.vstack(fillets) if fillets else np.array([])

    # # 2D Visualization of Deposition and Travel Paths
    # plt.figure(figsize=(8, 8))

    # # Plot deposition lines (as connected path)
    # if deposition_array.size > 0:
    #     plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # # Plot connecting travel moves
    # if tool_moves_array.size > 0:
    #     for i in range(0, len(tool_moves_array), 2):
    #         segment = tool_moves_array[i:i+2]
    #         if len(segment) == 2:
    #             plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # # Plot fly-in/fly-out lines
    # # if fly_lines_array.size > 0:
    # #     for i in range(0, len(fly_lines_array), 2):
    # #         segment = fly_lines_array[i:i+2]
    # #         if len(segment) == 2:
    # #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # # Optional: plot boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')

    # plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()


    return deposition_array,tool_moves_array, fly_lines_array, fillets_array

def generate_contour_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a contour toolpath by offsetting the polygon inward multiple times.
    Ensures that each rack (contour) is closed, and adds clean tool moves between contours.
    """

    boundary_points = np.asarray(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    slice_polygon = Polygon(boundary_points)

    if not slice_polygon.is_valid:
        print(f"Invalid polygon detected: {slice_polygon}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([])

    deposition_lines = []
    current_polygon = slice_polygon
    while current_polygon.area > 0:
        if isinstance(current_polygon, MultiPolygon):
            current_polygon = max(current_polygon.geoms, key=lambda p: p.area)

        coords = np.array(current_polygon.exterior.coords)

        # Ensure each contour is closed
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        deposition_lines.append(coords)

        current_polygon = current_polygon.buffer(-hatch_distance)
        if current_polygon.is_empty or not current_polygon.is_valid:
            break

    tool_moves = []
    for i in range(len(deposition_lines) - 1):
        end_point = deposition_lines[i][-1]
        start_point = deposition_lines[i + 1][0]
        tool_moves.append(np.array([end_point, start_point]))

    # Optional: Close the loop between the last and first contours
    if len(deposition_lines) > 1:
        last_end_point = deposition_lines[-1][-1]
        first_start_point = deposition_lines[0][0]
        tool_moves.append(np.array([last_end_point, first_start_point]))

    # Combine deposition lines and tool moves into arrays
    deposition_array = np.concatenate(deposition_lines) if deposition_lines else np.array([])
    tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
    fly_lines_array = np.array([])
    fillets_array = np.array([])

    # 2D Visualization of Deposition and Travel Paths
    plt.figure(figsize=(8, 8))

    # Plot deposition lines (as connected path)
    if deposition_array.size > 0:
        plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # Plot connecting travel moves
    if tool_moves_array.size > 0:
        for i in range(0, len(tool_moves_array), 2):
            segment = tool_moves_array[i:i+2]
            if len(segment) == 2:
                plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # Plot fly-in/fly-out lines
    # if fly_lines_array.size > 0:
    #     for i in range(0, len(fly_lines_array), 2):
    #         segment = fly_lines_array[i:i+2]
    #         if len(segment) == 2:
    #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # Optional: plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', label='Boundary')

    plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

def generate_spiral_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a continuous spiral toolpath that follows the boundary geometry inward toward the center.
    The tool moves in a continuous inward spiral while performing deposition along the path.
    """
    boundary_points = np.asarray(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    slice_polygon = Polygon(boundary_points)

    if not slice_polygon.is_valid:
        print(f"Invalid polygon detected: {slice_polygon}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([])

    # Initialize the spiral path
    spiral_points = []
    current_polygon = slice_polygon
    max_radius = current_polygon.area ** 0.5  # Rough maximum extent
    angle_step = 0.1  # Adjust for smoothness

    # Generate spiral by offsetting inward
    while current_polygon.area > 0:
        if isinstance(current_polygon, Polygon):
            coords = np.array(current_polygon.exterior.coords)
        else:
            # If it's a MultiPolygon, take the largest valid part
            coords = np.array(max(current_polygon.geoms, key=lambda p: p.area).exterior.coords)

        # Sample points along the current boundary
        for i in range(len(coords) - 1):
            # Interpolate between consecutive points to create smooth transitions
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            num_steps = max(int(np.linalg.norm([x2 - x1, y2 - y1]) / hatch_distance), 1)
            for t in np.linspace(0, 1, num_steps):
                spiral_points.append([(1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2])

        # Offset inward for the next spiral loop
        current_polygon = current_polygon.buffer(-hatch_distance)
        if current_polygon.is_empty or not current_polygon.is_valid:
            break

    # Convert to NumPy array
    spiral_points = np.array(spiral_points)

    # Tool moves between the spiral points
    tool_moves = []
    for i in range(len(spiral_points) - 1):
        tool_moves.append(np.array([spiral_points[i], spiral_points[i + 1]]))

    # Optional: Add fly-in moves if required
    fly_lines_array = np.array([])
    if fly_in_out:
        # Fly-in moves (from a specified location to the start of the spiral)
        start_point = spiral_points[0]
        fly_lines_array = np.array([np.array([boundary_points[0], start_point])])

    # Combine deposition lines and tool moves into arrays
    deposition_array = spiral_points
    tool_moves_array = np.array([])
    fillets_array = np.array([])

    # 2D Visualization of Deposition and Travel Paths
    plt.figure(figsize=(8, 8))

    # Plot deposition lines (as connected path)
    if deposition_array.size > 0:
        plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # Plot connecting travel moves
    if tool_moves_array.size > 0:
        for i in range(0, len(tool_moves_array), 2):
            segment = tool_moves_array[i:i+2]
            if len(segment) == 2:
                plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # Plot fly-in/fly-out lines
    # if fly_lines_array.size > 0:
    #     for i in range(0, len(fly_lines_array), 2):
    #         segment = fly_lines_array[i:i+2]
    #         if len(segment) == 2:
    #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # Optional: plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', label='Boundary')

    plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

def generate_hybrid1_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a hybrid toolpath combining contour deposition and zig-zag filling.
    """

    # Ensure boundary is in 2D array format
    boundary_points = np.asarray(boundary_points)
    if boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must have shape (N, 2).")

    # Ensure boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon and validate
    slice_polygon = Polygon(boundary_points)
    if not slice_polygon.is_valid:
        slice_polygon = slice_polygon.buffer(0)  # Attempt to fix self-intersections
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([]), np.array([])

    ### Step 1: Generate Contour Deposition ###
    # Offset the boundary inward by half the tool width to create a contour path
    contour_polygon = slice_polygon.buffer(-tool_width / 2)
    if contour_polygon.is_empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Get contour deposition points
    contour_points = np.array(contour_polygon.exterior.coords)
    contour_deposition = [contour_points]

    ### Step 2: Generate Zig-Zag Fill ###
    # Further offset to define the raster area
    fill_polygon = contour_polygon.buffer(-tool_width)
    if fill_polygon.is_empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Rotate the polygon for zig-zag alignment
    rotated_fill_polygon = rotate(fill_polygon, raster_angle, origin='center')
    min_x, min_y, max_x, max_y = rotated_fill_polygon.bounds

    # Generate raster lines
    zigzag_lines = []
    tool_moves = []
    y = min_y

    while y <= max_y:
        raster_line = LineString([(min_x, y), (max_x, y)])
        clipped = rotated_fill_polygon.intersection(raster_line)

        if isinstance(clipped, LineString):
            zigzag_lines.append(np.array(clipped.coords))
        elif isinstance(clipped, MultiLineString):
            for subline in clipped.geoms:
                zigzag_lines.append(np.array(subline.coords))

        y += hatch_distance

    # Reverse alternate lines for proper zigzag movement
    for i in range(len(zigzag_lines)):
        if i % 2 == 1:
            zigzag_lines[i] = zigzag_lines[i][::-1]

        # Add tool move between adjacent lines
        if i > 0:
            tool_moves.append(np.array([zigzag_lines[i - 1][-1], zigzag_lines[i][0]]))

    ### Step 3: Connect Contour to Zig-Zag (Tool Move) ###
    if zigzag_lines:
        tool_moves.insert(0, np.array([contour_deposition[-1][-1], zigzag_lines[0][0]]))

    ### Step 4: Fly-in and Fly-out ###
    fly_lines = []
    if fly_in_out and len(contour_deposition) > 0:
        start_point = contour_deposition[0][0]
        end_point = zigzag_lines[-1][-1] if zigzag_lines else contour_deposition[-1][-1]

        # Fly-in
        fly_lines.append(np.array([start_point - [fly_distance, 0], start_point]))

        # Fly-out
        fly_lines.append(np.array([end_point, end_point + [fly_distance, 0]]))

    ### Finalize Arrays ###
    deposition_array = np.vstack(contour_deposition + zigzag_lines) if contour_deposition or zigzag_lines else np.array([])
    tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # No fillets in hybrid path


    # 2D Visualization of Deposition and Travel Paths
    plt.figure(figsize=(8, 8))

    # Plot deposition lines (as connected path)
    if deposition_array.size > 0:
        plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # Plot connecting travel moves
    if tool_moves_array.size > 0:
        for i in range(0, len(tool_moves_array), 2):
            segment = tool_moves_array[i:i+2]
            if len(segment) == 2:
                plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # Plot fly-in/fly-out lines
    # if fly_lines_array.size > 0:
    #     for i in range(0, len(fly_lines_array), 2):
    #         segment = fly_lines_array[i:i+2]
    #         if len(segment) == 2:
    #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # Optional: plot boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')

    plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    return deposition_array, tool_moves_array, fly_lines_array, fillets_array


# def generate_hybrid2_toolpath(boundary_points, damage_contour, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius):
    """
    Generates a zigzag toolpath with polynomial raster lines, offset within the boundary.
    """

    # Step 1: Validate and Process Boundary Points
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must contain at least two 2D coordinates.")
    
    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Convert damage_contour to list of (x, y, z) points (assuming z = 0)
    damage_contour_3d = [(x, y, 0) for x, y in damage_contour]

    # Define the degree of the NURBS curve
    degree = 3  # Cubic NURBS for smoothness

    # Fit a NURBS curve to the damage contour
    nurbs_curve = approximate_curve(damage_contour_3d, degree)

    # Sample the NURBS curve to generate smooth points
    num_samples = 1000
    sampled_params = np.linspace(0, 1, num_samples)
    ransac_contour = np.array([nurbs_curve.evaluate_single(t)[:2] for t in sampled_params])  # Keep only x, y

   

    # # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    # Plot nurb curve
    plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], label="NURBS Curve", color="red", alpha=0.7)
    # Plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Perpendicular Linking Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 3: Compute Deposition Lines
    deposition_lines = []
    connecting_lines = []
    offset = 0
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()

    # Use a flag to alternate direction
    direction_flag = True

    while offset <= y_range:
        offset_points = []

        # Generate offset points based on the fitted NURBS curve
        for x_value in ransac_contour[:, 0]:
            y_value = np.polyval(np.polyfit(ransac_contour[:, 0], ransac_contour[:, 1], deg=2), x_value) + offset
            offset_points.append((x_value, y_value))

        # Convert to LineString and clip to the polygon
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        if not clipped_curve.is_empty:
            if isinstance(clipped_curve, LineString):
                line = np.array(clipped_curve.coords)
            elif isinstance(clipped_curve, MultiLineString):
                # Use only the first sub-curve (you can adjust this logic if needed)
                line = np.array(clipped_curve.geoms[0].coords)
            
            # Reverse the direction of the line if needed
            if not direction_flag:
                line = line[::-1]
            
            deposition_lines.append(line)
            direction_flag = not direction_flag  # Toggle the direction

        offset += hatch_distance

    # Step 4: Add Perpendicular Connecting Lines
    for i in range(len(deposition_lines) - 1):
        current_line = deposition_lines[i]
        next_line = deposition_lines[i + 1]

        if direction_flag:  # Connection from the end of the current to the start of the next
            connect_start = np.array([current_line[-1], next_line[0]])
        else:  # Connection from the start of the current to the end of the next
            connect_start = np.array([current_line[0], next_line[-1]])

        connecting_lines.append(connect_start)

    fly_lines = []
    if fly_in_out and deposition_lines:
        # Fly-In
        start_point = deposition_lines[0][0]
        start_dir = (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
        fly_in_point = start_point - fly_distance * start_dir
        fly_lines.append(np.array([fly_in_point, start_point]))

        # Fly-Out
        end_point = deposition_lines[-1][-1]
        end_dir = (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        fly_out_point = end_point + fly_distance * end_dir
        fly_lines.append(np.array([end_point, fly_out_point]))

    # # # Debug: Plot the Toolpath
    # plt.figure(figsize=(8, 8))
    # # Plot deposition lines
    # for line in deposition_lines:
    #     plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="blue", alpha=0.7)
    # # Plot connecting lines
    # for line in connecting_lines:
    #     plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    # # Plot boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    # plt.legend()
    # plt.title("Hybrid Toolpath with Perpendicular Linking Lines")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.grid()
    # plt.show()

    # Debug Visualization for Contour and Offsets
    plt.figure(figsize=(8, 8))
    
    # Plot original boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    
    # Plot the polynomial fitted contour
    plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], 'r--', label="Fitted Contour")
    
    # Plot the offset contours
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], 'b-', alpha=0.6, label="Offset Contour" if 'Offset Contour' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend()
    plt.title("Contour and Offset Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    connecting_array = np.array([])
    return deposition_array, connecting_array, fly_lines_array, fillets_array

# def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a zigzag toolpath with polynomial raster lines, offset within the boundary.
    """

    print('Number of points: ', len(non_planar_boundary_points))

    # Step 1: Validate and Process Boundary Points
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must contain at least two 2D coordinates.")
    
    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)
    
   # Validate non_planar_boundary_points
    if non_planar_boundary_points is None or len(non_planar_boundary_points) == 0:
        print("[ERROR] non_planar_boundary_points is empty!")
        return np.array([]), np.array([]), np.array([]), np.array([])  # Return valid empty outputs

    if non_planar_boundary_points.shape[1] != 2:
        print("[ERROR] Invalid shape for non_planar_boundary_points:", non_planar_boundary_points.shape)
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = non_planar_boundary_points[:, 0]
    y = non_planar_boundary_points[:, 1]

    if len(x) < 3:
        print("[ERROR] Not enough points for polynomial fitting (need at least 3, got", len(x), ")")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Try different polynomial degrees
    best_degree = 2
    best_aic = float('inf')

    for deg in range(1, min(3, len(x))):  # Avoid overfitting with high degrees
        coeffs = np.polyfit(x, y, deg)
        y_pred = np.polyval(coeffs, x)
        aic = compute_aic(y, y_pred, deg, len(x))

        if aic < best_aic:
            best_aic = aic
            best_degree = deg
            best_coeffs = coeffs

    poly_coeffs = best_coeffs  # Use best coefficients found
    print(f"Selected Polynomial Degree: {best_degree}")
    
    # poly_coeffs = np.polyfit(x, y, deg=best_degree)

    # Generate smooth polynomial points
    fitted_x = np.linspace(boundary_points[:, 0].min(), boundary_points[:, 0].max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    ransac_contour = np.column_stack((fitted_x, fitted_y))

    # Step 3: Compute Deposition Lines
    deposition_lines = []
    connecting_lines = []
    offset = 0
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()

    # Use a flag to alternate direction
    direction_flag = True

    while offset <= y_range:
        offset_points = []

        # Generate offset points based on the fitted polynomial
        for x_value in fitted_x:
            y_value = np.polyval(poly_coeffs, x_value) + offset
            offset_points.append((x_value, y_value))

        # Convert to LineString and clip to the polygon
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        if not clipped_curve.is_empty:
            if isinstance(clipped_curve, LineString):
                line = np.array(clipped_curve.coords)
            elif isinstance(clipped_curve, MultiLineString):
                # Use only the first sub-curve (you can adjust this logic if needed)
                line = np.array(clipped_curve.geoms[0].coords)
            
            # Reverse the direction of the line if needed
            if not direction_flag:
                line = line[::-1]
            
            deposition_lines.append(line)
            direction_flag = not direction_flag  # Toggle the direction

        offset += hatch_distance
    # print(offset, "<=", y_range)

    # Step 4: Add Perpendicular Connecting Lines
    for i in range(len(deposition_lines) - 1):
        current_line = deposition_lines[i]
        next_line = deposition_lines[i + 1]

        if direction_flag:  # Connection from the end of the current to the start of the next
            connect_start = np.array([current_line[-1], next_line[0]])
        else:  # Connection from the start of the current to the end of the next
            connect_start = np.array([current_line[0], next_line[-1]])

        connecting_lines.append(connect_start)

    
    fly_lines = []
    if fly_in_out and deposition_lines:
        # Fly-In
        start_point = deposition_lines[0][0]
        start_dir = (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
        fly_in_point = start_point - fly_distance * start_dir
        fly_lines.append(np.array([fly_in_point, start_point]))

        # Fly-Out
        end_point = deposition_lines[-1][-1]
        end_dir = (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        fly_out_point = end_point + fly_distance * end_dir
        fly_lines.append(np.array([end_point, fly_out_point]))
    
    
    # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    # Plot deposition lines
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="blue", alpha=0.7)
    # Plot connecting lines
    for line in connecting_lines:
        plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Perpendicular Linking Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # # Debug Visualization for Contour and Offsets
    # plt.figure(figsize=(8, 8))
    
    # # Plot original boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    
    # # Plot the polynomial fitted contour
    # plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], 'r--', label="Fitted Contour")
    
    # # Plot the offset contours
    # for line in deposition_lines:
    #     plt.plot(line[:, 0], line[:, 1], 'b-', alpha=0.6, label="Offset Contour" if 'Offset Contour' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # plt.legend()
    # plt.title("Contour and Offset Visualization")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.grid()
    # plt.show()

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    connecting_array = np.array([])
    return deposition_array, connecting_array, fly_lines_array, fillets_array

# def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a zigzag toolpath with polynomial raster lines, offset within the boundary.
    """

    print('Number of points: ', len(non_planar_boundary_points))

    # Step 1: Validate and Process Boundary Points
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must contain at least two 2D coordinates.")
    
    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Validate non_planar_boundary_points
    if non_planar_boundary_points is None or len(non_planar_boundary_points) == 0:
        print("[ERROR] non_planar_boundary_points is empty!")
        return np.array([]), np.array([]), np.array([]), np.array([])

    if non_planar_boundary_points.shape[1] != 2:
        print("[ERROR] Invalid shape for non_planar_boundary_points:", non_planar_boundary_points.shape)
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Step 2: Use the Given filter_points_ransac() Function for Fitting
    filtered_points, _ = filter_points_ransac(non_planar_boundary_points, degree=2, residual_threshold=5.0, visualize=False)
    
    if len(filtered_points) < 3:
        print("[ERROR] Not enough inlier points for polynomial fitting.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    # Fit a polynomial using only the given function
    poly_coeffs = np.polyfit(x, y, deg=2)
    print("Fitted Polynomial Coefficients:", poly_coeffs)

    # Generate smooth polynomial points
    fitted_x = np.linspace(boundary_points[:, 0].min(), boundary_points[:, 0].max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    ransac_contour = np.column_stack((fitted_x, fitted_y))

    # Step 3: Compute Deposition Lines
    deposition_lines = []
    connecting_lines = []
    offset = 0
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()
    direction_flag = True

    while offset <= y_range:
        offset_points = []

        # Generate offset points based on the fitted polynomial
        for x_value in fitted_x:
            y_value = np.polyval(poly_coeffs, x_value) + offset
            offset_points.append((x_value, y_value))

        # Clip to the polygon
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        if not clipped_curve.is_empty:
            if isinstance(clipped_curve, LineString):
                line = np.array(clipped_curve.coords)
            elif isinstance(clipped_curve, MultiLineString):
                # Select the longest valid segment
                valid_segments = sorted(clipped_curve.geoms, key=lambda g: g.length, reverse=True)
                line = np.array(valid_segments[0].coords)
            
            # Reverse the direction of the line if needed
            if not direction_flag:
                line = line[::-1]

            deposition_lines.append(line)
            direction_flag = not direction_flag  # Toggle the direction

        offset += hatch_distance
    # print(offset, "<=", y_range)

    # Step 4: Add Perpendicular Connecting Lines
    for i in range(len(deposition_lines) - 1):
        current_line = deposition_lines[i]
        next_line = deposition_lines[i + 1]

        if direction_flag:  # Connection from the end of the current to the start of the next
            connect_start = np.array([current_line[-1], next_line[0]])
        else:  # Connection from the start of the current to the end of the next
            connect_start = np.array([current_line[0], next_line[-1]])

        connecting_lines.append(connect_start)

    
    fly_lines = []
    if fly_in_out and deposition_lines:
        # Fly-In
        start_point = deposition_lines[0][0]
        start_dir = (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
        fly_in_point = start_point - fly_distance * start_dir
        fly_lines.append(np.array([fly_in_point, start_point]))

        # Fly-Out
        end_point = deposition_lines[-1][-1]
        end_dir = (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        fly_out_point = end_point + fly_distance * end_dir
        fly_lines.append(np.array([end_point, fly_out_point]))
    
    
    # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    # Plot deposition lines
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="blue", alpha=0.7)
    # Plot connecting lines
    for line in connecting_lines:
        plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Perpendicular Linking Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # # Debug Visualization for Contour and Offsets
    plt.figure(figsize=(8, 8))
    
    # Plot original boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    
    # Plot the polynomial fitted contour
    plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], 'r--', label="Fitted Contour")
    
    # Plot the offset contours
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], 'b-', alpha=0.6, label="Offset Contour" if 'Offset Contour' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend()
    plt.title("Contour and Offset Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    connecting_array = np.array([])
    return deposition_array, connecting_array, fly_lines_array, fillets_array

# def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a zigzag toolpath with polynomial raster lines, offset within the boundary.
    """

    print('Number of points: ', len(non_planar_boundary_points))

    # Step 1: Validate and Process Boundary Points
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must contain at least two 2D coordinates.")
    
    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Validate non_planar_boundary_points
    if non_planar_boundary_points is None or len(non_planar_boundary_points) == 0:
        print("[ERROR] non_planar_boundary_points is empty!")
        return np.array([]), np.array([]), np.array([]), np.array([])

    if non_planar_boundary_points.shape[1] != 2:
        print("[ERROR] Invalid shape for non_planar_boundary_points:", non_planar_boundary_points.shape)
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Step 2: Use the Given filter_points_ransac() Function for Fitting
    filtered_points, _ = filter_points_ransac(non_planar_boundary_points, degree=2, residual_threshold=5.0, visualize=False)
    
    if len(filtered_points) < 3:
        print("[ERROR] Not enough inlier points for polynomial fitting.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    # Fit a polynomial using only the given function
    poly_coeffs = np.polyfit(x, y, deg=2)
    print("Fitted Polynomial Coefficients:", poly_coeffs)

    # Generate smooth polynomial points
    fitted_x = np.linspace(boundary_points[:, 0].min(), boundary_points[:, 0].max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    ransac_contour = np.column_stack((fitted_x, fitted_y))

    # Step 3: Compute Deposition Lines
    deposition_lines = []
    connecting_lines = []
    offset = 0
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()
    direction_flag = True

    while offset <= y_range:
        offset_points = []

        # Generate offset points based on the fitted polynomial
        for x_value in fitted_x:
            y_value = np.polyval(poly_coeffs, x_value) + offset
            offset_points.append((x_value, y_value))

        # Clip to the polygon
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        if not clipped_curve.is_empty:
            if isinstance(clipped_curve, LineString):
                line = np.array(clipped_curve.coords)
            elif isinstance(clipped_curve, MultiLineString):
                # Select the longest valid segment
                valid_segments = sorted(clipped_curve.geoms, key=lambda g: g.length, reverse=True)
                line = np.array(valid_segments[0].coords)
            
            # Reverse the direction of the line if needed
            if not direction_flag:
                line = line[::-1]

            deposition_lines.append(line)
            direction_flag = not direction_flag  # Toggle the direction

        offset += hatch_distance
    # print(offset, "<=", y_range)

    # Step 4: Add Perpendicular Connecting Lines
    for i in range(len(deposition_lines) - 1):
        current_line = deposition_lines[i]
        next_line = deposition_lines[i + 1]

        if direction_flag:  # Connection from the end of the current to the start of the next
            connect_start = np.array([current_line[-1], next_line[0]])
        else:  # Connection from the start of the current to the end of the next
            connect_start = np.array([current_line[0], next_line[-1]])

        connecting_lines.append(connect_start)

    
    fly_lines = []
    if fly_in_out and deposition_lines:
        # Fly-In
        start_point = deposition_lines[0][0]
        start_dir = (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
        fly_in_point = start_point - fly_distance * start_dir
        fly_lines.append(np.array([fly_in_point, start_point]))

        # Fly-Out
        end_point = deposition_lines[-1][-1]
        end_dir = (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        fly_out_point = end_point + fly_distance * end_dir
        fly_lines.append(np.array([end_point, fly_out_point]))
    
    
    # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    # Plot deposition lines
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="blue", alpha=0.7)
    # Plot connecting lines
    for line in connecting_lines:
        plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Perpendicular Linking Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # # Debug Visualization for Contour and Offsets
    plt.figure(figsize=(8, 8))
    
    # Plot original boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    
    # Plot the polynomial fitted contour
    plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], 'r--', label="Fitted Contour")
    
    # Plot the offset contours
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], 'b-', alpha=0.6, label="Offset Contour" if 'Offset Contour' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend()
    plt.title("Contour and Offset Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    connecting_array = np.array([])
    return deposition_array, connecting_array, fly_lines_array, fillets_array

# def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):

    """
    Generates a zigzag toolpath based on non-planar boundary points, ensuring offsets stay inside the main boundary.
    """

    print('Number of points: ', len(non_planar_boundary_points))

    # Step 1: Validate and Process Boundary Points
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must contain at least two 2D coordinates.")
    
    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Validate non_planar_boundary_points
    if non_planar_boundary_points is None or len(non_planar_boundary_points) == 0:
        print("[ERROR] non_planar_boundary_points is empty!")
        return np.array([]), np.array([]), np.array([]), np.array([])

    if non_planar_boundary_points.shape[1] != 2:
        print("[ERROR] Invalid shape for non_planar_boundary_points:", non_planar_boundary_points.shape)
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Step 2: Use the Given filter_points_ransac() Function for Fitting
    filtered_points, _ = filter_points_ransac(non_planar_boundary_points, degree=2, residual_threshold=5.0, visualize=True)
    
    if len(filtered_points) < 3:
        print("[ERROR] Not enough inlier points for polynomial fitting.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    # Fit a polynomial using only the given function
    poly_coeffs = np.polyfit(x, y, deg=2)
    print("Fitted Polynomial Coefficients:", poly_coeffs)

    # Generate smooth polynomial points
    fitted_x = np.linspace(x.min(), x.max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    base_contour = np.column_stack((fitted_x, fitted_y))

    # Step 3: Generate Offsets within the Boundary
    deposition_lines = []
    connecting_lines = []
    offset = 0
    direction_flag = True

    while True:
        offset_points = np.column_stack((fitted_x, np.polyval(poly_coeffs, fitted_x) + offset))
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        if clipped_curve.is_empty:
            print("Offset curve is empty at offset:", offset)  
            break

        if isinstance(clipped_curve, LineString):
            line = np.array(clipped_curve.coords)
        elif isinstance(clipped_curve, MultiLineString):
            valid_segments = sorted(clipped_curve.geoms, key=lambda g: g.length, reverse=True)
            line = np.array(valid_segments[0].coords)

        if not direction_flag:
            line = line[::-1]

        deposition_lines.append(line)
        direction_flag = not direction_flag

        offset += hatch_distance  

    # Step 4: Add Perpendicular Connecting Lines
    for i in range(len(deposition_lines) - 1):
        current_line = deposition_lines[i]
        next_line = deposition_lines[i + 1]

        if direction_flag:  
            connect_start = np.array([current_line[-1], next_line[0]])
        else:  
            connect_start = np.array([current_line[0], next_line[-1]])

        connecting_lines.append(connect_start)

    # Step 5: Fly-In & Fly-Out
    fly_lines = []
    if fly_in_out and deposition_lines:
        start_point = deposition_lines[0][0]
        start_dir = (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
        fly_in_point = start_point - fly_distance * start_dir
        fly_lines.append(np.array([fly_in_point, start_point]))

        end_point = deposition_lines[-1][-1]
        end_dir = (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        fly_out_point = end_point + fly_distance * end_dir
        fly_lines.append(np.array([end_point, fly_out_point]))

    # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], 'b-', label="Deposition Line" if 'Deposition Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    for line in connecting_lines:
        plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Clipped Offsets")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 6: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    return deposition_array, connecting_array, fly_lines_array, fillets_array





# def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):

#     """
#     Generates a zigzag toolpath using Shapely, ensuring offsets stay inside the main boundary.
#     """
#     # Ensure the boundary is closed
#     if not np.allclose(boundary_points[0], boundary_points[-1]):
#         boundary_points = np.vstack([boundary_points, boundary_points[0]])
    
#     # Convert boundary points to a Shapely Polygon
#     boundary_polygon = Polygon(boundary_points)
#     if not boundary_polygon.is_valid:
#         raise ValueError("Invalid boundary polygon")
    
#     # Convert non-planar boundary points to a Shapely LineString
#     if len(non_planar_boundary_points) < 2:
#         raise ValueError("Non-planar boundary must have at least two points")
    
#     ordered_points = sorted(non_planar_boundary_points, key=lambda x: x[0])
    
#     contour_line = LineString(ordered_points)

#     if not contour_line.is_valid:
#         raise ValueError("Invalid contour line from non-planar boundary points")

#     # Plotting the contour line and boundary
#     plt.figure(figsize=(8, 8))

#     # Plot the contour line (non-planar boundary)
#     x, y = contour_line.xy  # Extract the x and y coordinates of the contour line
#     plt.plot(x, y, label="Contour Line", color='b', marker='o', markersize=5)  # Plot the contour line
#     plt.fill(x, y, alpha=0.1, color='b')  # Optional: Fill the contour for better visibility

#     # Plot the boundary (main boundary) as a reference
#     boundary_x, boundary_y = boundary_polygon.exterior.xy  # Get the boundary points for plotting
#     plt.plot(boundary_x, boundary_y, label="Boundary", color='k', linewidth=2)  # Plot the boundary in black
    
#     # Add titles, labels, and grid
#     plt.title("Visualization of the Contour Line and Boundary")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()

#     # Show the plot
#     plt.show()
    
#     # # Clip contour inside the main boundary
#     # clipped_contour = boundary_polygon.intersection(contour_line)

#     # if clipped_contour.is_empty:
#     #     raise ValueError("Clipped contour is empty. Check boundary and non-planar points alignment.")
    
#     # # Ensure clipped_contour is a LineString for offsetting
#     # if isinstance(clipped_contour, MultiLineString):
#     #     clipped_contour = max(clipped_contour.geoms, key=lambda g: g.length)  # Choose longest segment
    
#     # Initialize toolpath storage
#     deposition_lines = []
#     connecting_lines = []
#     fly_lines = []
#     offset = hatch_distance  # Start with the first offset
#     direction_flag = True
#     y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min() + 10 

#     while offset <= y_range:        
#         # Offset the clipped contour inward
#         # offset_contour = clipped_contour.parallel_offset(-offset, side='right', resolution=1)
#         offset_contour = contour_line.parallel_offset(offset, side='right', resolution=50)
       
        
#         # Stop if offset results in an empty line
#         if not isinstance(offset_contour, (Polygon, MultiLineString, LineString)):
#             print("Offset Contour is Invalid: Not a Polygon, MultiLineString, or LineString")
#             break
        
#         # Convert to LineString if it's a valid offset result
#         if isinstance(offset_contour, Polygon):
#             print("Polygon Detected")
#             boundary_line = LineString(offset_contour.exterior.coords)
#         if isinstance(offset_contour, MultiLineString):
#             print("MultiLineString Detected")
#             boundary_line = max(offset_contour.geoms, key=lambda g: g.length)  # Choose longest segment
#         if isinstance(offset_contour, LineString):
#             boundary_line = offset_contour
#         else:
#             break
                
        
#         # Convert to NumPy array
#         line = np.array(boundary_line.coords)

#         # Reverse direction for zigzag pattern
#         if not direction_flag:
#             line = line[::-1]
        
#         deposition_lines.append(line)
#         direction_flag = not direction_flag

#         # Move to next offset
#         offset += hatch_distance

    
#     plt.figure(figsize=(8, 8))

#     # Plot the original contour line (red)
#     x, y = contour_line.xy
#     plt.plot(x, y, 'r-', linewidth=2, label='Damage Contour')

#     # Plot the boundary (black)
#     plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2, label='Boundary')

#     # Plot all offset deposition lines (blue), using only one label
#     for idx, line in enumerate(deposition_lines):
#         if idx == 0:
#             plt.plot(line[:, 0], line[:, 1], 'b-', label='Offset Deposition Lines')
#         else:
#             plt.plot(line[:, 0], line[:, 1], 'b-')  # No label for rest

#     plt.axis('equal')
#     plt.title("Offset Deposition Lines with Boundary")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#         # print(offset, "<=", y_range)
    
#     trimmed_boundary = boundary_polygon.buffer(hatch_distance, cap_style=2)  

#     # Trim the offset contours using the new "trimmed boundary"
#     clipped_deposition_lines = []
#     for line in deposition_lines:
#         line_geom = LineString(line)
#         # Clip each deposition line to the trimmed boundary polygon
#         clipped_line = boundary_polygon.intersection(line_geom)
        
#         if not clipped_line.is_empty:
#             if isinstance(clipped_line, MultiLineString):
#                 # If the result is a MultiLineString, take the longest segment
#                 clipped_line = max(clipped_line.geoms, key=lambda g: g.length)
#             elif isinstance(clipped_line, LineString):
#                 # If the result is a single LineString, use it directly
#                 pass

#             clipped_deposition_lines.append(np.array(clipped_line.coords))

#     # # Trim the offset contours to stay within the boundary (Clip based on boundary)
#     # clipped_deposition_lines = []
#     # for line in deposition_lines:
#     #     line_geom = LineString(line)
#     #     # Clip each deposition line to the boundary polygon
#     #     clipped_line = boundary_polygon.intersection(line_geom)
        
#     #     if not clipped_line.is_empty:
#     #         if isinstance(clipped_line, MultiLineString):
#     #             # If the result is a MultiLineString, take the longest segment
#     #             clipped_line = max(clipped_line.geoms, key=lambda g: g.length)
#     #         elif isinstance(clipped_line, LineString):
#     #             # If the result is a single LineString, use it directly
#     #             pass

#     #         clipped_deposition_lines.append(np.array(clipped_line.coords))

#     # Visualization of offset lines and their clipped version
#     plt.figure(figsize=(8, 8))
#     for line in clipped_deposition_lines:
#         plt.plot(line[:, 0], line[:, 1], 'b-')
#     # for line in clipped_deposition_lines:
#     #     plt.plot(line[:, 0], line[:, 1], 'b-', label="Clipped Deposition Line")
#     plt.plot(line[:, 0], line[:, 1], 'b-', label="Deposition Line")
#     plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
#     plt.title("Clipped Offset Contour inside Boundary")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     # Create connecting lines
#     # for i in range(len(clipped_deposition_lines) - 1):
#     #     current_line = clipped_deposition_lines[i]
#     #     next_line = clipped_deposition_lines[i + 1]
#     #     connecting_lines.append(np.array([current_line[-1], next_line[0]]))

#     # Ensure (X, Y, Z) format for deposition lines
#     deposition_lines_3D = []
#     for i, line in enumerate(clipped_deposition_lines):
#         z_value = layer_index
#         line_3D = np.hstack([line, np.full((line.shape[0], 1), z_value)])

#         if i > 0:
#             # Add direct connection (ensuring it's (X, Y, Z))
#             prev_end = deposition_lines_3D[-1][-1]
#             curr_start = line_3D[0]
#             connecting_line = np.array([prev_end, curr_start])
#             deposition_lines_3D.append(connecting_line)

#         deposition_lines_3D.append(line_3D)


#     # Fly-in & Fly-out movements
#     if fly_in_out and deposition_lines:
#         start_point = deposition_lines[0][0]
#         end_point = deposition_lines[-1][-1]
        
#         fly_in_point = start_point - fly_distance * (deposition_lines[0][1] - start_point) / np.linalg.norm(deposition_lines[0][1] - start_point)
#         fly_out_point = end_point + fly_distance * (end_point - deposition_lines[-1][-2]) / np.linalg.norm(end_point - deposition_lines[-1][-2])
        
#         fly_lines.append(np.array([fly_in_point, start_point]))
#         fly_lines.append(np.array([end_point, fly_out_point]))
    
#     # Visualization
#     plt.figure(figsize=(8, 8))
#     for line in clipped_deposition_lines:
#         plt.plot(line[:, 0], line[:, 1], 'b-', label="Deposition Line" if 'Deposition Line' not in plt.gca().get_legend_handles_labels()[1] else "")
#     for line in connecting_lines:
#         plt.plot(line[:, 0], line[:, 1], 'g--', label="Connecting Line" if 'Connecting Line' not in plt.gca().get_legend_handles_labels()[1] else "")
#     plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
#     plt.legend()
#     plt.title("Hybrid Toolpath with Shapely Offsets and Clipped Contours")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.axis("equal")
#     plt.grid()
#     plt.show()
    
#     # Return toolpath data
#     deposition_array = np.vstack(clipped_deposition_lines) if clipped_deposition_lines else np.array([])
#     connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
#     fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
#     fillets_array = np.array([])  # Placeholder for fillets
#     return deposition_array, connecting_array, fly_lines_array, fillets_array

# good up
# extension bottom 

def generate_hybrid2_toolpath(boundary_points, non_planar_boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    
    extend = False
    
    def find_boundary_hit(start_point, direction, boundary_polygon, extension_length=20):
        ray = LineString([start_point, start_point + direction * extension_length])
        inter = boundary_polygon.intersection(ray)
        if inter.is_empty:
            return start_point + direction * extension_length
        elif inter.geom_type == 'Point':
            return np.array(inter.coords[0])
        elif inter.geom_type == 'MultiPoint':
            return min((np.array(p.coords[0]) for p in inter.geoms), key=lambda pt: np.linalg.norm(pt - start_point))
        else:
            return start_point + direction * extension_length
        
    def snap_to_main_axis(vec):
        vec = vec / np.linalg.norm(vec)
        if abs(vec[0]) > abs(vec[1]):
            return np.array([np.sign(vec[0]), 0.0])  # Snap to horizontal
        else:
            return np.array([0.0, np.sign(vec[1])])  # Snap to vertical
    
    def is_close_to_boundary(pt, boundary, tol=1e-3):
        return boundary.exterior.distance(Point(pt)) < tol
            

    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    boundary_polygon = Polygon(boundary_points)
    if not boundary_polygon.is_valid:
        raise ValueError("Invalid boundary polygon")

    if len(non_planar_boundary_points) < 2:
        raise ValueError("Non-planar boundary must have at least two points")

    ordered_points = sorted(non_planar_boundary_points, key=lambda x: x[0])
    contour_line = LineString(ordered_points)

    if not contour_line.is_valid:
        raise ValueError("Invalid contour line from non-planar boundary points")

    deposition_lines = []
    direction_flag = True
    offset = hatch_distance
    max_extent = boundary_polygon.bounds[3] - boundary_polygon.bounds[1] + 10

    while offset <= max_extent:
        offset_contour = contour_line.parallel_offset(-offset, side='right', resolution=50)

        if isinstance(offset_contour, (MultiLineString, LineString)):
            line = offset_contour
            if isinstance(offset_contour, MultiLineString):
                line = max(offset_contour.geoms, key=lambda g: g.length)

            coords = np.array(line.coords)
            if not direction_flag:
                coords = coords[::-1]
            deposition_lines.append(coords)
            direction_flag = not direction_flag
        offset += hatch_distance

    # plt.figure(figsize=(8, 8))

    # # Plot the original contour line (red)
    # x, y = contour_line.xy
    # plt.plot(x, y, 'r-', linewidth=2, label='Damage Contour')

    # # Plot the boundary (black)
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2, label='Boundary')

    # # Plot all offset deposition lines (blue), using only one label
    # for idx, line in enumerate(deposition_lines):
    #     if idx == 0:
    #         plt.plot(line[:, 0], line[:, 1], 'b-', label='Offset Deposition Lines')
    #     else:
    #         plt.plot(line[:, 0], line[:, 1], 'b-')  # No label for the rest

    # Axis formatting
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # ax.set_title("Contour Offset", fontsize=20)
    # ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    # ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
    # ax.tick_params(axis='both', labelsize=14)
    # ax.legend(fontsize=16)
    # ax.grid(True)


    # # Axis formatting
    # ax = plt.gca()
    # ax.set_aspect('equal')

    # # Keep only title
    # ax.set_title("Contour Offset", fontsize=20)

    # # Remove labels
    # ax.set_xlabel('')
    # ax.set_ylabel('')

    # # Remove ticks and numbers
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # Remove grid
    # ax.grid(False)


    # plt.tight_layout()
    # plt.savefig("Contour Offset Model A") 
    # plt.show()


    # Clip lines to boundary first
    clipped_lines = []
    for line in deposition_lines:
        clipped = boundary_polygon.intersection(LineString(line))
        if not clipped.is_empty:
            if isinstance(clipped, MultiLineString):
                clipped = max(clipped.geoms, key=lambda g: g.length)
            clipped_lines.append(np.array(clipped.coords))

    if extend:
        
        # # PLOT: Before extension
        # fig, axs = plt.subplots(1, 4, figsize=(16, 8))

        # axs[0].plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')
        # for line in clipped_lines:
        #     axs[0].plot(line[:, 0], line[:, 1], 'b-', linewidth=1.5)
        # axs[0].set_title("Before Extension")
        # axs[0].axis('equal')
        # axs[0].legend()
        # axs[0].grid(True)

        # Now apply extension
        # extended_lines = []
        # for line in clipped_lines:
        #     new_line = line.copy()

        #     start_pt = line[0]
        #     end_pt = line[-1]

        #     # Check if start and end are close to boundary
        #     start_near_boundary = is_close_to_boundary(start_pt, boundary_polygon)
        #     end_near_boundary = is_close_to_boundary(end_pt, boundary_polygon)

        #     # Start extension if needed
        #     if not start_near_boundary:
        #         start_vec = line[1] - line[0]
        #         start_dir = snap_to_main_axis(start_vec)
        #         new_start = find_boundary_hit(start_pt, -start_dir, boundary_polygon)
        #         new_line = np.vstack([new_start, new_line])

        #     # End extension if needed
        #     if not end_near_boundary:
        #         end_vec = line[-1] - line[-2]
        #         end_dir = snap_to_main_axis(end_vec)
        #         new_end = find_boundary_hit(end_pt, end_dir, boundary_polygon)
        #         new_line = np.vstack([new_line, new_end])

        #     extended_lines.append(new_line)

        # Determine dominant direction for extension based on bounding box aspect ratio

        standard_direction = np.array([1.0, 0.0])  # Force X+ direction

        def enforce_consistent_direction(lines, direction):
            fixed_lines = []
            for line in lines:
                vec = line[-1] - line[0]
                if np.dot(vec, direction) < 0:  # If pointing opposite, reverse
                    line = line[::-1]
                fixed_lines.append(line)
            return fixed_lines

        clipped_lines = enforce_consistent_direction(clipped_lines, standard_direction)

        minx, miny, maxx, maxy = boundary_polygon.bounds
        width = maxx - minx
        height = maxy - miny

        if width >= height:
            default_extension_dir = np.array([1.0, 0.0])  # Extend in X+
        else:
            default_extension_dir = np.array([0.0, 1.0])  # Extend in Y+

        extension_info = []  # Track extensions: (line_index, 'start'/'end', direction_vector)
        extended_lines = []

        # Apply extensions and record direction
        for line_idx, line in enumerate(clipped_lines):
            new_line = line.copy()

            start_pt = line[0]
            end_pt = line[-1]

            start_near_boundary = is_close_to_boundary(start_pt, boundary_polygon)
            end_near_boundary = is_close_to_boundary(end_pt, boundary_polygon)

            # Start extension
            if not start_near_boundary:
                new_start = find_boundary_hit(start_pt, -default_extension_dir, boundary_polygon)
                new_line = np.vstack([new_start, new_line])
                extension_info.append((len(extended_lines), 'start', -default_extension_dir))

            # End extension
            if not end_near_boundary:
                new_end = find_boundary_hit(end_pt, default_extension_dir, boundary_polygon)
                new_line = np.vstack([new_line, new_end])
                extension_info.append((len(extended_lines), 'end', default_extension_dir))

            extended_lines.append(new_line)

        # Correction pass to fix inconsistent extensions
        if extension_info:
            for line_idx, pos, vec in extension_info:
                dot = np.dot(vec, default_extension_dir)
                if dot < 0.5:  # Too different from dominant direction
                    line = extended_lines[line_idx]
                    if pos == 'start':
                        pt = line[1]  # Original start point (after extension)
                        corrected_pt = find_boundary_hit(pt, -default_extension_dir, boundary_polygon)
                        extended_lines[line_idx][0] = corrected_pt
                    elif pos == 'end':
                        pt = line[-2]  # Original end point (before extension)
                        corrected_pt = find_boundary_hit(pt, default_extension_dir, boundary_polygon)
                        extended_lines[line_idx][-1] = corrected_pt

            print(f"Corrected inconsistent extensions to align with dominant direction: {default_extension_dir}")

            
        # PLOT: After extension
        # axs[1].plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')
        # for line in extended_lines:
        #     axs[1].plot(line[:, 0], line[:, 1], 'g-', linewidth=1.5)
        # axs[1].set_title("After Extension")
        # axs[1].axis('equal')
        # axs[1].legend()
        # axs[1].grid(True)

        # -- Re-clip extended lines to boundary polygon --
        clipped_extended_lines = []
        for line in extended_lines:
            ls = LineString(line)
            clipped = boundary_polygon.intersection(ls)
            if not clipped.is_empty:
                if isinstance(clipped, MultiLineString):
                    clipped = max(clipped.geoms, key=lambda g: g.length)
                clipped_extended_lines.append(np.array(clipped.coords))



        # axs[2].plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')
        # for line in clipped_extended_lines:
        #     axs[2].plot(line[:, 0], line[:, 1], 'b-', linewidth=1.5)
        # axs[2].set_title("Before Extension and Clipping")
        # axs[2].axis('equal')
        # axs[2].legend()
        # axs[2].grid(True)

        # plt.suptitle("Before vs After Extension Comparison")

        sequential_lines = []

        # Start with the first extended line
        if not clipped_extended_lines:
            print("No clipped lines available for sequential offsetting.")
        else:
            initial_line = clipped_extended_lines[0]
            start_pt = initial_line[0]
            end_pt = initial_line[-1]

            # Extend the first line to boundary
            start_ext = find_boundary_hit(start_pt, -default_extension_dir, boundary_polygon)
            end_ext = find_boundary_hit(end_pt, default_extension_dir, boundary_polygon)
            extended = np.vstack([start_ext, *initial_line, end_ext])
            sequential_lines.append(extended)

            # Build the first line geometry
            current_geom = LineString(extended)
            side = 'right' if np.allclose(default_extension_dir, [1, 0]) else 'left'
            side = 'right'
            while True:
                # Offset in main direction
                try:
                    offset_geom = current_geom.parallel_offset(hatch_distance, side=side, join_style=2)
                except Exception:
                    # print("Offsetting failed due to geometry issues.")
                    break  # In case offsetting fails due to geometry

                if offset_geom.is_empty:
                    # print("Offset geometry is empty.")
                    break
                if isinstance(offset_geom, MultiLineString):
                    offset_geom = max(offset_geom.geoms, key=lambda g: g.length)

                # axs[3].plot(*offset_geom.xy, 'c--', alpha=0.5)

                clipped = boundary_polygon.intersection(offset_geom)
                if clipped.is_empty:
                    # print("Clipped geometry is empty.")
                    break

                if isinstance(clipped, LineString):
                    new_coords = np.array(clipped.coords)
                elif isinstance(clipped, MultiLineString):
                    clipped = max(clipped.geoms, key=lambda g: g.length)
                    new_coords = np.array(clipped.coords)
                else:
                    # print("Unexpected geometry type after clipping.")
                    break

                # Extend both ends to boundary if not already touching
                if not is_close_to_boundary(new_coords[0], boundary_polygon):
                    new_start = find_boundary_hit(new_coords[0], -default_extension_dir, boundary_polygon)
                    new_coords = np.vstack([new_start, new_coords])
                if not is_close_to_boundary(new_coords[-1], boundary_polygon):
                    new_end = find_boundary_hit(new_coords[-1], default_extension_dir, boundary_polygon)
                    new_coords = np.vstack([new_coords, new_end])

                sequential_lines.append(new_coords)
                current_geom = LineString(new_coords)
        
            
            # -- Re-clip extended lines to boundary polygon --
        clipped_sequential_lines = []
        for line in sequential_lines:
            ls = LineString(line)
            clipped = boundary_polygon.intersection(ls)
            if not clipped.is_empty:
                if isinstance(clipped, MultiLineString):
                    clipped = max(clipped.geoms, key=lambda g: g.length)
                clipped_sequential_lines.append(np.array(clipped.coords))

        # Plot 4th subplot
        # axs[3].plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')
        # for line in sequential_lines:
        #     axs[3].plot(line[:, 0], line[:, 1], 'm-', linewidth=1.5)
        # axs[3].plot(contour_line.xy[0], contour_line.xy[1], 'r-', label='Contour Line', alpha=0.5)
        # axs[3].set_title("Sequential Offset with Extension")
        # axs[3].axis('equal')
        # axs[3].legend()
        # axs[3].grid(True)

        # plt.show()


    else:
        clipped_sequential_lines = clipped_lines

        # Visualization
        # plt.figure(figsize=(8, 8))
        # for line in clipped_sequential_lines:
        #     plt.plot(line[:, 0], line[:, 1], 'b-', label="Deposition Line" if 'Deposition Line' not in plt.gca().get_legend_handles_labels()[1] else "")
        # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
        # plt.legend()
        # plt.title("Hybrid Toolpath with Shapely Offsets and Clipped Contours")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.axis("equal")
        # plt.grid()
        # plt.show()

        # Visualization
        # plt.figure(figsize=(8, 8))

        # for line in clipped_sequential_lines:
        #     plt.plot(
        #         line[:, 0], line[:, 1], 'b-',
        #         label="Deposition Line" if 'Deposition Line' not in plt.gca().get_legend_handles_labels()[1] else ""
        #     )

        # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")

        # # Format axes
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_title("Hybrid Toolpath with Shapely Offsets and Clipped Contours", fontsize=20)
        # ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
        # ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
        # ax.tick_params(axis='both', labelsize=14)
        # ax.legend(fontsize=16)
        # ax.grid(True)

        # plt.tight_layout()
        # plt.show()


    # Build 3D paths and connecting lines
    deposition_lines_3D = []
    connecting_lines = []
    fly_lines = []

    deposition_lines_3D = []
    for i, line in enumerate(clipped_sequential_lines):
        line_3D = np.hstack([line, np.full((line.shape[0], 1), layer_index)])
        if i > 0:
            prev_end = deposition_lines_3D[-1][-1]
            curr_start = line_3D[0]
            connecting_lines.append(np.array([prev_end, curr_start]))
            
        deposition_lines_3D.append(line_3D)


   
    if fly_in_out and clipped_sequential_lines:
        start = clipped_sequential_lines[0][0]
        end = clipped_sequential_lines[-1][-1]
        fly_in_vec = clipped_sequential_lines[0][1] - start
        fly_out_vec = end - clipped_sequential_lines[-1][-2]
        fly_lines.append(np.array([start - fly_in_vec / np.linalg.norm(fly_in_vec) * fly_distance, start]))
        fly_lines.append(np.array([end, end + fly_out_vec / np.linalg.norm(fly_out_vec) * fly_distance]))

    deposition_array = np.vstack(clipped_sequential_lines) if clipped_sequential_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])

    # # 2D Visualization of Deposition and Travel Paths
    # plt.figure(figsize=(8, 8))

    # # Plot deposition lines (as connected path)
    # if deposition_array.size > 0:
    #     plt.plot(deposition_array[:, 0], deposition_array[:, 1], 'b-', label='Deposition Path')

    # # Plot connecting travel moves
    # # if connecting_array.size > 0:
    # #     for i in range(0, len(connecting_array), 2):
    # #         segment = connecting_array[i:i+2]
    # #         if len(segment) == 2:
    # #             plt.plot(segment[:, 0], segment[:, 1], 'g--', label='Connecting Move' if i == 0 else "")

    # # Plot fly-in/fly-out lines
    # # if fly_lines_array.size > 0:
    # #     for i in range(0, len(fly_lines_array), 2):
    # #         segment = fly_lines_array[i:i+2]
    # #         if len(segment) == 2:
    # #             plt.plot(segment[:, 0], segment[:, 1], 'r--', label='Fly Move' if i == 0 else "")

    # # Optional: plot boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label='Boundary')

    # plt.title("2D Toolpath Visualization with Travel and Fly Lines")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()



    return deposition_array, connecting_array, fly_lines_array, fillets_array

    # deposition_lines_3D = []
    # fly_lines = []
    
    # # Alternate directions for zig-zag and flatten into a single deposition path
    # zigzag_path = []

    # # for i, line in enumerate(clipped_sequential_lines):
    # #     if i % 2 == 1:
    # #         line = line[::-1]  # Reverse every other line

    # #     if i > 0:
    # #         # Add a connecting segment (as deposition) from last to current start
    # #         connect = np.array([zigzag_path[-1], line[0]])
    # #         zigzag_path.extend(connect)

    # #     zigzag_path.extend(line)

    # # try:
    # #     zigzag_path = np.array(zigzag_path)
    # #     if zigzag_path.ndim != 2 or zigzag_path.shape[1] != 2:
    # #         raise ValueError("zigzag_path is not a 2D array of XY points.")
    # #     deposition_lines_3D = zigzag_path
    # # except Exception as e:
    # #     print(f"[Layer {layer_index}] Skipping zigzag deposition path due to error: {e}")
    # #     deposition_lines_3D = []  # or set to None, depending on your later code

    # for i, line in enumerate(clipped_sequential_lines):
    #     if i == 0:
    #         zigzag_path.extend(line)
    #         continue

    #     # Previous end point
    #     prev_end = zigzag_path[-1]

    #     # Check both ends of current line
    #     start = line[0]
    #     end = line[-1]

    #     # Use the closest end to connect
    #     if np.linalg.norm(start - prev_end) <= np.linalg.norm(end - prev_end):
    #         connect = np.array([prev_end, start])
    #         zigzag_path.extend(connect)
    #         zigzag_path.extend(line)
    #     else:
    #         connect = np.array([prev_end, end])
    #         zigzag_path.extend(connect)
    #         zigzag_path.extend(line[::-1])  # reverse line to go from end to start

    # # try:
    # #     zigzag_path = np.array(zigzag_path)Zigzagag deposition path due to error: {e}")
    # #     deposition_lines_3D = []

    # # Visualization of zigzag deposition + fly lines
    # plt.figure(figsize=(8, 8))

    # zigzag_path = np.array(zigzag_path)

    # # Plot the zigzag deposition path (blue)
    # plt.plot(zigzag_path[:, 0], zigzag_path[:, 1], 'b.-', label='Deposition Path')

    # # Optional: plot the boundary if needed
    # # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2, label='Boundary')

    # # Axis formatting (consistent style)
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # ax.set_title("Deposition Path", fontsize=20)
    # ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    # ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
    # ax.tick_params(axis='both', labelsize=14)
    # ax.legend(fontsize=16)
    # ax.grid(True)

    # plt.tight_layout()
    # plt.show()



    # # Add fly-in and fly-out
    # if fly_in_out and len(zigzag_path) >= 2:
    #     start = zigzag_path[0]
    #     end = zigzag_path[-1]
    #     fly_in_vec = zigzag_path[1] - start
    #     fly_out_vec = end - zigzag_path[-2]
    #     fly_lines.append(np.array([start - fly_in_vec / np.linalg.norm(fly_in_vec) * fly_distance, start]))
    #     fly_lines.append(np.array([end, end + fly_out_vec / np.linalg.norm(fly_out_vec) * fly_distance]))

    # # deposition_array = np.vstack(deposition_lines_3D) if len(deposition_lines_3D) > 0 else np.array([]) 
    # deposition_array = np.array(zigzag_path)  # Use the zigzag path directly
    # # print(deposition_array.shape)
    # connecting_array = np.array([])  # No separate travel moves in this version
    # fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    # fillets_array = np.array([])

    # return deposition_array, connecting_array, fly_lines_array, fillets_array

    












def moving_average(points, window_size=5):
    """
    Apply a moving average filter to the given points to smooth the contour.
    window_size controls the amount of smoothing.
    """
    smoothed_points = []
    for i in range(len(points)):
        # Get the window of points centered at i (with some boundary checks)
        start_idx = max(i - window_size // 2, 0)
        end_idx = min(i + window_size // 2 + 1, len(points))
        window = points[start_idx:end_idx]
        
        # Calculate the average of x and y coordinates
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        smoothed_points.append((avg_x, avg_y))
    
    return smoothed_points

def compute_aic(y_true, y_pred, degree, num_samples):
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    k = degree + 1  # Number of parameters
    aic = num_samples * np.log(mse) + 2 * k
    return aic

def filter_points_ransac(arc_points, degree=2, residual_threshold=5.0, visualize=False):
    """
    Filters arc points using RANSAC to remove noise and fits a polynomial.
    """
    arc_points = np.array(arc_points)
    # print(arc_points)
    if len(arc_points) < degree + 1:
        print("Not enough points to fit a polynomial.")
        return arc_points, np.array([])  # Return all points as inliers

    # Separate X and Y components
    x = arc_points[:, 0].reshape(-1, 1)
    y = arc_points[:, 1]

    # Fit a polynomial using RANSAC
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    model.fit(x, y)

    # Get inliers and outliers
    ransac = model.named_steps['ransacregressor']
    inlier_mask = ransac.inlier_mask_
    inlier_points = arc_points[inlier_mask]
    outlier_points = arc_points[~inlier_mask]

    # Optional visualization of inliers and outliers
    if visualize:
        plt.scatter(inlier_points[:, 0], inlier_points[:, 1], color="green", label="Inliers")
        plt.scatter(outlier_points[:, 0], outlier_points[:, 1], color="red", label="Outliers")
        x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
        y_plot = model.predict(x_plot)
        plt.plot(x_plot, y_plot, color="blue", label="Fitted Polynomial", linewidth=2)
        plt.legend()
        plt.show()

    return inlier_points, outlier_points

def intersect_triangle_with_plane(v0, v1, v2, plane_z):
    """
    Intersects a triangle with the slicing plane at a given Z-height.
    
    Parameters:
        v0, v1, v2 (np.array): Vertices of the triangle.
        plane_z (float): The Z height of the slicing plane.
    
    Returns:
        list: Intersection points (x, y) along the edges of the triangle.
    """
    intersection_points = []
    
    # Check for each edge of the triangle if it intersects with the slicing plane
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    for p1, p2 in edges:
        if (p1[2] < plane_z and p2[2] > plane_z) or (p1[2] > plane_z and p2[2] < plane_z):
            # Interpolate the intersection point along the edge
            t = (plane_z - p1[2]) / (p2[2] - p1[2])  # Parametric interpolation factor
            intersection_point = p1 + t * (p2 - p1)  # Get intersection point
            intersection_points.append(intersection_point[:2])  # Only store x, y
    
    return intersection_points

# def extract_damage_contour(mesh, boundary_points, layer_height, threshold=1.5, use_nurbs=True):
    """
    Segments the boundary based on the intersection of the non-planar mesh with the slicing plane.
    If use_nurbs=True, fits a NURBS curve to the extracted damage contour.
    
    Parameters:
        mesh (o3d.geometry.TriangleMesh): The non-planar mesh.
        boundary_points (np.ndarray): The existing boundary (N x 2).
        layer_height (float): The Z-height at which to slice.
        threshold (float): The max distance to consider a boundary point part of the damage contour.
        use_nurbs (bool): If True, fits a NURBS curve to the damage contour.
    
    Returns:
        damage_contour (np.ndarray): Segmented part of the boundary (either as points or a NURBS curve).
    """
    # Extract intersection points from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    intersection_points = []
    
    for tri in triangles:
        v0, v1, v2 = vertices[tri]  # Triangle vertices

        # Check if the triangle crosses the slicing plane (Z = layer_height)
        z_vals = np.array([v0[2], v1[2], v2[2]])
        if np.min(z_vals) <= layer_height <= np.max(z_vals):
            # Get intersection points for the triangle edges
            triangle_intersections = intersect_triangle_with_plane(v0, v1, v2, layer_height)
            intersection_points.extend(triangle_intersections)

    intersection_points = np.array(intersection_points)
    
    if len(intersection_points) == 0:
        print(f"[WARNING] No intersection found at height {layer_height}.")
        return None
    
    # Build a KDTree for fast nearest-neighbor search
    tree = cKDTree(intersection_points)
    
    # Identify boundary points that are close to the intersection points
    damage_contour = []
    for bp in boundary_points:
        distance, _ = tree.query(bp)  # Find closest intersection point
        if distance < threshold:
            damage_contour.append(bp)
    
    damage_contour = np.array(damage_contour)
    
    if len(damage_contour) == 0:
        print("[WARNING] No damage contour found within threshold.")
        return None
    
    # If NURBS fitting is enabled
    if use_nurbs and len(damage_contour) > 2:
        # Fit NURBS using scipy's splprep
        tck, _ = splprep([damage_contour[:, 0], damage_contour[:, 1]], s=0.5, k=3)
        u_new = np.linspace(0, 1, 100)
        nurbs_contour = np.array(splev(u_new, tck)).T
        damage_contour = nurbs_contour  # Replace original contour with the smooth curve
    
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b--', label="Boundary")  # Blue full boundary
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='r', label="Intersection Points")  # Red intersection points
    
    if use_nurbs:
        plt.plot(damage_contour[:, 0], damage_contour[:, 1], 'm-', linewidth=2, label="NURBS Contour")  # Magenta NURBS curve
    else:
        plt.plot(damage_contour[:, 0], damage_contour[:, 1], 'g-', linewidth=2, label="Damage Contour")  # Green segmented contour
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Segmented Contour at Z = {layer_height}")
    plt.grid()
    plt.show()
    
    return damage_contour
# 
# def extract_damage_contour(mesh, boundary_points, layer_height, threshold=0.8):
#     """
#     Segments the boundary based on the intersection of the non-planar mesh with the slicing plane.

#     Parameters:
#         mesh (o3d.geometry.TriangleMesh): The non-planar mesh.
#         boundary_points (np.ndarray): The existing boundary (N x 2).
#         layer_height (float): The Z-height at which to slice.
#         threshold (float): The max distance to consider a boundary point part of the damage contour.

#     Returns:
#         damage_contour (np.ndarray): Segmented part of the boundary.
#     """

#     # Extract intersection points from the mesh
#     vertices = np.asarray(mesh.vertices)
#     triangles = np.asarray(mesh.triangles)
    
#     intersection_points = []

#     for tri in triangles:
#         v0, v1, v2 = vertices[tri]  # Triangle vertices

#         # Check if the triangle crosses the slicing plane (Z = layer_height)
#         z_vals = np.array([v0[2], v1[2], v2[2]])
#         if np.min(z_vals) <= layer_height <= np.max(z_vals):
            
#             # Check edges for intersections
#             edges = [(v0, v1), (v1, v2), (v2, v0)]
#             for p1, p2 in edges:
#                 if (p1[2] < layer_height and p2[2] > layer_height) or (p1[2] > layer_height and p2[2] < layer_height):
#                     # Interpolate intersection point
#                     t = (layer_height - p1[2]) / (p2[2] - p1[2])
#                     intersection_point = p1 + t * (p2 - p1)
#                     intersection_points.append(intersection_point[:2])  # Store only X, Y

#     intersection_points = np.array(intersection_points)

#     if len(intersection_points) == 0:
#         print(f"[WARNING] No intersection found at height {layer_height}.")
#         return None

#     # Build a KDTree for fast nearest-neighbor search
#     tree = cKDTree(intersection_points)

#     # Identify boundary points that are close to the intersection points
#     damage_contour = []
#     for bp in boundary_points:
#         distance, _ = tree.query(bp)  # Find closest intersection point
#         if distance < threshold:
#             damage_contour.append(bp)

#     damage_contour = np.array(damage_contour)

#     if len(damage_contour) == 0:
#         print("[WARNING] No damage contour found within threshold.")
#         return None

#     # Visualization
#     plt.figure(figsize=(6, 6))
#     plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b--', label="Boundary")  # Blue full boundary
#     plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='r', label="Intersection Points")  # Red intersection points
#     plt.plot(damage_contour[:, 0], damage_contour[:, 1], 'g-', linewidth=2, label="Damage Contour")  # Green segmented contour
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.legend()
#     plt.title(f"Segmented Contour at Z = {layer_height}")
#     plt.grid()
#     plt.show()

#     return damage_contour



def extract_surface_contour(mesh, layer_height):
    """
    Extracts an open contour from a surface mesh at a given Z height.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The surface mesh.
        layer_height (float): The slicing height.

    Returns:
        contour (np.ndarray): Extracted open boundary points.
    """
    contour = []
    # Convert to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=100000)

    # Extract slice using point cloud slicing
    contour = slice_surface(pcd, layer_height)

    contour = np.vstack(contour) if contour.all else np.array([])
    return contour

def slice_surface(pcd, layer_height, tolerance=0.05):
    """
    Slices a surface (point cloud) instead of a volume.

    Parameters:
        pcd (o3d.geometry.PointCloud): The input surface point cloud.
        layer_height (float): The height at which to slice.
        tolerance (float): Tolerance for selecting points near the slice height.

    Returns:
        slice_points (np.ndarray): Extracted open boundary points.
    """
    points = np.asarray(pcd.points)
    mask = np.abs(points[:, 2] - layer_height) < tolerance  # Find points near the slicing height
    slice_points = points[mask][:, :2]  # Keep only (X, Y)

    return slice_points

def extract_surface_contour_by_mesh(mesh, layer_height):
    """
    Extracts 2D contour(s) from an already loaded Trimesh object.

    Parameters:
        mesh (trimesh.Trimesh): The triangle mesh.
        layer_height (float): The slicing height (Z).

    Returns:
        contours (List[np.ndarray]): List of (N, 2) arrays representing 2D contours at that height.
    """

    # Safety check: Ensure it's a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Expected a trimesh.Trimesh object.")

    # Define slicing plane
    plane_origin = [0, 0, layer_height]
    plane_normal = [0, 0, 1]

    # Perform slicing at the desired height
    sections = mesh.section_multiplane(
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        heights=[layer_height]
    )

    contours = []
    if sections:
        for path in sections:
            if path is not None and path.vertices.shape[0] > 1:
                # Get only X, Y
                contours.append(path.vertices[:, :2])

    return contours

    
def segment_mesh_planes(mesh, visualize=False):
    """
    Segment a 3D mesh into exactly 3 dominant planar regions and a non-planar region.
slice_surface
slice_surfacelanar surface reconstruction
    - visualize: If True, displays the segmented planes

    Returns:
    - planes: List of Open3D meshes representing the detected 3 planes
    - non_planar_mesh: Open3D mesh representing the remaining complex surface
    """

    # Convert mesh to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=80000)
    detected_planes = []

    distance_threshold=1
    ransac_n=3
    num_iterations=1000
    alpha=1
    
    for i in range(2):  # Detect only 3 planes
        plane_model, inliers = pcd.segment_plane(distance_threshold,
                                                 ransac_n,
                                                 num_iterations)
        
        if len(inliers) < 100:  # If the detected plane is too small, stop
            print(f"Plane {i+1}: Not enough points detected.")
            break

        # Extract the plane points
        plane_cloud = pcd.select_by_index(inliers)
        
        # Convert to Open3D TriangleMesh (Convex hull for better visualization)
        convex_hull, _ = plane_cloud.compute_convex_hull()
        convex_hull.paint_uniform_color([random.random(), random.random(), random.random()])  # Assign a random color
        
        detected_planes.append(convex_hull)
        
        # Remove plane points and continue
        pcd = pcd.select_by_index(inliers, invert=True)
   
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

    # Apply DBSCAN clustering to remove noise
    labels = np.array(pcd.cluster_dbscan(eps=30, min_points=1000, print_progress=True))  # Adjust eps and min_points as needed
    print(f'labels: {labels}')

    # Remove noise (label = -1 indicates noise in DBSCAN)
    if len(labels) > 0:
        largest_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))  # Get largest cluster
        pcd = pcd.select_by_index(np.where(labels == largest_cluster_label)[0])  # Keep only largest cluster

    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    
    # # Handle remaining points (non-planar surface)
    non_planar_mesh = None
    if len(pcd.points) > 100:  # Ensure enough points remain
        try:
            non_planar_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            if len(non_planar_mesh.triangles) == 0:  # Check if alpha shape worked
                raise ValueError("Alpha shape failed. Using convex hull instead.")
        except:
            print("[WARNING] Alpha shape reconstruction failed. Trying convex hull...")
            non_planar_mesh, _ = pcd.compute_convex_hull()
        
        non_planar_mesh.remove_duplicated_triangles()
        non_planar_mesh.remove_unreferenced_vertices()
        non_planar_mesh.paint_uniform_color([1, 0, 0])  # Color non-planar region red

    # Visualization
    if visualize:
        to_visualize = detected_planes
        o3d.visualization.draw_geometries(to_visualize, window_name="Segmented Mesh (Planes)")
        to_visualize = [non_planar_mesh] if non_planar_mesh else []
        o3d.visualization.draw_geometries(to_visualize, window_name="Segmented Mesh (Damage Surface)")

    return detected_planes, non_planar_mesh

def rotate_points(points, angle, origin):
    """
    Rotates an array of 2D points around a given origin by an angle in degrees.
    """
    if points.size == 0:
        return points
    angle_rad = np.radians(angle)
    origin = np.array([origin.x, origin.y])
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(points - origin, rotation_matrix.T) + origin

# -------------------- Strategies Vis ---------------------------

def debug_print(segment, name):
    if segment is not None:
        print(f"Debug: {name} shape = {segment.shape}")
        print(segment)
    else:
        print(f"Debug: {name} is None")

# def link_layers_and_visualize(toolpaths, layer_height):
#     """
#     Links all layers to create a continuous 3D trajectory and visualizes it.

#     Parameters:
#         toolpaths (list): List of tuples containing toolpaths for each layer.
#                           Each tuple has (deposition, tool_moves, fly_lines, fillets).
#         layer_height (float): Height increment between each layer.
#     """
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     import numpy as np

#     continuous_trajectory = []  # To store the continuous 3D trajectory
#     current_z = 0  # Starting Z-coordinate

#     print("Debug: Starting to process layers...")
#     for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(toolpaths):
#         # Helper function to ensure data is at least 2D
#         def ensure_2d(array):
#             if array.ndim == 1:
#                 return array.reshape(-1, 1)  # Convert 1D array to 2D (single column)
#             return array

#         # Convert 2D paths to 3D by adding the current Z-coordinate
#         if len(deposition) > 0:
#             deposition = ensure_2d(deposition)
#             deposition_3d = np.hstack((deposition[:, :3], np.full((deposition.shape[0], 1), current_z)))  # Ensure deposition is 3D
#             # debug_print(deposition_3d, "Deposition 3D")  # Debug
#             continuous_trajectory.append(deposition_3d)
#             # print(f"Layer {i}, Deposition Points (3D):\n{deposition_3d}")
#         else:
#             print(f"Layer {i}: No deposition points.")

#         # Process tool moves, fly lines, and other components similarly
#         if len(tool_moves) > 0:
#             for move in tool_moves:
#                 move = ensure_2d(move)
#                 move_3d = np.hstack((move[:, :3], np.full((move.shape[0], 1), current_z)))  # Ensure move is 3D
#                 debug_print(move_3d, "Tool Moves 3D")  # Debug
#                 continuous_trajectory.append(move_3d)

#         if len(fly_lines) > 0:
#             for fly in fly_lines:
#                 fly = ensure_2d(fly)
#                 fly_3d = np.hstack((fly[:, :3], np.full((fly.shape[0], 1), current_z)))  # Ensure fly is 3D
#                 # debug_print(fly_3d, "Fly Lines 3D")  # Debug
#                 continuous_trajectory.append(fly_3d)
            
#         if len(fillets) > 0:
#             for fillet in fillets:
#                 fillet = ensure_2d(fillet)
#                 fillet_3d = np.hstack((fillet[:, :3], np.full((fillet.shape[0], 1), current_z)))
#                 continuous_trajectory.append(fillet_3d)

#         # Link this layer to the next by adding a vertical connection in Z
#         if i < len(toolpaths) - 1:  # If not the last layer
#             next_z = current_z + layer_height

#             end_point = deposition_3d[-1][:3] if len(deposition) > 0 else None  # Ensure end_point is 3D
#             next_start_point = (
#                 np.hstack((toolpaths[i + 1][0][0][:3], next_z)) if len(toolpaths[i + 1][0]) > 0 else None
#             )  # Ensure next_start_point is 3D

#             if end_point is not None and next_start_point is not None:
#                 # Debug: Check and log shapes
#                 # print(f"Linking Layer {i} to Layer {i + 1}")
#                 # print(f"  End Point Shape: {end_point.shape}, End Point: {end_point}")
#                 # print(f"  Next Start Point Shape: {next_start_point.shape}, Next Start Point: {next_start_point}")

#                 # Ensure both points are 3D
#                 if len(end_point) == 3 and len(next_start_point) == 3:
#                     vertical_link = np.array([end_point, next_start_point])
#                     continuous_trajectory.append(vertical_link)
#                 else:
#                     print(f"Shape mismatch: end_point={end_point}, next_start_point={next_start_point}")
#             else:
#                 print(f"Skipping vertical link for Layer {i} to Layer {i + 1} due to missing points.")

#         # Increment Z-coordinate for the next layer
#         current_z += layer_height

#     # Flatten the trajectory into a single array for visualization
#     flattened_trajectory = []
#     for segment in continuous_trajectory:
#         if segment.shape[1] == 3:  # Ensure all arrays have exactly 3 columns
#             flattened_trajectory.append(segment)
#         else:
#             print(f"Skipped segment with invalid shape: {segment.shape}")

#     if flattened_trajectory:
#         flattened_trajectory = np.vstack(flattened_trajectory)
#     else:
#         flattened_trajectory = np.empty((0, 3))

#     # Final visualization
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     for segment in continuous_trajectory:
#         if segment.shape[1] == 3:  # Only plot valid segments
#             ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color='b')

#     ax.set_title("3D Toolpath Visualization")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.show()

#     return flattened_trajectory  # Return the full 3D trajectory for further use

def link_layers_and_visualize(toolpaths, layer_height):
    """
    Links all layers to create a continuous 3D trajectory and visualizes it.

    Parameters:
        toolpaths (list): List of tuples containing toolpaths for each layer.
                          Each tuple has (deposition, tool_moves, fly_lines, fillets).
        layer_height (float): Height increment between each layer.
    """

    print("Start linking layers and visualizing...")

    continuous_trajectory = []  # To store the continuous 3D trajectory
    updated_toolpaths = []  # Store the updated toolpaths with vertical connections
    current_z = 0  # Starting Z-coordinate

    print("Debug: Starting to process layers...")
    
    for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(toolpaths):
        # Helper function to ensure data is at least 2D
        def ensure_2d(array):
            if array.ndim == 1:
                return array.reshape(-1, 1)  # Convert 1D array to 2D (single column)
            return array

        # Convert 2D paths to 3D by adding the current Z-coordinate
        if len(deposition) > 0:
            deposition = ensure_2d(deposition)
            deposition_3d = np.hstack((deposition[:, :3], np.full((deposition.shape[0], 1), current_z)))  # Ensure deposition is 3D
            continuous_trajectory.append(deposition_3d)
        else:
            print(f"Layer {i}: No deposition points.")
        
        # Add the deposition 3D to updated toolpaths
        updated_toolpaths.append((deposition_3d, tool_moves, fly_lines, fillets))

        # Process tool moves, fly lines, and other components similarly
        if len(tool_moves) > 0:
            for move in tool_moves:
                move = ensure_2d(move)
                move_3d = np.hstack((move[:, :3], np.full((move.shape[0], 1), current_z)))  # Ensure move is 3D
                continuous_trajectory.append(move_3d)

        if len(fly_lines) > 0:
            for fly in fly_lines:
                fly = ensure_2d(fly)
                fly_3d = np.hstack((fly[:, :3], np.full((fly.shape[0], 1), current_z)))  # Ensure fly is 3D
                continuous_trajectory.append(fly_3d)
            
        if len(fillets) > 0:
            for fillet in fillets:
                fillet = ensure_2d(fillet)
                fillet_3d = np.hstack((fillet[:, :3], np.full((fillet.shape[0], 1), current_z)))
                continuous_trajectory.append(fillet_3d)

        # Link this layer to the next by adding a vertical connection in Z
        if i < len(toolpaths) - 1:  # If not the last layer
            next_z = current_z + layer_height

            end_point = deposition_3d[-1][:3] if len(deposition) > 0 else None  # Ensure end_point is 3D
            next_start_point = (
                np.hstack((toolpaths[i + 1][0][0][:3], next_z)) if len(toolpaths[i + 1][0]) > 0 else None
            )  # Ensure next_start_point is 3D

            if end_point is not None and next_start_point is not None:
                vertical_link = np.array([end_point, next_start_point])
                continuous_trajectory.append(vertical_link)
                
                # Also add the vertical link to the updated toolpaths
                updated_toolpaths.append((np.array([end_point]), np.array([next_start_point]), [], []))
            else:
                print(f"Skipping vertical link for Layer {i} to Layer {i + 1} due to missing points.")

        # Increment Z-coordinate for the next layer
        current_z += layer_height

    # Flatten the trajectory into a single array for visualization
    flattened_trajectory = []
    for segment in continuous_trajectory:
        if segment.shape[1] == 3:  # Ensure all arrays have exactly 3 columns
            flattened_trajectory.append(segment)
        else:
            print(f"Skipped segment with invalid shape: {segment.shape}")

    if flattened_trajectory:
        flattened_trajectory = np.vstack(flattened_trajectory)
    else:
        flattened_trajectory = np.empty((0, 3))

    # Final visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for segment in continuous_trajectory:
        if segment.shape[1] == 3:  # Only plot valid segments
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color='b')

    ax.set_title("3D Toolpath Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    return flattened_trajectory, updated_toolpaths

def visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets):
    """
    Visualizes the toolpath in 2D, showing deposition and tool moves in different colors.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot deposition lines (horizontal) in blue
    # if deposition.size > 0:
    #     for i in range(0, deposition.shape[0], 2):  # Each line is defined by 2 points
    #         plt.plot(
    #             deposition[i:i+2, 0], 
    #             deposition[i:i+2, 1], 
    #             '-o', color='blue', label='Deposition' if i == 0 else ""
    #         )
  
    # Plot deposition lines (as contours) in blue
    if deposition.size > 0:
        # Assuming deposition points are grouped by contours
        plt.plot(
            deposition[:, 0], 
            deposition[:, 1], 
            '-o', color='blue', label='Deposition Contour'
        )

    # Plot tool moves (diagonal) in red 
    if tool_moves is not None and len(tool_moves) > 0:
        for i in range(0, tool_moves.shape[0], 2):  # Each tool move is defined by 2 points
            plt.plot(
                tool_moves[i:i+2, 0], 
                tool_moves[i:i+2, 1], 
                '--o', color='red', label='Tool Moves' if i == 0 else ""
            )


    if fly_lines.size > 0:
        for i in range(0, fly_lines.shape[0], 2):  # Each tool move is defined by 2 points
            plt.plot(
                fly_lines[i:i+2, 0], 
                fly_lines[i:i+2, 1], 
                '--o', color='green', label='Fly-in/out Moves' if i == 0 else ""
            )   

    if fillets.size > 0:
        for i in range(0, fillets.shape[0], 2):  # Each tool move is defined by 2 points
            plt.plot(
                fillets[i:i+2, 0], 
                fillets[i:i+2, 1], 
                '-o', color='yellow', label='Fillet Moves' if i == 0 else ""
            )   
   

    
    # Add labels, legend, and styling2
    plt.axis('equal')
    plt.legend()
    plt.title('2D Toolpath Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def align_meshes_icp(mesh, non_planar_mesh, max_correspondence_distance=0.5, icp_threshold=1e-6):
    """
    Aligns the non-planar mesh to the reference mesh using ICP.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The reference mesh.
        non_planar_mesh (o3d.geometry.TriangleMesh): The mesh to be aligned.
        max_correspondence_distance (float): Maximum distance for point correspondences.
        icp_threshold (float): Convergence threshold for ICP.

    Returns:
        aligned_mesh (o3d.geometry.TriangleMesh): The aligned non-planar mesh.
        transformation (np.ndarray): The 4x4 transformation matrix.
    """
    # Convert meshes to point clouds
    source_pcd = mesh.sample_points_uniformly(number_of_points=200000)
    target_pcd = non_planar_mesh.sample_points_uniformly(number_of_points=200000)

    # Initial alignment (center & scale matching)
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    # Perform ICP registration
    icp_result = o3d.pipelines.registration.registration_icp(
        target_pcd, source_pcd, max_correspondence_distance,
        np.eye(4),  # Initial guess: identity matrix
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=icp_threshold)
    )

    # Apply transformation to the non-planar mesh
    aligned_mesh = non_planar_mesh.transform(icp_result.transformation)

    return aligned_mesh, icp_result.transformation

def visualize_mesh_with_slice(mesh, non_planar_mesh, layer_height, plane_size=700.0, plane_color=[1, 0, 0], non_planar_mesh_color=[0, 0, 1], mesh_color=[0, 1, 1]):
    """
    Visualizes the mesh with a horizontal plane at the slicing height.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The surface mesh.
        layer_height (float): The height where the slice is taken.
        plane_size (float): The size of the visual slicing plane.
        plane_color (list): RGB color for the plane.
        mesh_color (list): RGB color for the mesh.
    """

    # Color the mesh for better visualization
    mesh.paint_uniform_color(mesh_color)
    non_planar_mesh.paint_uniform_color(non_planar_mesh_color)

    # Create a 3D plane at the slicing height
    plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.001)
    
    # Center the plane at (0, 0, layer_height)
    plane.translate([-plane_size / 2, -plane_size / 2, layer_height])
    
    # Color the plane
    plane.paint_uniform_color(plane_color)

    # Visualize the mesh and the slicing plane
    o3d.visualization.draw_geometries([mesh, plane, non_planar_mesh], window_name="Mesh with Slicing Plane")

def export_trajectory_to_csv(trajectory, filename="3d_trajectory_4_hybrid.csv", simplify=True, epsilon=10):
    """
    Exports the 3D trajectory to a CSV file, with an option to simplify using RDP.

    Parameters:
        trajectory (ndarray): Array of 3D points representing the trajectory.
        filename (str): The name of the output CSV file.
        simplify (bool): Whether to simplify the trajectory points.
        epsilon (float): RDP algorithm tolerance (lower = more detailed).
    """
    if simplify:
        trajectory = rdp_simplify(trajectory, epsilon)

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])  # Header
        for point in trajectory:
            writer.writerow(point)

    print(f"Trajectory exported successfully to {filename}")

def rdp_simplify(trajectory, epsilon):
    """
    Simplifies a 3D trajectory using the Ramer-Douglas-Peucker algorithm.

    Parameters:
        trajectory (ndarray): Array of 3D points representing the trajectory.
        epsilon (float): Maximum allowed distance between the original and simplified points.

    Returns:
        ndarray: Simplified trajectory.
    """
    def rdp_recursive(points, epsilon):
        # Find the point with the maximum distance from the line segment start-end
        start, end = points[0], points[-1]
        max_distance = 0
        index = 0

        for i in range(1, len(points) - 1):
            distance = np.linalg.norm(np.cross(end - start, start - points[i])) / np.linalg.norm(end - start)
            if distance > max_distance:
                max_distance = distance
                index = i

        # If max distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            left = rdp_recursive(points[:index + 1], epsilon)
            right = rdp_recursive(points[index:], epsilon)
            return np.vstack((left[:-1], right))
        else:
            # Keep only start and end points
            return np.array([start, end])

    return rdp_recursive(trajectory, epsilon)

def calculate_total_length(path):
    """Calculate the total length of a path (sum of Euclidean distances between consecutive points)."""
    total_length = 0
    for i in range(1, len(path)):
        total_length += np.linalg.norm(path[i] - path[i-1])
    return total_length
 
def deposition_to_travel_ratio(deposition_length, travel_length):
    """Calculate the Deposition-to-Travel Ratio."""
    if travel_length == 0:
        return float('inf')  # Avoid division by zero
    return deposition_length / travel_length

def analyze_layerwise_metrics(toolpaths, slices):
    """
    Analyze total path length and deposition-to-travel ratio per layer.

    Parameters:
        toolpaths (list): List of toolpath tuples (deposition, tool_moves, fly_lines, fillets).
        slices (list): List of (layer_height, boundary_points) tuples.

    Returns:
        None (prints the values for analysis)
    """
    total_deposition_length = 0
    total_travel_length = 0

    print("\nLayer-wise Metrics:")
    for (layer_height, _), (deposition, tool_moves, _, _) in zip(slices, toolpaths):
        # Calculate lengths
        deposition_length = calculate_total_length(deposition)
        travel_length = calculate_total_length(tool_moves)

        # Compute ratio
        ratio = deposition_to_travel_ratio(deposition_length, travel_length)

        # Accumulate totals
        total_deposition_length += deposition_length
        total_travel_length += travel_length

        # Print per-layer values
        print(f"Layer {layer_height:.3f} mm -> Deposition: {deposition_length:.2f} mm, Travel: {travel_length:.2f} mm, Ratio: {ratio:.3f}")

    # Compute overall values
    overall_ratio = deposition_to_travel_ratio(total_deposition_length, total_travel_length)
    
    print("\nOverall Metrics:")
    print(f"Total Deposition Length: {total_deposition_length:.2f} mm")
    print(f"Total Travel Length: {total_travel_length:.2f} mm")
    print(f"Overall Deposition-to-Travel Ratio: {overall_ratio:.3f}\n")



def calculate_deposition_coverage_per_layer(toolpaths, slices, tool_width):
    """
    Calculates the deposition coverage percentage per layer.
    
    Parameters:
    - toolpaths (list): List of (deposition, tool_moves, fly_lines, fillets) per layer.
    - slices (list): List of (layer_height, boundary_points) tuples, where boundary_points define the expected deposition region.
    - tool_width (float): Width of the deposition tool, used for buffering toolpaths.

    Returns:
    - list of tuples: [(layer_height, coverage_percentage), ...] for each layer.
    """
    coverage_results = []

    for layer_index, ((layer_height, boundary_points), (deposition, _, _, _)) in enumerate(zip(slices, toolpaths)):
        boundary_polygon = Polygon(boundary_points)  # Convert boundary points to a polygon
        boundary_area = boundary_polygon.area

        # Collect deposition paths for this layer
        deposition_lines = []
        if len(deposition) > 1:
            deposition_lines.append(LineString(deposition))  

        if deposition_lines:
            # Merge all deposition paths into a single geometry
            merged_deposition = unary_union(deposition_lines)
            # Convert deposition lines to area by buffering (simulating material spread)
            deposition_polygon = merged_deposition.buffer(tool_width / 2, cap_style=2)  
            # Compute valid deposited area inside the boundary
            valid_deposition_area = deposition_polygon.intersection(boundary_polygon).area
        else:
            valid_deposition_area = 0  # No deposition in this layer

        # Compute coverage percentage
        coverage_percentage = (valid_deposition_area / boundary_area) * 100 if boundary_area > 0 else 0
        coverage_results.append((layer_height, coverage_percentage))
        print(f"Layer {layer_index}: Coverage = {coverage_percentage:.2f}%")

    # Compute and display average coverage
    average_coverage = sum(c[1] for c in coverage_results) / len(coverage_results) if coverage_results else 0
    # print(f"\nAverage Deposition Coverage: {average_coverage:.2f}%\n")

    return coverage_results, average_coverage



def visualize_layer_coverage(slices, toolpaths, coverage_results, tool_width):
    """
    Visualizes deposition coverage for a selected layer.

    - slices: List of (layer_height, boundary_points)
    - toolpaths: List of deposition paths per layer
    - coverage_results: Coverage percentage per layer
    - tool_width: Width of the deposition tool
    """
    while True:
        layer_input = input("Enter layer number to visualize (or 'q' to quit): ")
        if layer_input.lower() == 'q':
            break

        try:
            layer_index = int(layer_input)
            if layer_index < 0 or layer_index >= len(slices):
                print("Invalid layer number. Try again.")
                continue
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")
            continue

        layer_height, boundary_points = slices[layer_index]
        boundary_polygon = Polygon(boundary_points)

        deposition, _, _, _ = toolpaths[layer_index]
        deposition_lines = []
        if len(deposition) > 1:
            deposition_lines.append(LineString(deposition))  

        if deposition_lines:
            merged_deposition = unary_union(deposition_lines)
            deposition_polygon = merged_deposition.buffer(tool_width / 2, cap_style=2)  
        else:
            deposition_polygon = None

        # Plot the layer
        plt.figure(figsize=(6, 6))

        # Plot boundary
        x, y = boundary_polygon.exterior.xy
        plt.plot(x, y, 'r-', label="Boundary")

        # Plot deposition
        if deposition_polygon:
            x, y = deposition_polygon.exterior.xy
            plt.fill(x, y, 'b', alpha=0.5, label="Deposition")

        # Display coverage info
        # coverage_percentage_layer = coverage_results[layer_index][1]

        plt.title(f"Layer {layer_index} (Height {layer_height:.3f})")
        plt.legend()
        plt.axis("equal")
        plt.grid()
        plt.show()


def turn_angle_analysis(toolpaths):
    """Analyze sharp turns by calculating turn angles between consecutive path segments."""
    total_turns = 0
    num_turns = 0  # Count the number of valid turns

    for layer_index, (deposition, _, _, _) in enumerate(toolpaths):
        if len(deposition) > 2:  # Need at least 3 points to form an angle
            for i in range(1, len(deposition) - 1):
                p1, p2, p3 = np.array(deposition[i - 1]), np.array(deposition[i]), np.array(deposition[i + 1])
                
                v1 = p2 - p1  # First segment vector
                v2 = p3 - p2  # Second segment vector
                
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cosine_angle = np.clip(cosine_angle, -1, 1)  # Avoid floating-point errors
                    angle_radians = np.arccos(cosine_angle)  # Get angle in radians
                    angle_degrees = np.degrees(angle_radians)  # Convert to degrees
                    
                    if angle_degrees < 0.1:  # Ignore near-zero turns
                        continue
                    if angle_degrees > 180: 
                        print(angle_degrees)

                    total_turns += angle_degrees
                    num_turns += 1  # Count valid turns

                    # Debugging Output
                    # print(f"Layer {layer_index}, Turn {i}: Angle = {angle_degrees:.2f}°")
                
    # Debug total values
    # print(f"Total Turn Angles: {total_turns:.2f}°, Total Turns: {num_turns}")
    if num_turns == 0:
        return 0    
    else:
        avg_turn_angle = total_turns/num_turns

    # print(f"Average Turn Angle: {avg_turn_angle:.2f}°")

    return avg_turn_angle


# def hausdorff_distance_per_layer(toolpaths, slices, tool_width, visualize=False):
#     """
#     Computes the Hausdorff distance per layer and optionally visualizes deposition vs. expected boundary.

#     Parameters:
#     - toolpaths (list): List of (deposition, tool_moves, fly_lines, fillets) per layer.
#     - slices (list): List of (layer_height, boundary_points) tuples.
#     - tool_width (float): Width of the deposition tool, used for buffering toolpaths.
#     - visualize (bool): If True, plots each layer for debugging.

#     Returns:
#     - list of tuples: [(layer_height, hausdorff_distance), ...] for each layer.
#     """
#     hausdorff_results = []
#     total_distance = 0
#     valid_layers = 0  # Count layers with valid distance computation

#     for layer_index, ((layer_height, boundary_points), (deposition, _, _, _)) in enumerate(zip(slices, toolpaths)):
#         if len(deposition) < 2 or len(boundary_points) < 2:
#             hausdorff_results.append((layer_height, float('inf')))  # No valid data
#             continue

#         # Convert slice boundary points to a NumPy array
#         boundary_array = np.array(boundary_points)

#         # Compute boundary of deposited material
#         deposition_lines = [LineString(deposition)] if len(deposition) > 1 else []
#         if deposition_lines:
#             merged_deposition = unary_union(deposition_lines)  # Merge deposition paths
#             deposition_boundary = merged_deposition.buffer(tool_width / 2, cap_style=2)  # Get boundary

#             if deposition_boundary.is_empty:
#                 hausdorff_results.append((layer_height, float('inf')))
#                 continue

#             # Extract boundary points from the buffered deposition
#             deposition_boundary_points = np.array(deposition_boundary.exterior.coords) if deposition_boundary.geom_type == 'Polygon' else []

#             if len(deposition_boundary_points) == 0:
#                 hausdorff_results.append((layer_height, float('inf')))
#                 continue

#             # Compute Hausdorff Distance
#             d1 = directed_hausdorff(deposition_boundary_points, boundary_array)[0]  # Deposited → Expected
#             d2 = directed_hausdorff(boundary_array, deposition_boundary_points)[0]  # Expected → Deposited
#             hausdorff_dist = max(d1, d2)  # Symmetric Hausdorff distance

#             hausdorff_results.append((layer_height, hausdorff_dist))

#             total_distance += hausdorff_dist
#             valid_layers += 1

#             # Debugging Output
#             # print(f"Layer {layer_index}: Hausdorff Distance = {hausdorff_dist:.2f}")

#             # **Visualization**
#             if visualize:
#                 plt.figure(figsize=(6, 6))
#                 plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'b-', label="Expected Boundary (Slice)")
#                 plt.plot(*deposition_boundary.exterior.xy, 'r-', label="Deposited Boundary (Buffered)")
#                 plt.plot(*zip(*deposition), 'g--', marker='o', label="Deposition Path (Centerline)")
#                 plt.title(f"Layer {layer_index} - Hausdorff Distance: {hausdorff_dist:.2f}")
#                 plt.legend()
#                 plt.axis("equal")
#                 plt.show()

#         else:
#             hausdorff_results.append((layer_height, float('inf')))  # No valid deposition boundary
        
#         # **Compute Average Hausdorff Distance**
#     average_hausdorff = total_distance / valid_layers if valid_layers > 0 else float('inf')
#     # print(f"\n🔹 **Average Hausdorff Distance across layers: {average_hausdorff:.2f}** 🔹\n")

#     return hausdorff_results, average_hausdorff

# def calculate_centroid(points):
#     """Calculate the centroid of a set of 2D points."""
#     points = np.array(points)
#     return np.mean(points, axis=0)

# def radial_sample(points, centroid, num_samples=360):
#     """Sample points on a boundary based on their angle relative to the centroid."""
#     angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
#     indices = np.argsort(angles)

#     sorted_points = points[indices]
#     sorted_angles = angles[indices]

#     sampled_points = []
#     for theta in np.linspace(-np.pi, np.pi, num_samples):
#         closest_idx = np.argmin(np.abs(sorted_angles - theta))
#         sampled_points.append(sorted_points[closest_idx])

#     return np.array(sampled_points)

# def centroid_based_distance_per_layer(toolpaths, slices, tool_width, visualize=False, num_samples=360):
#     """
#     Computes a centroid-based distance measure per layer and optionally visualizes deposition vs. expected boundary.

#     Parameters:
#     - toolpaths (list): List of (deposition, tool_moves, fly_lines, fillets) per layer.
#     - slices (list): List of (layer_height, boundary_points) tuples.
#     - tool_width (float): Width of the deposition tool, used for buffering toolpaths.
#     - visualize (bool): If True, plots each layer for debugging.
#     - num_samples (int): Number of angular samples to take for comparison.

#     Returns:
#     - list of tuples: [(layer_height, average_distance), ...] for each layer.
#     """
#     results = []
#     total_distance = 0
#     valid_layers = 0

#     for layer_index, ((layer_height, boundary_points), (deposition, _, _, _)) in enumerate(zip(slices, toolpaths)):
#         if len(deposition) < 2 or len(boundary_points) < 2:
#             results.append((layer_height, float('inf')))
#             continue

#         # Convert to arrays
#         boundary_array = np.array(boundary_points)
#         deposition_array = np.array(deposition)

#         # Compute boundaries
#         deposition_lines = [LineString(deposition_array)] if len(deposition_array) > 1 else []

#         if deposition_lines:
#             merged_deposition = unary_union(deposition_lines)
#             deposition_boundary = merged_deposition.buffer(tool_width / 2, cap_style=2)

#             if deposition_boundary.is_empty:
#                 results.append((layer_height, float('inf')))
#                 continue

#             deposition_boundary_points = np.array(deposition_boundary.exterior.coords) if deposition_boundary.geom_type == 'Polygon' else []

#             if len(deposition_boundary_points) == 0:
#                 results.append((layer_height, float('inf')))
#                 continue

#             # Calculate centroids
#             centroid_boundary = calculate_centroid(boundary_array)
#             centroid_deposition = calculate_centroid(deposition_boundary_points)

#             # Sample points radially
#             sampled_boundary = radial_sample(boundary_array, centroid_boundary, num_samples)
#             sampled_deposition = radial_sample(deposition_boundary_points, centroid_deposition, num_samples)

#             # Compute point-wise distances
#             distances = np.linalg.norm(sampled_boundary - sampled_deposition, axis=1)

#             # Average distance
#             avg_distance = np.mean(distances)

#             results.append((layer_height, avg_distance))
#             total_distance += avg_distance
#             valid_layers += 1

#             # Visualization
#             if visualize:
#                 plt.figure(figsize=(8, 8))
#                 plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'b-', label="Expected Boundary (Slice)")
#                 plt.plot(*deposition_boundary.exterior.xy, 'r-', label="Deposited Boundary (Buffered)")
#                 plt.scatter(sampled_boundary[:, 0], sampled_boundary[:, 1], c='blue', s=10, label="Sampled Expected")
#                 plt.scatter(sampled_deposition[:, 0], sampled_deposition[:, 1], c='red', s=10, label="Sampled Deposited")

#                 plt.plot(*zip(*deposition), 'g--', marker='o', label="Deposition Path (Centerline)")
#                 plt.scatter(*centroid_boundary, c='cyan', marker='x', label="Centroid Expected")
#                 plt.scatter(*centroid_deposition, c='magenta', marker='x', label="Centroid Deposited")

#                 # Draw lines connecting paired points
#                 for point_b, point_d in zip(sampled_boundary, sampled_deposition):
#                     plt.plot([point_b[0], point_d[0]], [point_b[1], point_d[1]], 'k--', linewidth=0.5)

#                 plt.title(f"Layer {layer_index} - Average Distance: {avg_distance:.2f}")
#                 plt.legend()
#                 plt.axis("equal")
#                 plt.show()

#         else:
#             results.append((layer_height, float('inf')))

#     average_distance = total_distance / valid_layers if valid_layers > 0 else float('inf')

#     return results, average_distance


def calculate_centroid(points):
    """Calculate the centroid of a set of 2D points."""
    points = np.array(points)
    return np.mean(points, axis=0)

def radial_sample(points, centroid, num_samples=360):
    """Sample points on a boundary based on their angle relative to the centroid."""
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    indices = np.argsort(angles)

    sorted_points = points[indices]
    sorted_angles = angles[indices]

    sampled_points = []
    for theta in np.linspace(-np.pi, np.pi, num_samples):
        closest_idx = np.argmin(np.abs(sorted_angles - theta))
        sampled_points.append(sorted_points[closest_idx])

    return np.array(sampled_points)

def centroid_based_distance_per_layer(toolpaths, slices, tool_width, visualize=False, num_samples=360):
    """
    Computes a centroid-based distance measure per layer and optionally visualizes deposition vs. expected boundary.

    Parameters:
    - toolpaths (list): List of (deposition, tool_moves, fly_lines, fillets) per layer.
    - slices (list): List of (layer_height, boundary_points) tuples.
    - tool_width (float): Width of the deposition tool, used for buffering toolpaths.
    - visualize (bool): If True, plots each layer for debugging.
    - num_samples (int): Number of angular samples to take for comparison.

    Returns:
    - list of tuples: [(layer_height, average_distance), ...] for each layer.
    """
    results = []
    total_distance = 0
    valid_layers = 0

    for layer_index, ((layer_height, boundary_points), (deposition, _, _, _)) in enumerate(zip(slices, toolpaths)):
        if len(deposition) < 2 or len(boundary_points) < 2:
            results.append((layer_height, float('inf')))
            continue

        # Convert to arrays
        boundary_array = np.array(boundary_points)
        deposition_array = np.array(deposition)

        # Compute boundaries
        deposition_lines = [LineString(deposition_array)] if len(deposition_array) > 1 else []

        if deposition_lines:
            merged_deposition = unary_union(deposition_lines)
            deposition_boundary = merged_deposition.buffer(tool_width / 2, cap_style=2)

            if deposition_boundary.is_empty:
                results.append((layer_height, float('inf')))
                continue

            deposition_boundary_points = np.array(deposition_boundary.exterior.coords) if deposition_boundary.geom_type == 'Polygon' else []

            if len(deposition_boundary_points) == 0:
                results.append((layer_height, float('inf')))
                continue

            # Calculate centroids
            centroid_boundary = calculate_centroid(boundary_array)
            centroid_deposition = calculate_centroid(deposition_boundary_points)

            # Sample points radially
            sampled_boundary = radial_sample(boundary_array, centroid_boundary, num_samples)
            sampled_deposition = radial_sample(deposition_boundary_points, centroid_deposition, num_samples)

            # For each point on the expected boundary, find the closest point on the deposited boundary
            min_distances = []
            closest_pairs = []
            for point in sampled_boundary:
                dists = np.linalg.norm(sampled_deposition - point, axis=1)
                min_idx = np.argmin(dists)
                min_distances.append(dists[min_idx])
                closest_pairs.append((point, sampled_deposition[min_idx]))

            min_distances = np.array(min_distances)

            # Filter distances using standard deviation
            mean_dist = np.mean(min_distances)
            std_dist = np.std(min_distances)
            filtered_distances = min_distances[np.abs(min_distances - mean_dist) <= 2 * std_dist]

            # Average distance after filtering
            avg_distance = np.mean(filtered_distances) if len(filtered_distances) > 0 else float('inf')

            results.append((layer_height, avg_distance))
            total_distance += avg_distance
            valid_layers += 1

            # Visualization
            if visualize:
                plt.figure(figsize=(8, 8))
                plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'b-', label="Expected Boundary (Slice)")
                plt.plot(*deposition_boundary.exterior.xy, 'r-', label="Deposited Boundary (Buffered)")
                plt.scatter(sampled_boundary[:, 0], sampled_boundary[:, 1], c='blue', s=10, label="Sampled Expected")
                plt.scatter(sampled_deposition[:, 0], sampled_deposition[:, 1], c='red', s=10, label="Sampled Deposited")

                plt.plot(*zip(*deposition), 'g--', marker='o', label="Deposition Path (Centerline)")
                plt.scatter(*centroid_boundary, c='cyan', marker='x', label="Centroid Expected")
                plt.scatter(*centroid_deposition, c='magenta', marker='x', label="Centroid Deposited")

                # Draw lines connecting the closest point pairs
                for point_b, point_d in closest_pairs:
                    plt.plot([point_b[0], point_d[0]], [point_b[1], point_d[1]], 'k--', linewidth=0.5)

                plt.title(f"Layer {layer_index} - Average Distance: {avg_distance:.2f}")
                plt.legend()
                plt.axis("equal")
                plt.show()

        else:
            results.append((layer_height, float('inf')))

    average_distance = total_distance / valid_layers if valid_layers > 0 else float('inf')

    return results, average_distance




def calculate_deposition_color_map(toolpaths, slices, tool_width, grid_size=500, contours=None, debug_layer=None):
    color_maps = []
    over_deposition_metrics = []

    for layer_index, ((layer_height, boundary_points), (deposition, _, _, _), (contour)) in enumerate(zip(slices, toolpaths, contours)):
        if debug_layer is not None and layer_index != debug_layer:
            continue
        
        # print(f"Processing layer {layer_index}, Height = {layer_height}")

        boundary_polygon = Polygon(boundary_points)
        grid = np.zeros((grid_size, grid_size))  # Grid for deposition and contour accumulation
        contour_grid = np.zeros((grid_size, grid_size))  # Separate grid to track contour

        # Get grid bounds for mapping
        min_x, min_y, max_x, max_y = boundary_polygon.bounds
        x_range = max_x - min_x
        y_range = max_y - min_y

        # Helper function to map (x, y) to grid coordinates
        def map_to_grid(x, y):
            i = int((x - min_x) / x_range * grid_size)
            j = int((y - min_y) / y_range * grid_size)
            return i, j

        # Process Contour (Wall Zone) First
        ordered_points = sorted(contour, key=lambda x: x[0])
        contour_line = LineString(ordered_points)

        contour_points = list(contour_line.coords)
        try:
            tolerance = tool_width/2

            for k in range(len(contour_points) - 1):
                point1 = contour_points[k]
                point2 = contour_points[k + 1]
                line = LineString([point1, point2])

                for i in range(grid_size):
                    for j in range(grid_size):
                        x = i * (x_range / grid_size) + min_x
                        y = j * (y_range / grid_size) + min_y

                        # Mark the contour on a separate grid
                        if line.distance(Point(x, y)) <= tolerance and boundary_polygon.contains(Point(x, y)):
                            contour_grid[i, j] = 1
                            grid[i, j] = 1  # Initial surface state

        except Exception as e:
            print(f"Skipping invalid contour segment. Error: {e}")
        

        # Rasterize deposition paths
        force_horizontal = False  # Set to True to allow only horizontal lines, False to keep original behavior

        for i in range(len(deposition) - 1):
            try:
                point1 = deposition[i]
                point2 = deposition[i + 1]

                # Enforce horizontal lines if the flag is set
                if force_horizontal and not np.isclose(point1[1], point2[1]):
                    continue  # Skip non-horizontal lines

                line = LineString([point1, point2])

                # Limit to the boundary area
                deposition_area = line.buffer(tool_width / 2, cap_style=2).intersection(boundary_polygon)
                if deposition_area.is_empty:
                    continue

                # Map deposition area to grid
                for i in range(grid_size):
                    for j in range(grid_size):
                        x = i * (x_range / grid_size) + min_x
                        y = j * (y_range / grid_size) + min_y
                        if deposition_area.contains(Point(x, y)):
                            grid[i, j] += 1

            except Exception as e:
                print(f"Skipping invalid deposition path: {point1} -> {point2} - Error: {e}")

        # Compute Wall Engagement Index (WEI)
        support_time_grid = np.zeros_like(grid)  # Stores when each contour pixel was first supported

        for segment_index in range(len(deposition) - 1):
            try:
                point1 = deposition[segment_index]
                point2 = deposition[segment_index + 1]
                line = LineString([point1, point2])

                deposition_area = line.buffer(tool_width / 2, cap_style=2).intersection(boundary_polygon)
                if deposition_area.is_empty:
                    continue

                for i in range(grid_size):
                    for j in range(grid_size):
                        x = i * (x_range / grid_size) + min_x
                        y = j * (y_range / grid_size) + min_y
                        if deposition_area.contains(Point(x, y)):
                            if contour_grid[i, j] == 1 and support_time_grid[i, j] == 0:
                                support_time_grid[i, j] = segment_index + 1  # 1-based index
            except Exception as e:
                print(f"Skipping support time segment: {point1} -> {point2} - Error: {e}")

        # Calculate normalized WEI score
        used_times = support_time_grid[support_time_grid > 0]
        if len(used_times) > 0:
            wall_engagement_index = 1 - (np.mean(used_times) / (len(deposition) - 1))
        else:
            wall_engagement_index = 0



        # print(f"Layer {layer_index}, Contour: {contour}, Type: {type(contour)}")

        # Calculate deposition metrics
        deposition_only_grid = grid - contour_grid
        deposition_only_grid[deposition_only_grid < 0] = 0  # Ensure no negative values

        # Calculate deposition metrics
        deposition_wall_grid = grid + contour_grid
        deposition_wall_grid[deposition_wall_grid < 0] = 0  # Ensure no negative values

        # Mask for boundary area
        boundary_mask = np.zeros((grid_size, grid_size), dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * (x_range / grid_size) + min_x
                y = j * (y_range / grid_size) + min_y
                if boundary_polygon.contains(Point(x, y)):
                    boundary_mask[i, j] = True

        total_deposition_cells = np.sum((deposition_only_grid >= 1) & boundary_mask)
        support_cells = np.sum((deposition_wall_grid == 2) & boundary_mask)
        over_deposition_cells = np.sum((deposition_wall_grid > 2) & boundary_mask)
        zero_deposition_cells = np.sum((deposition_only_grid == 0) & boundary_mask)
        total_cells = np.sum(boundary_mask)

        coverage_percentage = (total_deposition_cells / total_cells) * 100 if total_cells > 0 else 0
        lateral_support_percentage = (support_cells / total_cells) * 100 if total_cells > 0 else 0
        over_deposition_percentage = (over_deposition_cells / total_cells) * 100 if total_cells > 0 else 0

        

        # print(f"Layer {layer_index}: Max Deposition Overlap = {grid.max()}, Coverage = {coverage_percentage:.2f}%, Support Deposition = {lateral_support_percentage:.2f}%, Over-Deposition = {over_deposition_percentage:.2f}%, Zero Deposition = {zero_deposition_cells}")
        
        # Compute contour support effectiveness
        contour_pixels = (contour_grid == 1)
        supporting_contour_pixels = (contour_pixels & (grid > 1))
        total_contour_pixels = np.sum(contour_pixels)
        supporting_contour_pixel_count = np.sum(supporting_contour_pixels)

        supporting_contour_percentage = (supporting_contour_pixel_count / total_contour_pixels * 100
                                        if total_contour_pixels > 0 else 0)

        # print(f"Layer {layer_index}: Supporting Contour = {supporting_contour_percentage:.2f}%")
        print(f"Layer {layer_index}: Max Deposition Overlap = {grid.max()}, Coverage = {coverage_percentage:.2f}%, Support Deposition = {lateral_support_percentage:.2f}%, Over-Deposition = {over_deposition_percentage:.2f}%, Zero Deposition = {zero_deposition_cells}, Supporting Contour = {supporting_contour_percentage:.2f}%, Wall Engagement Index = {wall_engagement_index:.2f}")



        color_maps.append(grid)
        over_deposition_metrics.append(over_deposition_percentage)

        visualize_debug_layer(color_maps[0], slices, debug_layer, contours, min_x, min_y, max_x, max_y)

        visualize_deposition_flow(deposition, slices, layer_index, contour, tool_width)

    return color_maps, coverage_percentage, lateral_support_percentage, over_deposition_percentage, supporting_contour_percentage, wall_engagement_index



def visualize_color_map(color_maps):
    for layer_index, grid in enumerate(color_maps):
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='plasma', interpolation='nearest')  # Plasma highlights intensity better
        plt.colorbar(label='Deposition Count')
        plt.title(f"Layer {layer_index} Deposition Map")
        plt.show()

def visualize_debug_layer(grid, slices, layer_index, contours, min_x, min_y, max_x, max_y):
    plt.figure(figsize=(8, 8))

    # Mirror the grid horizontally (left-right)
    mirrored_grid = np.fliplr(grid)


    # Rotate the grid 90 degrees counterclockwise (CCW)
    rotated_grid = np.rot90(mirrored_grid)

   
    # Define the real-world extent in mm
    extent = [min_x, max_x, min_y, max_y]

    layer_height, boundary_points = slices[layer_index]
    boundary_polygon = Polygon(boundary_points)

    # Plot boundary
    x, y = boundary_polygon.exterior.xy
    plt.plot(x, y, 'r-', label="Boundary")

    contour = contours[layer_index]
    ordered_points = sorted(contour, key=lambda x: x[0])
    contour_line = LineString(ordered_points)
    contour_points = list(contour_line.coords)
    x, y = zip(*contour_points)

    plt.plot(x, y, 'g-', label="Contour")

    plt.imshow(rotated_grid, cmap='plasma', interpolation='nearest', extent=extent, origin="lower")
    plt.colorbar(label='Deposition Count')
    plt.title(f"Debug: Layer {layer_index} Deposition Map (mm)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.legend()
    plt.show()

def visualize_deposition_flow(deposition, slices, layer_index, contour, tool_width):
    # Extract boundary
    layer_height, boundary_points = slices[layer_index]
    boundary_polygon = Polygon(boundary_points)

    # Get bounds for axis limits
    min_x, min_y, max_x, max_y = boundary_polygon.bounds

    # Plot setup
    plt.figure(figsize=(8, 8))

    # Plot boundary
    x, y = boundary_polygon.exterior.xy
    # plt.plot(x, y, 'r-', label="Boundary")

    # Plot contour
    ordered_points = sorted(contour, key=lambda x: x[0])
    contour_line = LineString(ordered_points)
    x, y = zip(*list(contour_line.coords))
    plt.plot(x, y, 'g--', label="Contour")

    # Convert deposition path into segments
    points = np.array(deposition)
    if len(points) < 2:
        print("Not enough deposition points to visualize flow.")
        return
    
    segments = np.array([points[i:i+2] for i in range(len(points) - 1)])
    cmap_values = np.linspace(0, 1, len(segments))

    # Create color-mapped LineCollection
    lc = LineCollection(segments, cmap='viridis', array=cmap_values, linewidth=2)
    plt.gca().add_collection(lc)

    # Plot start and end points for clarity
    plt.plot(*points[0], 'ko', label='Start')
    plt.plot(*points[-1], 'ks', label='End')

    # Final plot settings
    plt.colorbar(lc, label='Deposition Flow (start → end)')
    plt.title(f"Deposition Flow Visualization - Layer {layer_index}")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluation_metrics(toolpaths, layer_height, hatch_distance, tool_width, slices, contours):
    """Evaluate toolpath performance based on different metrics."""
    start_time = time.time()

    # Metric 1: Efficiency - Total Path Length
    total_deposition_length = sum(calculate_total_length(deposition) for deposition, _, _, _ in toolpaths)
   
    total_travel_length = sum(calculate_total_length(tool_move) for _, tool_move, _, _ in toolpaths)
    
    # Metric 2: Efficiency - Deposition-to-Travel Ratio
    ratio = deposition_to_travel_ratio(total_deposition_length, total_travel_length)
    # analyze_layerwise_metrics(toolpaths, slices)
    
    # # Metric 3: Coverage - Deposition Coverage %
    # coverage, coverage_avr = calculate_deposition_coverage_per_layer(toolpaths, slices, tool_width)
    # visualize_layer_coverage(slices, toolpaths, coverage_avr, tool_width)
    
    # Metric 4: Smoothness - Turn Angle Analysis (degrees)
    turn_angle = turn_angle_analysis(toolpaths) # Convert radian to degrees
   
    # Metric 5: Accuracy - Hausdorff Distance (Deviation)
    # hausdorff_result, hausdorff_avr = hausdorff_distance_per_layer(toolpaths, slices, tool_width, True)

    # Metric 5: Accuracy - Centroid-Aligned Nearest Point Deviation (CANPD) (Deviation)
    CANPD_result, CANPD_avr = centroid_based_distance_per_layer(toolpaths, slices, tool_width, visualize=False, num_samples=360)
    
    # Metric 6: 
    debug_layer = 7
    color_maps, coverage_percentage, lateral_support_percentage, over_deposition_percentage, supporting_contour_percentage, wall_engagement_index = calculate_deposition_color_map(toolpaths, slices, tool_width, grid_size=100, contours=contours, debug_layer=debug_layer)

    visualize_color_map(color_maps)
    


    # Results Table
    print("\nEvaluation Metrics Table:")
    print(f"{'Metric':<30} {'Desired Value':<20} {'Calculated Value':<10}")
    print("=" * 80)

    metrics = [
        ("Deposition Length", "None", total_deposition_length),
        ("Travel Length", "Lower", total_travel_length),
        ("Dep-to-Travel Ratio", "Higher", ratio),
        ("Average Turn Angle", "Lower", turn_angle),
        ("Centroid-Aligned Nearest Point Deviation", "Lower", CANPD_avr),
        ("Coverage %", "Higher", coverage_percentage),
        ("Over-Deposition %", "Lower", over_deposition_percentage),
        ("Supported Deposition %", "Higher", lateral_support_percentage),
        ("Supporting Contour %", "Higher", supporting_contour_percentage)
    ]

    for metric in metrics:
        print(f"{metric[0]:<30} {metric[1]:<20} {metric[2]:<10.4f}")
    
    # Save metrics to a text file
    output_file = "evaluation_metrics.txt"

    with open(output_file, "w") as f:
        f.write("Evaluation Metrics Table:\n")
        f.write(f"{'Metric':<30} {'Desired Value':<20} {'Calculated Value':<10}\n")
        f.write("=" * 80 + "\n")
        for metric in metrics:
            f.write(f"{metric[0]:<30} {metric[1]:<20} {metric[2]:<10.4f}\n")

    print(f"\nMetrics saved to {output_file}")


# Step 1: Densify boundary
def densify_boundary(points, step=1.0):
    densified = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        dist = np.linalg.norm(p2 - p1)
        num_segments = max(int(dist // step), 1)
        for j in range(num_segments):
            interp = p1 + (p2 - p1) * (j / num_segments)
            densified.append(interp)
    return np.array(densified)



def fit_largest_circles(polygon, boundary_points, num_circles=3, resolution=0.5, tolerance=0.1, debug=False):
    """
    Fit circles inside the polygon such that they touch exactly two *well-separated* points on the boundary.

    Returns:
    - circles: List of tuples (center_x, center_y, radius)
    - circles_with_boundary_points: List of tuples (center_x, center_y, radius, [p1, p2])
    """

    minx, miny, maxx, maxy = polygon.bounds
    circles = []
    circles_with_boundary_points = []

    if debug:
        fig, ax = plt.subplots()
        x, y = boundary_points[:, 0], boundary_points[:, 1]
        ax.plot(x, y, 'b-', label='Boundary')

    for _ in range(num_circles):
        best_circle = None
        max_radius = 0

        x_vals = np.arange(minx, maxx, resolution)
        y_vals = np.arange(miny, maxy, resolution)

        for x in x_vals:
            for y in y_vals:
                center = Point(x, y)

                if not polygon.contains(center):
                    continue

                distance_to_boundary = polygon.exterior.distance(center)

                # Step 1: Find candidate boundary points close to the expected radius
                dists = np.linalg.norm(boundary_points - np.array([x, y]), axis=1)
                indices_close = np.where(np.abs(dists - distance_to_boundary) < tolerance)[0]
                candidate_points = boundary_points[indices_close]

                found_pair = False
                for i in range(len(candidate_points)):
                    for j in range(i + 1, len(candidate_points)):
                        p1 = candidate_points[i]
                        p2 = candidate_points[j]

                        # Arc distance filter (approximate): use index spacing
                        index_diff = abs(indices_close[i] - indices_close[j])
                        if index_diff < 10:  # <-- adjust this threshold if needed
                            continue

                        # Check if the points are nearly opposite relative to center
                        v1 = p1 - np.array([x, y])
                        v2 = p2 - np.array([x, y])
                        v1 /= np.linalg.norm(v1)
                        v2 /= np.linalg.norm(v2)
                        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

                        if angle < np.deg2rad(150):  # optional stricter filter
                            continue

                        near_points = np.array([p1, p2])
                        found_pair = True
                        break
                    if found_pair:
                        break

                if not found_pair:
                    continue

                # Check for overlap with previous circles
                if any(center.distance(Point(c[0], c[1])) < (c[2] + distance_to_boundary) for c in circles):
                    continue

                # Accept this as the best so far
                if distance_to_boundary > max_radius:
                    max_radius = distance_to_boundary
                    best_circle = (x, y, distance_to_boundary, near_points)

        if best_circle:
            circles.append(best_circle[:3])
            circles_with_boundary_points.append(best_circle)

            if debug:
                circle_x, circle_y, r, near_pts = best_circle
                ax.add_patch(plt.Circle((circle_x, circle_y), r, color='blue', alpha=0.3))
                if len(circles) == 1:  # Plot labels only for the first circle
                    ax.plot(circle_x, circle_y, 'ko', label='Circle Centers')
                    ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx', label='Boundary Points')
                else:  # Plot without labels for subsequent circles
                    ax.plot(circle_x, circle_y, 'ko')
                    ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx')

    if debug:
        ax.set_aspect('equal')
        ax.set_title("Fitted Circles", fontsize=20)
        ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
        ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(loc="upper left", fontsize=12)
        # ax.legend(loc="upper left", fontsize=16)
        ax.grid(True)
        # plt.tight_layout()
        # plt.show()
        # ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust to fit legend at side
        plt.show()

    return circles, circles_with_boundary_points

def compute_curvature_influence(fitted_circles, num_samples=500):

    # Extract and sort centers by Y (ascending)
    centers = np.array([[c[0], c[1]] for c in fitted_circles])
    centers = centers[np.argsort(centers[:, 1])]  # Sort by Y axis (bottom to top)
    
    x, y = centers[:, 0], centers[:, 1]

    # Ensure enough points for spline degree
    m = len(centers)
    if m < 2:
        raise ValueError("Need at least 2 circle centers to fit a curve.")
    k = min(3, m - 1)  # Use lower spline degree if fewer points

    # Fit spline to sorted centers
    tck, u = splprep([x, y], s=0, k=k)

    # Sample along spline
    u_fine = np.linspace(0, 1, num_samples)
    x_spline, y_spline = splev(u_fine, tck)
    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)

    # Curvature calculation
    curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    curvature = np.nan_to_num(curvature)

    # Find curvature at each original (sorted) center
    spline_points = np.vstack((x_spline, y_spline)).T
    tree = cKDTree(spline_points)
    curvature_values = [curvature[tree.query(center)[1]] for center in centers]

    return centers, curvature_values, x_spline, y_spline

def filter_centers_by_curvature(centers, curvature_values, z_thresh=2.0):
    """
    Filters out centers with curvature values beyond z_thresh standard deviations from the mean.

    Parameters:
        centers (Nx2 array): circle centers
        curvature_values (list): curvature values at each center
        z_thresh (float): number of standard deviations to tolerate

    Returns:
        filtered_centers (Nx2 array): centers within threshold
        filtered_curvatures (list): associated curvature values
    """

    curvature_array = np.abs(np.array(curvature_values))
    mean_curv = np.mean(curvature_array)
    std_curv = np.std(curvature_array)

    # Z-score threshold filtering
    z_scores = (curvature_array - mean_curv) / std_curv
    keep_mask = np.abs(z_scores) <= z_thresh

    filtered_centers = centers[keep_mask]
    filtered_curvatures = curvature_array[keep_mask]

    return filtered_centers, filtered_curvatures

def plot_filtered_result(boundary_points, fitted_circles, centers, curvature_values, filtered_centers, filtered_curvatures, x_spline, y_spline):
  

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot boundary
    x_b, y_b = boundary_points[:, 0], boundary_points[:, 1]
    # ax.plot(x_b, y_b, 'b-', label='Boundary')
    ax.plot(x_b, y_b, 'b-')

    # Plot original fitted circles
    for (cx, cy, r) in fitted_circles:
        circle = plt.Circle((cx, cy), r, color='blue', alpha=0.2)
        ax.add_patch(circle)
    ax.scatter(centers[:, 0], centers[:, 1], c='gray', s=60, label='', alpha=0.4)

    # Plot fitted spline
    ax.plot(x_spline, y_spline, 'k--', linewidth=1.2, label='B-spline')

    # Plot filtered centers (colored by curvature)
    sc = ax.scatter(filtered_centers[:, 0], filtered_centers[:, 1],
                    c=np.abs(filtered_curvatures), cmap='plasma', s=100,
                    edgecolors='black')

    # plt.colorbar(sc, ax=ax, label='Curvature Deviation', fontsize=18)
    # ax.set_aspect('equal')
    # ax.set_title("Filtered Centers Based on Curvature Deviation", fontsize=20)
    # ax.legend()
    # ax.grid(True)
    # plt.show()

    # Colorbar with larger font
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Curvature Deviation', fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    # Formatting
    ax.set_aspect('equal')
    ax.set_title("Filtered Centers Based on Curvature Deviation", fontsize=20)
    ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)

    # ax.legend(fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=12)
    # ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space on right for legend
    plt.show()

def refit_spline_from_filtered_centers(filtered_centers, num_samples=500):

    # Ensure centers are sorted from bottom (Y min) to top (Y max)
    filtered_centers = filtered_centers[np.argsort(filtered_centers[:, 1])]
    x, y = filtered_centers[:, 0], filtered_centers[:, 1]

    # Spline degree check
    m = len(filtered_centers)
    if m < 2:
        raise ValueError("Need at least 2 filtered centers to fit a spline.")
    k = min(3, m - 1)

    # Fit spline
    tck, u = splprep([x, y], s=0, k=k)
    u_fine = np.linspace(0, 1, num_samples)
    x_spline, y_spline = splev(u_fine, tck)

    return x_spline, y_spline

def fit_left_right_splines_from_circle_contacts(filtered_circles, smoothing=0):
    left_points = []
    right_points = []

    for cx, cy, r, contact_points in filtered_circles:
        point1, point2 = contact_points
        dx1 = point1[0] - cx
        dx2 = point2[0] - cx

        if dx1 < dx2:
            left_points.append(point1)
            right_points.append(point2)
        else:
            left_points.append(point2)
            right_points.append(point1)

    left_points = np.array(left_points)
    right_points = np.array(right_points)

    left_points = left_points[np.argsort(left_points[:, 1])]
    right_points = right_points[np.argsort(right_points[:, 1])]

    def fit_spline(points):
        if len(points) < 2:
            return None  # Not enough points
        k = min(3, len(points) - 1)
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing, k=k)
        u_fine = np.linspace(0, 1, 100)
        return splev(u_fine, tck)

    left_spline = fit_spline(left_points)
    right_spline = fit_spline(right_points)

    return left_spline, right_spline, left_points, right_points

def plot_splines_with_points(boundary_points, left_spline, right_spline, left_points, right_points):
    plt.figure(figsize=(8, 8))
    # Plot boundary and splines
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-')
    plt.plot(left_spline[0], left_spline[1], 'g-', label='Left Spline')
    plt.plot(right_spline[0], right_spline[1], 'r-', label='Right Spline')
    plt.plot(left_points[:, 0], left_points[:, 1], 'gx', label='Left Points')
    plt.plot(right_points[:, 0], right_points[:, 1], 'rx', label='Right Points')

    ax = plt.gca()
    ax.set_aspect('equal')

    # Set axis labels and title
    ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
    ax.set_title("Splines from Circle Contacts", fontsize=20)

    # Set ticks size
    ax.tick_params(axis='both', labelsize=14)

    # Set legend at the top left
    ax.legend(loc='upper left', fontsize=12)

    # Show grid
    ax.grid(True)

    # Show the plot
    plt.show()

def plot_splines_with_points_2(boundary_points, left_spline, right_spline, left_points, right_points, 
                             x_mid_ext, y_mid_ext, x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection_x, intersection_y):

    plt.figure(figsize=(8, 8))

    # Plot the boundary
    # plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-', label='Boundary')
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-')

    # Plot the original left and right splines
    # plt.plot(left_spline[0], left_spline[1], 'g-', label='Left Spline')
    # plt.plot(right_spline[0], right_spline[1], 'r-', label='Right Spline')

    # Plot the extended splines
    plt.plot(x_left_ext, y_left_ext, 'g--', label='Extended Left Spline')
    plt.plot(x_right_ext, y_right_ext, 'r--', label='Extended Right Spline')
    # plt.plot(x_mid_ext, y_mid_ext, 'g--', label='Extended Middle Spline')  # Extended middle spline in green

    # Plot the points on the left and right splines
    # plt.plot(left_points[:, 0], left_points[:, 1], 'gx', label='Left Points')
    # plt.plot(right_points[:, 0], right_points[:, 1], 'rx', label='Right Points')

    plt.plot(intersection_x, intersection_y, 'bo', markersize=5, label='Intersection Point')

    # Formatting
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
    ax.set_title("Splines with Extensions", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True)

    # Show the plot
    plt.show()


def extend_spline(spline, left_extension=10, right_extension=10, step=0.1):
    """
    Extend a 2D spline represented either as (x, y) or as [x_array, y_array].
    
    Args:
        spline: Either (x, y) as two separate 1D arrays, or [x_array, y_array] as a list or 2xN array.
        left_extension: Distance to extend at the beginning.
        right_extension: Distance to extend at the end.
        step: Resolution of the extension.
        
    Returns:
        x_ext, y_ext: The extended x and y arrays.
    """
    # Normalize input to (x, y)
    if isinstance(spline, (list, np.ndarray)) and len(spline) == 2:
        x, y = np.array(spline[0]), np.array(spline[1])
    else:
        x, y = np.array(spline[0]), np.array(spline[1])  # assuming tuple (x, y)

    # Compute tangent vectors
    dx_left = x[1] - x[0]
    dy_left = y[1] - y[0]
    norm_left = np.hypot(dx_left, dy_left)
    dx_left /= norm_left
    dy_left /= norm_left

    dx_right = x[-1] - x[-2]
    dy_right = y[-1] - y[-2]
    norm_right = np.hypot(dx_right, dy_right)
    dx_right /= norm_right
    dy_right /= norm_right

    # Generate extension points
    t_left = np.arange(step, left_extension + step, step)
    x_left = x[0] - t_left * dx_left
    y_left = y[0] - t_left * dy_left

    t_right = np.arange(step, right_extension + step, step)
    x_right = x[-1] + t_right * dx_right
    y_right = y[-1] + t_right * dy_right

    # Concatenate
    x_ext = np.concatenate([x_left[::-1], x, x_right])
    y_ext = np.concatenate([y_left[::-1], y, y_right])

    return x_ext, y_ext

def compute_spline_intersection(x_left, y_left, x_right, y_right):
    """
    Compute the intersection point between two splines represented by x/y coordinates.
    """
    # Create LineString objects from left and right spline coordinates
    left_line = LineString(np.column_stack((x_left, y_left)))
    right_line = LineString(np.column_stack((x_right, y_right)))

    # Compute the intersection using shapely
    intersection = left_line.intersection(right_line)

    # Handle different types of intersection results
    if intersection.is_empty:
        print("No intersection found.")
        return None
    elif intersection.geom_type == 'Point':
        return intersection.x, intersection.y
    else:
        print(f"Intersection is of type {intersection.geom_type}, handling not implemented.")
        return None


def rebuild_boundary_with_spline_top(boundary_points, point1, point2, 
                                     x_left_ext, y_left_ext, 
                                     x_right_ext, y_right_ext, 
                                     intersection):
    point1 = np.array(point1)
    point2 = np.array(point2)
    intersection = np.array(intersection)

    # 1. Find the highest Y between point1 and point2
    highest_y = max(point1[1], point2[1])

    # 2. Trim the boundary: keep only points below that height
    trimmed = boundary_points[boundary_points[:, 1] <= highest_y]

    # 3. Prepare left and right spline extensions
    left_ext = np.column_stack([x_left_ext, y_left_ext])
    right_ext = np.column_stack([x_right_ext, y_right_ext])

    # Slice left spline: from point1 to intersection
    idx1 = np.argmin(np.linalg.norm(left_ext - point1, axis=1))
    idx2 = np.argmin(np.linalg.norm(left_ext - intersection, axis=1))
    if idx1 > idx2:
        left_segment = left_ext[idx1:idx2-1:-1]
    else:
        left_segment = left_ext[idx1:idx2+1]

    # Slice right spline: from intersection to point2
    idx1 = np.argmin(np.linalg.norm(right_ext - intersection, axis=1))
    idx2 = np.argmin(np.linalg.norm(right_ext - point2, axis=1))
    if idx1 > idx2:
        right_segment = right_ext[idx1:idx2-1:-1]
    else:
        right_segment = right_ext[idx1:idx2+1]

    # 4. Assemble full boundary
    new_boundary = np.vstack([
        trimmed,
        left_segment,
        intersection[np.newaxis, :],
        right_segment
    ])

    return new_boundary



def plot_updated_boundary(boundary, point1, point2, intersection):
    plt.figure(figsize=(8, 8))

    # Plot the final boundary polygon
    plt.plot(boundary[:, 0], boundary[:, 1], 'b--', label='Updated Boundary')

    # Close the polygon visually
    plt.plot([boundary[-1, 0], boundary[0, 0]], [boundary[-1, 1], boundary[0, 1]], 'b--')

    # Plot key points (optional, currently commented out)
    # plt.plot(*point1, 'bo', label='Point 1 (Left Circle Contact)')
    # plt.plot(*point2, 'ro', label='Point 2 (Right Circle Contact)')
    # plt.plot(*intersection, 'go', label='Intersection')

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
    ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_title("Final Trimmed Boundary with Spline Extensions", fontsize=20)

    plt.tight_layout()
    plt.show()


def sort_points_counterclockwise(points):
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    sort_order = np.argsort(angles)
    return points[sort_order]



# --------------------------------------------------------------------
# ---------------------- MAIN ----------------------------------------
# --------------------------------------------------------------------

def main():
    print("=== Additive Manufacturing Toolpath Generation for Repair ===")

    # Load the mesh file
    mesh_file = "/home/miguel-cruz/Documents/AS4RCSD/Dataset/Boolean Comparison/Case5_mv.STL"
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # damaged_mesh_file = "/home/miguel-cruz/Documents/Dataset/Case5 D.STL"
    # damaged_mesh = o3d.io.read_triangle_mesh(damaged_mesh_file)


    non_planar_surface_file = "/home/miguel-cruz/Documents/AS4RCSD/Dataset/Boolean Comparison/Case5_non_planar_surface.STL"
    non_planar_mesh = o3d.io.read_triangle_mesh(non_planar_surface_file)


    vertices = np.asarray(non_planar_mesh.vertices)
    faces = np.asarray(non_planar_mesh.triangles)
    non_planar_mesh_for_contour = trimesh.Trimesh(vertices=vertices, faces=faces)
   
  
    # Apply colors (optional for better visualization)
    mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
    non_planar_mesh.paint_uniform_color([1.0, 0.3, 0.3])  # Red
    # damaged_mesh.paint_uniform_color([0.0, 0.7, 0.0])  # Green
    # o3d.visualization.draw_geometries(
    #     [damaged_mesh], 
    #     window_name="Damaged Mesh"
    # )


    # Visualize the aligned meshes
    o3d.visualization.draw_geometries(
        [mesh, non_planar_mesh], 
        window_name="Mesh + Non-Planar Surface + Damaged Mesh"
    )

 
    # User Input: Choose slicing axis
    axis_map = {"x": 0, "y": 1, "z": 2}
    slicing_axis = "z"  # Default slicing axis

    new_axis = input("Enter a slicing axis (x, y, or z): ").strip().lower()
    if new_axis in axis_map:
        slicing_axis = new_axis
    else:
        print("Invalid input. Defaulting to Z-axis.")

    # User Input: Rotate 180°?
    rotate_180 = input("Do you want to rotate the model 180°? (yes/no): ").strip().lower() == "yes"

    # # Apply necessary rotations based on slicing direction
    # if slicing_axis == "x":
    #     print("Rotating mesh to align X with Z...")
    #     mesh = rotate_mesh(mesh, "y", 90)  # Rotate around Y-axis
    #     non_planar_mesh = rotate_mesh(non_planar_mesh, "y", 90)
    # elif slicing_axis == "y":
    #     print("Rotating mesh to align Y with Z...")
    #     mesh = rotate_mesh(mesh, "x", 90)  # Rotate around X-axis
    #     non_planar_mesh = rotate_mesh(non_planar_mesh, "x", 90)

    # # Rotate 180° if needed
    # if rotate_180:
    #     print("Rotating model 180°...")
    #     mesh = rotate_mesh(mesh, "x", 180)
    #     non_planar_mesh = rotate_mesh(non_planar_mesh, "x", 180)

    # Compute a common center for both meshes
    common_center = (mesh.get_center() + non_planar_mesh.get_center()) / 2

    # Apply necessary rotations based on slicing direction
    if slicing_axis == "x":
        print("Rotating mesh to align X with Z...")
        mesh = rotate_mesh(mesh, "y", 90, center=common_center)
        non_planar_mesh = rotate_mesh(non_planar_mesh, "y", 90, center=common_center)
    elif slicing_axis == "y":
        print("Rotating mesh to align Y with Z...")
        mesh = rotate_mesh(mesh, "x", 90, center=common_center)
        non_planar_mesh = rotate_mesh(non_planar_mesh, "x", 90, center=common_center)

    # Rotate 180° if needed
    if rotate_180:
        print("Rotating model 180°...")
        mesh = rotate_mesh(mesh, "x", 180, center=common_center)
        non_planar_mesh = rotate_mesh(non_planar_mesh, "x", 180, center=common_center)

        
    # Compute vertex normals (important for better visualization)
    mesh.compute_vertex_normals()
    non_planar_mesh.compute_vertex_normals()

    
    o3d.visualization.draw_geometries(
        [mesh, non_planar_mesh], 
        window_name="Mesh + Non-Planar Surface + Damaged Mesh"
    )

    # Ask user for layer height
    layer_height = input("Enter the layer height for slicing (in mm): ").strip()


    try:
        layer_height = float(layer_height)
        if layer_height <= 0:
            print("Invalid input. Defaulting to 1 mm.")
            layer_height = 1
    except ValueError:
        print("Invalid input. Defaulting to 1 mm.")
        layer_height = 1

    # Ask user if they want to add extra layers
    add_layers = input("Enter the number of extra layers to add (or 0 if none): ").strip()
    try:
        add_layers = int(add_layers)
    except ValueError:
        add_layers = 0

    # Call slice function with user-defined parameters
    slices = slice_mesh(mesh, layer_height, add_layers)
    num_layers = len(slices)

    # # all_planes, non_planar_mesh = segment_mesh_planes(mesh, True)
    # slices_non_planar = slice_mesh(aligned_non_planar_mesh, layer_height, add_layers)
    # non_planar_dict = {h: bp for h, bp in slices_non_planar}

    # Ask for the toolpath strategy to apply
    print("\nChoose a toolpath strategy to apply to all layers:")
    print("1. Raster Path")
    print("2. Zig-Zag Path")
    print("3. Contour Path")
    print("4. Spiral Path")
    print("5. Hybrid 1 Path")
    print("6. Hybrid 2 Path")
    
    strategy = int(input("Enter the number corresponding to the strategy: "))
    tool_width = float(input("Enter the laser spot diameter (in mm): "))
    hatch_distance = float(input("Enter the hatch distance between raster lines (in mm): "))
    fly_in_out = input("Enable fly-in and fly-out paths? (True/False): ").strip().lower() == "true"
    fly_distance = float(input("Enter the fly-in and fly-out distance (in mm): "))
    raster_angle = float(input("Enter the cross section angle (in degrees, e.g., 0 for horizontal, 90 for vertical): "))
    corner_radius = float(input("Enter the corner radius (in mm): "))

    # aligned_non_planar_mesh, transformation = align_meshes_icp(mesh, non_planar_mesh)
    
    toolpaths = []
    contours = []

    # Iterate through each slice (which contains layer height and boundary points)
    for layer_index, (layer_height, boundary_points) in enumerate(slices):
        print(f"Processing Layer {layer_index+1} at height {layer_height:.3f}")

        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])


        # === If turbine blade repair ===
        
        try :

            densified_points = densify_boundary(boundary_points, step=0.2)
            polygon = Polygon(densified_points)
            fitted_circles, circles_with_boundary_points = fit_largest_circles(polygon, densified_points, num_circles=5, resolution=0.05, tolerance=0.05)
            centers, curvature_values, x_spline, y_spline = compute_curvature_influence(fitted_circles)
            filtered_centers, filtered_curvatures = filter_centers_by_curvature(centers, curvature_values, z_thresh=2)
            x_spline, y_spline = refit_spline_from_filtered_centers(filtered_centers)
            # plot_filtered_result(boundary_points, fitted_circles, centers, curvature_values, filtered_centers, filtered_curvatures, x_spline, y_spline)
            filtered_circles = [c for c in circles_with_boundary_points if any(np.isclose(c[0], center[0]) and np.isclose(c[1], center[1]) for center in filtered_centers)]
            left_spline, right_spline, left_points, right_points = fit_left_right_splines_from_circle_contacts(filtered_circles)
            # plot_splines_with_points(boundary_points, left_spline, right_spline, left_points, right_points)
            middle_spline = [x_spline, y_spline]
            # x_mid_ext, y_mid_ext = extend_spline(middle_spline)
            x_left_ext, y_left_ext = extend_spline(left_spline)
            left_ext = [x_left_ext, y_left_ext]
            x_right_ext, y_right_ext = extend_spline(right_spline)
            right_ext = [x_right_ext, y_right_ext]
            intersection_x, intersection_y = compute_spline_intersection(x_left_ext, y_left_ext, x_right_ext, y_right_ext)
            intersection = np.array([intersection_x, intersection_y])

            # plot_splines_with_points_2(boundary_points, left_spline, right_spline, left_points, right_points, 
            #                  x_spline, y_spline, x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection_x, intersection_y)


            best_circle = None
            max_y = float('-inf')  # Initialize with negative infinity

            for c in filtered_circles:
                if c[1] > max_y:  # Check if the current circle's y-coordinate is higher
                    max_y = c[1]
                    best_circle = c
            

            final_boundary = rebuild_boundary_with_spline_top(
                boundary_points,
                best_circle[3][0],
                best_circle[3][1],
                x_left_ext,
                y_left_ext,
                x_right_ext,
                y_right_ext,
                intersection
                )

            final_boundary = sort_points_counterclockwise(final_boundary)
            boundary_points = final_boundary



            # plot_updated_boundary(final_boundary, best_circle[3][0], best_circle[3][1], intersection)

            # plot_splines_with_points_2(final_boundary, left_spline, right_spline, left_points, right_points, 
            #                  x_spline, y_spline, x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection_x, intersection_y)
            

       


        except Exception as e:
            print(f"Error processing layer {layer_index + 1}: {e}")
            continue

        #  === If turbine blade repair ===

        contour = extract_surface_contour(non_planar_mesh, layer_height)

        # contour2 = extract_surface_contour_by_mesh(non_planar_mesh_for_contour, layer_height)
       
        # if not np.allclose(boundary_points[0], boundary_points[-1]):
        #     boundary_points = np.vstack([boundary_points, boundary_points[0]])


        # # Visualization
        # plt.figure(figsize=(8, 8))

        # # Plot boundary
        # if boundary_points.size > 0:
        #     plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-', label="Convex-hull Boundary")

        # # Plot extracted contour
        # if contour.size > 0:
        #     plt.scatter(contour[:, 0], contour[:, 1], color='r', s=10, label="Damage Contour")  # Points

        # fixed_min_x = 0
        # fixed_max_x = 30
        # fixed_min_y = 0
        # fixed_max_y = 30

        # # Get current axis
        # ax = plt.gca()
        # ax.set_aspect('equal')
        
        # ax.set_xlim(fixed_min_x, fixed_max_x)
        # ax.set_ylim(fixed_min_y, fixed_max_y)

        # ax.set_xlabel("X-axis (mm)", fontsize=18, labelpad=10)
        # ax.set_ylabel("Y-axis (mm)", fontsize=18, labelpad=10)
        # ax.set_title("Model A", fontsize=20)
        # ax.tick_params(axis='both', labelsize=14)
        # ax.grid(True)

        # # Legend in upper left corner with padding
        # ax.legend(loc="upper left", fontsize=16)

        # plt.tight_layout()
        # plt.show()

        if strategy == 1:
            deposition, tool_moves, fly_lines, fillets = generate_raster_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle
            )

        elif strategy == 2:
            deposition, tool_moves, fly_lines, fillets = generate_zigzag_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius
            )

        elif strategy == 3:
            deposition, tool_moves, fly_lines, fillets = generate_contour_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius
            )

        elif strategy == 4:
            deposition, tool_moves, fly_lines, fillets = generate_spiral_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius
            )

        elif strategy == 5:
            deposition, tool_moves, fly_lines, fillets = generate_hybrid1_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius
            )

        elif strategy == 6:
            deposition, tool_moves, fly_lines, fillets = generate_hybrid2_toolpath(
                boundary_points, contour, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius
            )
            
        else:
            print("Invalid choice, using raster toolpath by default.")
            deposition, tool_moves, fly_lines, fillets = generate_raster_toolpath(
                boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle
            )

        
        # Store toolpath for this layer
        toolpaths.append((deposition, tool_moves, fly_lines, fillets))
        contours.append(contour)

    evaluation_metrics(toolpaths, layer_height, hatch_distance, tool_width, slices, contours)

    # toolpaths = add_missing_layers_to_toolpaths(toolpaths, slices, layers_with_insufficient_points) 

    # full_3d_trajectory, toolpath = link_layers_and_visualize(toolpaths, layer_height)

    
    # export_trajectory_to_csv(full_3d_trajectory)

    # Ask if the user wants to see all 2D Strategy
    visualize_all = input("Do you want to visualize all 2D Strategy for each layer? (yes/no): ").strip().lower()

    if visualize_all == 'yes':
        # Visualize all layers' toolpaths
        for i, (deposition, tool_moves) in enumerate(toolpaths):
            print(f"Visualizing layer {i + 1}")
            visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets)
    else:
        # Ask for a specific layer to visualize
        while True:
            layer_idx = input("Enter the layer index (or 'q' to quit): ").strip()
            if layer_idx.lower() == 'q':
                break
            try:
                layer_idx = int(layer_idx)
                if 0 <= layer_idx < len(toolpaths):
                    deposition, tool_moves, fly_lines, fillets = toolpaths[layer_idx]
                    visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets)
                else:
                    print("Invalid index, please enter a valid layer index.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")

if __name__ == "__main__":
    main()
