import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from geomdl import BSpline, utilities, NURBS, fitting
from scipy.spatial import cKDTree, ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev, Rbf, CubicSpline
from scipy.signal import savgol_filter
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, Point
from shapely.affinity import rotate
from shapely.validation import explain_validity
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.fftpack import fft, ifft
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import csv
from sklearn.decomposition import PCA
from skimage.segmentation import morphological_chan_vese
from scipy.spatial.distance import cdist


def rotate_point_cloud(pcd, angle_degrees, axis="x"):
    """
    Rotates a point cloud around a specified axis.

    :param pcd: Open3D PointCloud object
    :param angle_degrees: Rotation angle in degrees
    :param axis: Rotation axis ('x', 'y', or 'z')
    :return: Rotated Open3D PointCloud object
    """
    # Convert degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Define rotation matrices for each axis
    if axis == "x":
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == "y":
        R = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == "z":
        R = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Apply rotation
    pcd.rotate(R, center=(0, 0, 0))  # Rotate around the origin
    return pcd


def voxelize_mesh(model_mesh, voxel_size):
    """
    Converts a 3D model mesh to a voxelized grid with a given voxel size.
    """

    # Convert mesh to point cloud
    pcd = model_mesh.sample_points_poisson_disk(number_of_points=80000)
    # print(pcd)
    # Voxelize point cloud (which represents the repair volume)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    
    return voxel_grid

def slice_voxelized_cloud(voxel_grid, layer_height):
    """
    Slices a voxelized point cloud into layers based on the specified layer height.
    Skips the last layer and adds extra layers if requested.
    """

    # Get the origin and bounding box size
    min_bound = voxel_grid.get_min_bound()
    max_bound = voxel_grid.get_max_bound()
    voxel_size = voxel_grid.voxel_size

    # Calculate Z range (height range)
    min_z = min_bound[2]
    max_z = max_bound[2]

    # Generate slice heights
    slice_heights = np.arange(min_z, max_z, layer_height)

    slices = []
    for height in slice_heights[:-1]:  # Skip the last layer
        # Get all voxel centers
        voxel_centers = np.asarray(voxel_grid.get_voxels())
        voxel_centers = np.array([voxel.grid_index for voxel in voxel_centers]) * voxel_size + min_bound

        # Filter voxels within the current layer height
        selected_voxels = voxel_centers[
            (voxel_centers[:, 2] >= height) & (voxel_centers[:, 2] < height + layer_height)
        ]

        if len(selected_voxels) > 0:
            slice = o3d.geometry.PointCloud()
            slice.points = o3d.utility.Vector3dVector(selected_voxels)
            slices.append(slice)

    return slices, slice_heights

def slice_point_cloud(pcd, layer_height):
    """
    Slices a point cloud into layers along a user-selected axis.
    If the chosen axis is X or Y, the function rotates the point cloud
    so that slicing always occurs along the Z-axis.

    Parameters:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        layer_height (float): The height of each layer.

    Returns:
        slices (list): A list of sliced point clouds.
        slice_heights (numpy array): Heights (positions) of the slices.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    slicing_axis = "z"  # Default slicing axis

    # Ask user for slicing direction
    new_axis = input("Enter a slicing axis (x, y, or z): ").strip().lower()
    if new_axis in axis_map:
        slicing_axis = new_axis
    else:
        print("Invalid input. Defaulting to Z-axis.")

    # Ask user if rotation by 180° is needed
    rotate_180 = input("Do you want to rotate the model 180°? (yes/no): ").strip().lower() == "yes"


    # Rotate the point cloud if needed
    if slicing_axis == "x":
        # Rotate around Y-axis to align X with Z
        R = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))  # 90° around Y-axis
        pcd.rotate(R, center=(0, 0, 0))
    elif slicing_axis == "y":
        # Rotate around X-axis to align Y with Z
        R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  # 90° around X-axis
        pcd.rotate(R, center=(0, 0, 0))

    # Rotate 180° if the user chooses
    if rotate_180:
        R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # 180° around X (flips model)
        pcd.rotate(R, center=(0, 0, 0))
        print("Model rotated 180°.")


    # Now we slice along the Z-axis
    points = np.asarray(pcd.points)
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    slice_heights = np.arange(min_z, max_z, layer_height)

    slices = []
    for pos in slice_heights[:-2]:  # Skip the last layer
        mask = (points[:, 2] >= pos) & (points[:, 2] < pos + layer_height)
        selected_points = points[mask]

        if len(selected_points) > 0:
            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(selected_points)
            slices.append(slice_pcd)

    # Visualize slices
    print(f"Displaying slices along {slicing_axis}-axis (aligned with Z-axis)...")
    o3d.visualization.draw_geometries(slices, window_name=f"Sliced along {slicing_axis.upper()}-axis")

    return slices, slice_heights

def visualize_slices(slices, slice_heights):
    """
    Visualizes each slice of the point cloud with Open3D.
    """
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='All Slices Visualization')

    # Define a color map for different slices
    colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))

    # for slice, height in zip(slices, slice_heights):
        # print(f"Visualizing slice at height: {height:.2f} mm")
        # o3d.visualization.draw_geometries([slice], window_name=f'Slice at {height:.2f} mm')


    for slice, height, color in zip(slices, slice_heights, colors):
        # Assign color to the slice
        slice.paint_uniform_color(color[:3])  # Use RGB values from the colormap

        # Add the slice to the visualizer
        vis.add_geometry(slice)

        # Optionally, add text labels (Open3D doesn't support text in 3D view directly)
        # You might need to use external tools or GUI frameworks for advanced annotations

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def sort_points_by_angle(points_2d):
    """make_pipeline
    Sorts the points in counter-clockwise order based on their polar angle with respect to the centroid.

    Args:
        points_2d (numpy.ndarray): 2D points of the boundary.

    Returns:
        numpy.ndarray: Sorted 2D points in counter-clockwise order.
    """
    # Compute the centroid of the points
    centroid = np.mean(points_2d, axis=0)

    # Compute the polar angle of each point relative to the centroid
    def polar_angle(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    # Sort the points by their polar angle in counter-clockwise direction
    sorted_points = sorted(points_2d, key=polar_angle)
    
    return np.array(sorted_points)

def fit_nurbs_to_boundary(boundary_points, degree):
    """
    Fits a NURBS curve to the given boundary points.
    Only fits if the number of points is sufficient (>= degree + 1).

    Args:
        boundary_points (numpy.ndarray): 2D points of the boundary (Nx2).
        degree (int): Degree of the NURBS curve.

    Returns:
        BSpline.Curve: Fitted NURBS curve or None if the number of points is insufficient.
    """
    # Sort the points in counter-clockwise order to ensure proper curve fitting
    boundary_points_sorted = sort_points_by_angle(boundary_points)

    # Ensure the boundary is closed
    if not np.allclose(boundary_points_sorted[0], boundary_points_sorted[-1]):
        boundary_points_sorted = np.vstack([boundary_points_sorted, boundary_points_sorted[0]])

    # Check if we have enough control points for the degree of the curve
    if len(boundary_points_sorted) < degree + 1:
        print("Not enough points for NURBS fitting (at least degree + 1 points required). Skipping this slice.")
        return None

    # Set up the NURBS curve
    curve = BSpline.Curve()
    curve.degree = degree

    # Set the control points (sorted boundary points)
    curve.ctrlpts = boundary_points_sorted.tolist()

    # Generate the knot vector. In this case, we use a "clamped" knot vector.
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

    # Compute the curve
    curve.evaluate()

    return curve

def determine_bounds_with_nurbs(slices, degree):
    """
    Determines the cross-section boundary curve using NURBS fitting for each slice.
    Returns the fitted NURBS curves for each slice.

    Args:
        slices (list): List of Open3D point clouds representing slices.
        degree (int): Degree of the NURBS curve (default is cubic).

    Returns:
        list: A list of NURBS curves for each slice (None if there weren't enough points).
    """
    boundaries = []
    layers_with_insufficient_points = []  # Initialize as a list to store indices

    for idx, slice_cloud in enumerate(slices):  # Use enumerate to track the layer index
        # Extract 2D points from the slice
        points = np.asarray(slice_cloud.points) 
        points_2d = points[:, :2]

        # Fit a NURBS curve to the points
        curve = fit_nurbs_to_boundary(points_2d, degree=degree)

        if curve is not None:
            boundaries.append(curve)
        else:
            print(f"Slice {idx} NURBS fitting failed due to insufficient points.")
            layers_with_insufficient_points.append(idx)  # Append the index of the failed slice

    
    return boundaries, layers_with_insufficient_points

def visualize_nurbs_curve(curve, num_points=100):
    """
    Visualizes the NURBS curve in 2D.

    Args:
        curve (BSpline.Curve): The NURBS curve to visualize.
        num_points (int): Number of points to evaluate the curve.
    """
    # Evaluate the curve at equally spaced values
    curve_points = np.array([curve.evaluate_single(i / (num_points - 1)) for i in range(num_points)])

    plt.figure(figsize=(6, 6))
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', label="NURBS Curve")
    plt.title('Fitted NURBS Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio is equal
    plt.legend()
    
    # Block until 'q' is pressed
    plt.show(block=True)
    print("Press 'q' to proceed to the next layer or continue.")

# def visualize_nurbs_in_3d(boundaries):
#     """
#     Visualizes all NURBS curves in 3D by plotting each one in its corresponding layer.
    
#     Args:
#         boundaries (list): List of NURBS curves for each slice.
#     """
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     for i, curve in enumerate(boundaries):
#         # Evaluate the NURBS curve at equally spaced values
#         curve_points = np.array([curve.evaluate_single(i / 100.0) for i in range(101)])

#         # Plot the NURBS curve in 3D, using the evaluated points
#         ax.plot(curve_points[:, 0], curve_points[:, 1], [slice_height for slice_height in [i]*len(curve_points)], label=f"Layer {i+1}")
    
#     ax.set_title("3D Visualization of All NURBS Curves")
#     ax.set_xlabel("X-axis (mm)")
#     ax.set_ylabel("Y-axis (mm)")
#     ax.set_zlabel("Layer (Z-axis)")
#     ax.legend()
#     plt.show()

def visualize_nurbs_in_3d(boundaries):
    """
    Visualizes all NURBS curves in 3D by plotting each one in its corresponding layer.
    
    Args:
        boundaries (list): List of NURBS curves for each slice.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, curve in enumerate(boundaries):
        if curve is None:
            print(f"Skipping visualization for layer {i} (No NURBS curve).")
            continue  # Skip None values

        # Evaluate the NURBS curve at equally spaced values
        curve_points = np.array([curve.evaluate_single(t) for t in np.linspace(0, 1, 101)])

        # Plot the NURBS curve in 3D
        ax.plot(curve_points[:, 0], curve_points[:, 1], [i] * len(curve_points), label=f"Layer {i+1}")

    ax.set_title("3D Visualization of All NURBS Curves")
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_zlabel("Layer (Z-axis)")
    ax.legend()
    plt.show()

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

    # Validate the polygon
    if not slice_polygon.is_valid:
        print(f"Invalid polygon detected: {explain_validity(slice_polygon)}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
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
                deposition_lines.append(np.array(clipped_line.coords))
            elif isinstance(clipped_line, MultiLineString):
                for subline in clipped_line.geoms:
                    deposition_lines.append(np.array(subline.coords))

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



    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

# --------- zig-zag with arc on transitions ------
# def generate_zigzag_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
#     """
#     Generates a zig-zag raster toolpath with optional fly-in and fly-out.
#     Includes tangent corner smoothing with adjustable fillet radius to link racks.
#     """

#     # Ensure boundary points are in 2D array format
#     boundary_points = np.asarray(boundary_points)
#     if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
#         raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

#     # Ensure the boundary is closed
#     if not np.allclose(boundary_points[0], boundary_points[-1]):
#         boundary_points = np.vstack([boundary_points, boundary_points[0]])

#     # Create Shapely Polygon
#     try:
#         slice_polygon = Polygon(boundary_points)
#     except ValueError as e:
#         print(f"Error creating polygon: {e}")
#         return np.array([]), np.array([])

#     # Validate the polygon
#     if not slice_polygon.is_valid:
#         print(f"Invalid polygon detected: {explain_validity(slice_polygon)}")
#         slice_polygon = slice_polygon.buffer(0)
#         if not slice_polygon.is_valid:
#             print("Could not fix the polygon. Skipping this slice.")
#             return np.array([]), np.array([])

#     # Determine the rotation angle based on the layer index
#     effective_angle = raster_angle if layer_index % 2 != 0 else 0

#     # Rotate the polygon by the effective_angle
#     rotated_polygon = rotate(slice_polygon, angle=effective_angle, origin='center', use_radians=False)

#     # Compute the bounding box and generate raster lines
#     min_x, min_y, max_x, max_y = rotated_polygon.bounds

#     deposition_lines = []
#     y = min_y
#     while y <= max_y:
#         line_start = (min_x, y)
#         line_end = (max_x, y)
#         line = LineString([line_start, line_end])
#         clipped_line = rotated_polygon.intersection(line)

#         if not clipped_line.is_empty:
#             if isinstance(clipped_line, LineString):
#                 deposition_lines.append(np.array(clipped_line.coords))
#             elif isinstance(clipped_line, MultiLineString):
#                 for subline in clipped_line.geoms:
#                     deposition_lines.append(np.array(subline.coords))

#         y += hatch_distance

#     # Limit the corner radius to half the hatch distance
#     max_radius = hatch_distance / 2
#     if corner_radius > max_radius:
#         corner_radius = max_radius

#     # Generate zig-zag path with proper tangent connections between racks
#     zig_zag_lines = []
#     for i, line in enumerate(deposition_lines):
#         if i % 2 == 1:  # Reverse direction for every other line
#             line = line[::-1]

#         # Add connection logic between racks
#         if i > 0:
#             prev_end = zig_zag_lines[-1][-1]
#             curr_start = line[0]

#             if corner_radius > 0:
#                 # Compute tangent fillet arcs
#                 connection_line, arc1, arc2 = create_tangent_fillet_connection(prev_end, curr_start, corner_radius)
#                 zig_zag_lines.append(arc1)  # First arc
#                 zig_zag_lines.append(connection_line)  # Straight connection
#                 zig_zag_lines.append(arc2)  # Second arc
#             else:
#                 # Direct line connection
#                 connection_line = np.array([prev_end, curr_start])
#                 zig_zag_lines.append(connection_line)

#         zig_zag_lines.append(line)

#     # Add fly-in and fly-out only for the starting and ending points
#     fly_lines = []
#     if fly_in_out and zig_zag_lines:
#         start_point = zig_zag_lines[0][0]
#         end_point = zig_zag_lines[-1][-1]

#         # Calculate direction vectors for fly-in and fly-out
#         start_dir = (zig_zag_lines[0][1] - start_point) / np.linalg.norm(zig_zag_lines[0][1] - start_point)
#         end_dir = (end_point - zig_zag_lines[-1][-2]) / np.linalg.norm(end_point - zig_zag_lines[-1][-2])

#         fly_in_point = start_point - fly_distance * start_dir
#         fly_out_point = end_point + fly_distance * end_dir

#         fly_lines.append(np.array([fly_in_point, start_point]))
#         fly_lines.append(np.array([end_point, fly_out_point]))

#     # Combine deposition lines into a single array
#     deposition_array = np.vstack(zig_zag_lines) if zig_zag_lines else np.array([])
#     fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])

#     # Rotate the toolpath back to the original orientation
#     if effective_angle != 0:
#         deposition_array = rotate_points(deposition_array, -effective_angle, slice_polygon.centroid)
#         fly_lines_array = rotate_points(fly_lines_array, -effective_angle, slice_polygon.centroid)

    
#     tool_moves_array = np.array([])  # No tool moves for zig-zag pattern


#     return deposition_array, tool_moves_array, fly_lines_array


# def create_tangent_fillet_connection(prev_end, curr_start, radius):
#     """
#     Creates a tangent connection between two points with two fillet arcs and a straight line in between.
#     """

#     # Vector from previous end to current start
#     direction = curr_start - prev_end
#     distance = np.linalg.norm(direction)

#     # Adjust radius if the distance is too small
#     if distance < 2 * radius:
#         radius = distance / 2

#     # Unit direction vector
#     direction_unit = direction / distance

#     # Compute the start and end points of the connecting line
#     connection_start = prev_end + radius * direction_unit
#     connection_end = curr_start - radius * direction_unit

#     # Tangent arc centers
#     arc1_center = prev_end + radius * np.array([-direction_unit[1], direction_unit[0]])
#     arc2_center = curr_start - radius * np.array([-direction_unit[1], direction_unit[0]])

#     # Generate arc points
#     arc1 = create_arc_points(prev_end, connection_start, arc1_center, radius)
#     arc2 = create_arc_points(connection_end, curr_start, arc2_center, radius)

#     # Straight connection line between the arcs
#     connection_line = np.array([connection_start, connection_end])

#     return connection_line, arc1, arc2


# def create_arc_points(start, end, center, radius):
    """
    Generates a tangent arc between two points with a given radius and center.
    """
    theta1 = np.arctan2(start[1] - center[1], start[0] - center[0])
    theta2 = np.arctan2(end[1] - center[1], end[0] - center[0])

    # Ensure proper angle direction
    if theta2 < theta1:
        theta2 += 2 * np.pi

    num_points = 10  # Number of points for the arc
    theta = np.linspace(theta1, theta2, num_points)

    arc_points = np.array([
        [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
        for t in theta
    ])

    return arc_points


def generate_zigzag_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a zig-zag raster toolpath with optional fly-in and fly-out.
    Includes tangent corner smoothing with adjustable fillet radius.
    """

    # Ensure boundary points are in 2D array format
    boundary_points = np.asarray(boundary_points)
    print(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Validate the polygon
    if not slice_polygon.is_valid:
        print(f"Invalid polygon: {explain_validity(slice_polygon)}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([])

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

    return deposition_array,tool_moves_array, fly_lines_array, fillets_array

def generate_hybrid1_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):

        """
        Generates a hybrid toolpath combining contour deposition and zig-zag filling.
        """
        boundary_points = np.asarray(boundary_points)
        if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
            raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

        # Ensure the boundary is closed
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])

        # Create Shapely Polygon
        slice_polygon = Polygon(boundary_points)

        # Validate the polygon
        if not slice_polygon.is_valid:
            print(f"Invalid polygon: {slice_polygon}")
            slice_polygon = slice_polygon.buffer(0)
            if not slice_polygon.is_valid:
                print("Could not fix the polygon. Skipping this slice.")
                return np.array([]), np.array([]), np.array([]), np.array([])

        ### Step 1: Generate Contour Deposition ###
        # Offset the boundary by the hatch distance for the contour path
        contour_polygon = slice_polygon.buffer(-hatch_distance)  # Offset inward
        if not contour_polygon.is_valid or contour_polygon.is_empty:
            print("Contour polygon is invalid or empty.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        contour_points = np.array(contour_polygon.exterior.coords)
        contour_deposition = [contour_points]

        ### Step 2: Generate Zig-Zag Interior Fill ###
        # Use the inner polygon from the contour deposition as the zig-zag boundary
        fill_boundary_polygon = contour_polygon.buffer(-tool_width / 2)  # Offset inward again for filling
        if not fill_boundary_polygon.is_valid or fill_boundary_polygon.is_empty:
            print("Fill boundary polygon is invalid or empty.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Rotate the polygon for zig-zag alignment
        rotated_polygon = rotate(fill_boundary_polygon, angle=raster_angle, origin='center', use_radians=False)
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        zigzag_lines = []
        y = min_y
        while y <= max_y:
            line_start = (min_x, y)
            line_end = (max_x, y)
            line = LineString([line_start, line_end])
            clipped_line = rotated_polygon.intersection(line)

            if not clipped_line.is_empty:
                if isinstance(clipped_line, LineString):
                    zigzag_lines.append(np.array(clipped_line.coords))
                elif isinstance(clipped_line, MultiLineString):
                    for subline in clipped_line.geoms:
                        zigzag_lines.append(np.array(subline.coords))
            y += hatch_distance

        # Reverse alternate zigzag lines for continuity
        for i, line in enumerate(zigzag_lines):
            if i % 2 == 1:
                zigzag_lines[i] = line[::-1]

        ### Step 3: Connect Contour to Zig-Zag (Tool Move) ###
        tool_moves = []
        if zigzag_lines:
            # The last point of the contour
            last_contour_point = contour_deposition[-1][-1]
            # The first point of the zig-zag path
            first_zigzag_point = zigzag_lines[0][0]

            # Add a tool move from the last contour point to the first zig-zag point
            tool_moves.append(np.array([last_contour_point, first_zigzag_point]))

        ### Combine Contour and Zig-Zag Deposition ###
        deposition_lines = contour_deposition + zigzag_lines
        deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])

        ### Optional: Fly-in and Fly-out ###
        fly_lines = []
        if fly_in_out and len(deposition_array) > 0:
            start_point = deposition_array[0]
            end_point = deposition_array[-1]

            # Fly-in direction
            fly_in_point = start_point - fly_distance * (deposition_array[1] - start_point) / np.linalg.norm(deposition_array[1] - start_point)
            fly_lines.append(np.array([fly_in_point, start_point]))

            # Fly-out direction
            fly_out_point = end_point + fly_distance * (end_point - deposition_array[-2]) / np.linalg.norm(end_point - deposition_array[-2])
            fly_lines.append(np.array([end_point, fly_out_point]))

        fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])

        ### Output Results ###
        tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
        fillets_array = np.array([])  # No fillets for simplicity in hybrid path

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array


def create_fillet_between_racks(prev_end, curr_start, radius):
    """
    Creates a tangent fillet between two points.
    Returns two arcs and a connecting straight line.
    """
    direction = curr_start - prev_end
    distance = np.linalg.norm(direction)

    # Adjust radius if the distance is too small
    if distance < 2 * radius:
        radius = distance / 2

    # Unit vector in the direction of the connection
    unit_dir = direction / distance

    # Define trimmed points for the connecting line
    arc1_end = prev_end + radius * unit_dir
    arc2_start = curr_start - radius * unit_dir

    # Define arc centers
    arc1_center = prev_end + radius * np.array([-unit_dir[1], unit_dir[0]])
    arc2_center = curr_start - radius * np.array([-unit_dir[1], unit_dir[0]])

    # Create arcs
    arc1 = create_arc_points(prev_end, arc1_end, arc1_center, radius)
    arc2 = create_arc_points(arc2_start, curr_start, arc2_center, radius)

    # Connecting straight line
    connecting_line = np.array([arc1_end, arc2_start])

    return arc1, connecting_line, arc2


def create_arc_points(start, end, center, radius):
    """
    Generates an arc between two points given the center and radius.
    """
    theta1 = np.arctan2(start[1] - center[1], start[0] - center[0])
    theta2 = np.arctan2(end[1] - center[1], end[0] - center[0])

    # Ensure proper angle direction
    if theta2 < theta1:
        theta2 += 2 * np.pi

    num_points = 10  # Number of points for the arc
    theta = np.linspace(theta1, theta2, num_points)

    arc_points = np.array([
        [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
        for t in theta
    ])

    return arc_points


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
    tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
    fillets_array = np.array([])

    return deposition_array, tool_moves_array, fly_lines_array, fillets_array


# def generate_hybrid2_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
    """
    Generates a hybrid toolpath with concentric offsets interlinked with zig-zag trajectories,
    focusing on specific boundary sections (e.g., corner damage).
    """
    # Ensure boundary points are valid
    boundary_points = np.asarray(boundary_points)
    if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

    # Ensure the boundary is closed
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    # Create Shapely Polygon
    slice_polygon = Polygon(boundary_points)

    # Validate the polygon
    if not slice_polygon.is_valid:
        slice_polygon = slice_polygon.buffer(0)  # Attempt to fix invalid polygon
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([]), np.array([])

    ### Step 1: Divide Boundary into Sections Using RANSAC ###
    boundary_sections = divide_boundary_with_ransac(boundary_points, 2)

    ### Step 2: Identify Region of Interest (e.g., Corner) ###
    # Assuming corner damage is identified as the smallest arc or curved segment
    corner_section = identify_corner_section(boundary_sections)

    ### Step 3: Offset the Corner Section ###
    corner_offsets = []
    offset_arc = Polygon(corner_section)
    while True:
        offset_arc = offset_arc.buffer(-hatch_distance)  # Inward offset
        if not offset_arc.is_valid or offset_arc.is_empty:
            break
        corner_offsets.append(np.array(offset_arc.exterior.coords))

    ### Step 4: Create Zig-Zag Trajectory ###
    zigzag_paths = []
    for i in range(len(corner_offsets) - 1):
        outer_offset = corner_offsets[i]
        inner_offset = corner_offsets[i + 1]

        # Ensure matching sizes between offsets
        inner_offset, outer_offset = interpolate_to_match(inner_offset, outer_offset)

        # Generate zig-zag lines connecting outer and inner offsets
        for j in range(len(inner_offset) - 1):
            start_point = inner_offset[j]
            end_point = outer_offset[j]
            zigzag_paths.append(np.array([start_point, end_point]))

    ### Step 5: Combine All Paths ###
    deposition_lines = corner_offsets + zigzag_paths
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])

    ### Step 6: Optional Fly-In and Fly-Out ###
    fly_lines = []
    if fly_in_out and len(deposition_array) > 0:
        # Fly-in direction
        start_point = deposition_array[0]
        fly_in_point = start_point - fly_distance * (deposition_array[1] - start_point) / np.linalg.norm(deposition_array[1] - start_point)
        fly_lines.append(np.array([fly_in_point, start_point]))

        # Fly-out direction
        end_point = deposition_array[-1]
        fly_out_point = end_point + fly_distance * (end_point - deposition_array[-2]) / np.linalg.norm(end_point - deposition_array[-2])
        fly_lines.append(np.array([end_point, fly_out_point]))

    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])

    ### Output Results ###
    tool_moves_array = np.array([])  # No separate tool moves
    fillets_array = np.array([])  # No fillets for simplicity

    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

# ------------ Hybrid2 Approach usuign ransac to detected straight lines ------------

# def divide_boundary_with_ransac(boundary_points, num_ransac_parts):
    """Divide the boundary into sections using RANSAC."""
    segments = []
    for _ in range(num_ransac_parts):
        model = RANSACRegressor()
        X = boundary_points[:, 0].reshape(-1, 1)
        y = boundary_points[:, 1]
        model.fit(X, y)

        inlier_mask = model.inlier_mask_
        outlier_mask = ~inlier_mask

        inlier_points = boundary_points[inlier_mask]
        segments.append(inlier_points)

        boundary_points = boundary_points[outlier_mask]  # Remove inliers for next iteration

    return segments

# def divide_boundary_with_ransac(boundary_points, num_ransac_parts, ransac_params=None):
    """
    Divide the boundary into sections using RANSAC and visualize the result for debugging.

    Parameters:
        boundary_points (ndarray): Array of boundary points (N x 2).
        num_ransac_parts (int): Number of RANSAC divisions to attempt.
        ransac_params (dict): Dictionary of RANSAC parameters, e.g., {'residual_threshold': 1.0, 'max_trials': 100}.
    """
    if ransac_params is None:
        ransac_params = {
            'residual_threshold':0.09,  # Default max distance for inliers
            'max_trials': 100,         # Default max iterations
            'stop_probability': 0.99   # Default stopping probability
        }

    segments = []
    remaining_points = boundary_points.copy()

    plt.figure(figsize=(8, 6))  # Create a figure for visualization
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', label="Original Boundary")  # Plot the original boundary

    colors = plt.cm.get_cmap('tab10', num_ransac_parts)  # Use a colormap for distinct colors

    for i in range(num_ransac_parts):
        if len(remaining_points) < 2:
            print(f"Not enough points remaining for RANSAC iteration {i + 1}.")
            break

        # Configure and fit RANSAC
        model = RANSACRegressor(
            residual_threshold=ransac_params['residual_threshold'],
            max_trials=ransac_params['max_trials'],
            stop_probability=ransac_params.get('stop_probability', 0.99)
        )
        X = remaining_points[:, 0].reshape(-1, 1)
        y = remaining_points[:, 1]

        try:
            model.fit(X, y)
        except Exception as e:
            print(f"RANSAC failed during iteration {i + 1}: {e}")
            break

        inlier_mask = model.inlier_mask_
        outlier_mask = ~inlier_mask

        # Extract inlier points as a section
        inlier_points = remaining_points[inlier_mask]
        segments.append(inlier_points)

        # Plot the inliers for this segment
        plt.plot(inlier_points[:, 0], inlier_points[:, 1], marker='o', color=colors(i), label=f"Segment {i + 1}")

        # Update remaining points for the next iteration
        remaining_points = remaining_points[outlier_mask]

    # Plot the remaining points, if any
    if len(remaining_points) > 0:
        plt.plot(remaining_points[:, 0], remaining_points[:, 1], 'r.', label="Remaining Points")

    plt.title("Boundary Division using RANSAC")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.show()  # Display the plot

    return segments

# def identify_corner_section(boundary_sections):
#     """Identify the section corresponding to the corner damage."""
#     # Example: Identify the shortest arc section
#     corner_section = min(boundary_sections, key=lambda section: LineString(section).length)
#     return corner_section

# def interpolate_to_match(inner, outer):
#     """Interpolate points to ensure inner and outer have the same number of points."""
#     inner = np.asarray(inner)
#     outer = np.asarray(outer)

#     inner_length = len(inner)
#     outer_length = len(outer)

#     if inner_length == outer_length:
#         return inner, outer

#     # Interpolate each dimension (x and y) independently
#     if inner_length > outer_length:
#         x_interp = np.interp(
#             np.linspace(0, outer_length - 1, inner_length),
#             np.arange(outer_length),
#             outer[:, 0]
#         )
#         y_interp = np.interp(
#             np.linspace(0, outer_length - 1, inner_length),
#             np.arange(outer_length),
#             outer[:, 1]
#         )
#         interpolated_outer = np.column_stack((x_interp, y_interp))
#         return inner, interpolated_outer

#     x_interp = np.interp(
#         np.linspace(0, inner_length - 1, outer_length),
#         np.arange(inner_length),
#         inner[:, 0]
#     )
#     y_interp = np.interp(
#         np.linspace(0, inner_length - 1, outer_length),
#         np.arange(inner_length),
#         inner[:, 1]
#     )
#     interpolated_inner = np.column_stack((x_interp, y_interp))
#     return interpolated_inner, outer


# ------------ Hybrid2 Approach usuign ransac to detected arcs ------------
# def generate_hybrid2_toolpath(
#     boundary_points, 
#     tool_width, 
#     hatch_distance, 
#     fly_in_out, 
#     fly_distance, 
#     layer_index, 
#     raster_angle, 
#     corner_radius=0, 
#     ransac_params=None, 
#     arc_detection=True
# ):
#     """
#     Generates a hybrid toolpath with concentric offsets interlinked with zig-zag trajectories,
#     using the detected arc as the main contour.
#     """
#     from shapely.geometry import Polygon, LineString
#     import numpy as np
    
#     # Helper functions
#     def interpolate_to_match(inner, outer):
#         """Interpolate points to ensure inner and outer have the same number of points."""
#         inner = np.asarray(inner)
#         outer = np.asarray(outer)

#         inner_length = len(inner)
#         outer_length = len(outer)

#         if inner_length == outer_length:
#             return inner, outer

#         # Interpolate each dimension (x and y) independently
#         if inner_length > outer_length:
#             x_interp = np.interp(
#                 np.linspace(0, outer_length - 1, inner_length),
#                 np.arange(outer_length),
#                 outer[:, 0]
#             )
#             y_interp = np.interp(
#                 np.linspace(0, outer_length - 1, inner_length),
#                 np.arange(outer_length),
#                 outer[:, 1]
#             )
#             interpolated_outer = np.column_stack((x_interp, y_interp))
#             return inner, interpolated_outer

#         x_interp = np.interp(
#             np.linspace(0, inner_length - 1, outer_length),
#             np.arange(inner_length),
#             inner[:, 0]
#         )
#         y_interp = np.interp(
#             np.linspace(0, inner_length - 1, outer_length),
#             np.arange(inner_length),
#             inner[:, 1]
#         )
#         interpolated_inner = np.column_stack((x_interp, y_interp))
#         return interpolated_inner, outer

#     # Step 1: Detect Arc Using RANSAC (if enabled)
#     if arc_detection:
#         arc_points, remaining_points = detect_arc_with_ransac(boundary_points, ransac_params, True)
#         if len(arc_points) == 0:
#             print("No arc detected. Using the original boundary.")
#             contour_to_follow = boundary_points
#         else:
#             print("Arc detected! Using it as the primary contour.")
#             contour_to_follow = arc_points
#     else:
#         contour_to_follow = boundary_points

#     # Step 2: Create Shapely Polygon
#     contour_to_follow = np.asarray(contour_to_follow)
#     if len(contour_to_follow.shape) != 2 or contour_to_follow.shape[1] != 2:
#         raise ValueError("Contour points must be a 2D array with shape (N, 2).")

#     # Ensure the contour is closed
#     if not np.allclose(contour_to_follow[0], contour_to_follow[-1]):
#         contour_to_follow = np.vstack([contour_to_follow, contour_to_follow[0]])

#     # Create Shapely Polygon
#     slice_polygon = Polygon(contour_to_follow)

#     # Validate the polygon
#     if not slice_polygon.is_valid:
#         print(f"Invalid polygon: {slice_polygon}")
#         slice_polygon = slice_polygon.buffer(0)
#         if not slice_polygon.is_valid:
#             print("Could not fix the polygon. Skipping this slice.")
#             return np.array([]), np.array([]), np.array([]), np.array([])

#     # Step 3: Generate Concentric Offsets
#     concentric_offsets = []
#     offset_polygon = slice_polygon
#     while True:
#         offset_polygon = offset_polygon.buffer(-hatch_distance)  # Inward offset
#         if not offset_polygon.is_valid or offset_polygon.is_empty:
#             break
#         concentric_offsets.append(np.array(offset_polygon.exterior.coords))

#     # Step 4: Create Interconnecting Zig-Zag Paths
#     zigzag_paths = []
#     for i in range(len(concentric_offsets) - 1):
#         outer_offset = concentric_offsets[i]
#         inner_offset = concentric_offsets[i + 1]

#         # Ensure matching sizes between offsets
#         inner_offset, outer_offset = interpolate_to_match(inner_offset, outer_offset)

#         # Generate zig-zag lines connecting outer and inner offsets
#         for j in range(len(inner_offset) - 1):
#             start_point = inner_offset[j]
#             end_point = outer_offset[j]
#             zigzag_paths.append(np.array([start_point, end_point]))

#     # Step 5: Combine All Paths
#     deposition_lines = concentric_offsets + zigzag_paths
#     deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])

#     # Step 6: Optional Fly-In and Fly-Out
#     fly_lines = []
#     if fly_in_out and len(deposition_array) > 0:
#         # Fly-in direction
#         start_point = deposition_array[0]
#         fly_in_point = start_point - fly_distance * (deposition_array[1] - start_point) / np.linalg.norm(deposition_array[1] - start_point)
#         fly_lines.append(np.array([fly_in_point, start_point]))

#         # Fly-out direction
#         end_point = deposition_array[-1]
#         fly_out_point = end_point + fly_distance * (end_point - deposition_array[-2]) / np.linalg.norm(end_point - deposition_array[-2])
#         fly_lines.append(np.array([end_point, fly_out_point]))

#     fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])

#     # Output Results
#     tool_moves_array = np.array([])  # No separate tool moves
#     fillets_array = np.array([])  # No fillets for simplicity

#     return deposition_array, tool_moves_array, fly_lines_array, fillets_array

# def detect_arc_with_ransac(boundary_points, ransac_params=None, visualize=False):
    """
    Detects an arc using RANSAC from a set of boundary points.
    Returns the points that fit the arc and the remaining points.
    """
    if ransac_params is None:
        ransac_params = {
            "residual_threshold": 5,
            "max_trials": 100,
            "stop_probability": 0.99,
            "min_samples": 20  # Minimum 3 points to define a circle
        }

    boundary_points = np.asarray(boundary_points)
    if boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")
    
    # Define a Circle Model
    class CircleModel(BaseEstimator):
        def __init__(self):
            self.center = None
            self.radius = None
        
        def fit(self, data, y=None):
            x = data[:, 0]
            y = data[:, 1]
            A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
            b = x**2 + y**2
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            self.center = solution[:2]
            self.radius = np.sqrt(solution[2] + np.sum(self.center**2))
            return self
        
        def predict(self, data):
            distances = np.sqrt((data[:, 0] - self.center[0])**2 + (data[:, 1] - self.center[1])**2)
            return distances
        
        def score(self, data, y):
            # Calculate residuals
            distances = self.predict(data)
            residuals = np.abs(distances - self.radius)
            return -np.mean(residuals)  # Negative mean residual as a score

    # Fit using RANSAC
    model = CircleModel()
    ransac = RANSACRegressor(
        estimator=model,
        residual_threshold=ransac_params["residual_threshold"],
        max_trials=ransac_params["max_trials"],
        stop_probability=ransac_params["stop_probability"],
        min_samples=ransac_params["min_samples"]
    )
    
    try:
        ransac.fit(boundary_points, np.zeros(len(boundary_points)))  # Targets are dummy
        inlier_mask = ransac.inlier_mask_
        outlier_mask = ~inlier_mask
        arc_points = boundary_points[inlier_mask]
        remaining_points = boundary_points[outlier_mask]
        
        # Debug Visualization
        if visualize:
            plt.figure(figsize=(8, 6))
            plt.scatter(boundary_points[:, 0], boundary_points[:, 1], label="All Points", alpha=0.5)
            plt.scatter(arc_points[:, 0], arc_points[:, 1], color="green", label="Detected Arc (Inliers)", s=50)
            plt.scatter(remaining_points[:, 0], remaining_points[:, 1], color="red", label="Remaining Points (Outliers)", s=50)
            
            # Plot the detected circle
            circle = plt.Circle(model.center, model.radius, color='blue', fill=False, label="Detected Arc")
            plt.gca().add_artist(circle)
            
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("RANSAC Detected Arc")
            plt.axis("equal")
            plt.grid(True)
            plt.show()
        
        return arc_points, remaining_points
    
    except Exception as e:
        print(f"RANSAC failed: {e}")
        return np.array([]), boundary_points

def slice_boundary_by_curvature(boundary_points, angle_threshold=0.2, visualize=False):
    """
    Slices the boundary into segments based on angular changes.
    Args:
        boundary_points: List of (x, y) coordinates.
        angle_threshold: Minimum angular change (in radians) to consider a curve.
        visualize: Whether to visualize the slicing process.
    Returns:
        A list of boundary segments and their corresponding angles.
    """
    boundary_points = np.asarray(boundary_points)
    if boundary_points.shape[1] != 2:
        raise ValueError("Boundary points must be a 2D array with shape (N, 2).")
    
    # Calculate vectors between consecutive points
    vectors = boundary_points[1:] - boundary_points[:-1]
    magnitudes = np.linalg.norm(vectors, axis=1)
    unit_vectors = vectors / magnitudes[:, np.newaxis]
    
    # Calculate angles between consecutive vectors
    dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip to avoid numerical errors
    
    # Identify curve points based on angle threshold
    curve_indices = np.where(angles > angle_threshold)[0] + 1  # +1 for alignment with boundary points
    
    # Create slices based on curve indices
    segments = []
    start_idx = 0
    for idx in curve_indices:
        segments.append(boundary_points[start_idx:idx + 1])  # Include the curve point
        start_idx = idx
    if start_idx < len(boundary_points) - 1:
        segments.append(boundary_points[start_idx:])  # Add the remaining points
    
    # Visualization (optional)
    if visualize:
        plt.figure(figsize=(8, 6))
        for i, segment in enumerate(segments):
            segment = np.asarray(segment)
            plt.plot(segment[:, 0], segment[:, 1], marker="o", label=f"Segment {i+1}")
        plt.title("Sliced Boundary Based on Angular Changes")
        plt.legend()
        plt.axis("equal")
        plt.show()
    
    return segments


def detect_most_irregular_curve(boundary_segments, freq_threshold=5, min_irregularity=0, max_irregularity=100, visualize=False):
    """
    Filters boundary segments to detect and return the most irregular curves within a specified range of irregularity.
    Args:
        boundary_segments: List of (x, y) segments.
        freq_threshold: Frequency threshold to measure irregularity.
        min_irregularity: Minimum irregularity level to consider a segment.
        max_irregularity: Maximum irregularity level to consider a segment.
        visualize: Whether to visualize the Fourier analysis.
    Returns:
        A list of filtered segments within the irregularity range.
    """
    filtered_segments = []

    for i, segment in enumerate(boundary_segments):
        if len(segment) < 1:  # Skip small segments
            continue

        # Extract X and Y components
        x, y = segment[:, 0], segment[:, 1]

        # Perform Fourier Transform
        fft_x = fft(x)
        fft_y = fft(y)

        # Measure irregularity (sum of high frequencies)
        irregularity = np.sum(np.abs(fft_x[freq_threshold:])) + np.sum(np.abs(fft_y[freq_threshold:]))

        # Check if irregularity is within the specified range
        if irregularity < max_irregularity:
            filtered_segments.append(segment)

        # # Visualization for debugging
        # if visualize:
        #     plt.figure(figsize=(6, 4))
        #     plt.plot(x, y, label=f"Segment {i+1}: Irregularity = {irregularity:.2f}", marker="o")
        #     plt.title(f"Segment {i+1}")
        #     plt.legend()
        #     plt.show()

    # Visualization of filtered segments only
    if visualize and filtered_segments:
        plt.figure(figsize=(8, 6))
        for i, segment in enumerate(filtered_segments):
            segment = np.asarray(segment)
            plt.plot(segment[:, 0], segment[:, 1], marker="o", label=f"Filtered Segment {i+1}")
        plt.title("Filtered Segments Based on Irregularity Levels")
        plt.legend()
        plt.axis("equal")
        plt.show()
    elif visualize and not filtered_segments:
        print("No segments met the irregularity criteria. Nothing to visualize.")

    return filtered_segments


def filter_points_ransac(arc_points, degree=2, residual_threshold=5.0, visualize=False):
    """
    Filters arc points using RANSAC to remove noise and fits a polynomial.
    """
    arc_points = np.array(arc_points)
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

# ------- Hybrid2 V1
# def generate_hybrid2_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
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

    # Validate the polygon
    if not slice_polygon.is_valid:
        print(f"Invalid polygon: {explain_validity(slice_polygon)}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([])

    # Step 2: Extract and Debug Inliers with RANSAC
    inlier_points, _ = filter_points_ransac(boundary_points, degree=2, residual_threshold=1.0, visualize=False)
    
    # # Debug: Plot inliers against boundary points
    # plt.figure(figsize=(8, 8))
    # plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color="black", label="Boundary")
    # plt.scatter(inlier_points[:, 0], inlier_points[:, 1], color="orange", label="RANSAC Inliers")

    # Step 3: Fit Polynomial and Generate Contour
    x = inlier_points[:, 0]
    y = inlier_points[:, 1]

    # Fit a 2nd-degree polynomial
    poly_coeffs = np.polyfit(x, y, deg=2)

    # Generate smooth polynomial points
    fitted_x = np.linspace(boundary_points[:, 0].min(), boundary_points[:, 0].max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    ransac_contour = np.column_stack((fitted_x, fitted_y))

    # # Debug: Plot RANSAC contour
    # plt.plot(ransac_contour[:, 0], ransac_contour[:, 1], 'b-', label="Fitted Polynomial Contour")

    # Step 4: Compute Deposition Lines
    deposition_lines = []
    offset = 0

    # Use the y-range of the boundary points for offset bounds
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()

    while offset <= y_range:
        print(f"Generating raster line at offset: {offset}")
        offset_points = []
        
        # Generate offset points based on the fitted polynomial
        for x_value in fitted_x:
            y_value = np.polyval(poly_coeffs, x_value) + offset
            offset_points.append((x_value, y_value))

        # Convert to LineString and clip to the polygon
        curve = LineString(offset_points)
        clipped_curve = slice_polygon.intersection(curve)

        # Add valid clipped curves to deposition lines
        if not clipped_curve.is_empty:
            print(f"Clipped curve type: {type(clipped_curve)}")
            if isinstance(clipped_curve, LineString):
                deposition_lines.append(np.array(clipped_curve.coords))
            elif isinstance(clipped_curve, MultiLineString):
                for sub_curve in clipped_curve.geoms:
                    deposition_lines.append(np.array(sub_curve.coords))

        offset += hatch_distance

    # Debug: Print the number of deposition lines generated
    print(f"Number of deposition lines generated: {len(deposition_lines)}")

    # # Step 5: Visualization
    # for line in deposition_lines:
    #     plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="red", alpha=0.7)

    # plt.legend()
    # plt.title("Hybrid Toolpath with Polynomial Raster Lines")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.grid()
    # plt.show()

    # Step 6: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    tool_moves_array = np.array([])  # Placeholder for tool moves
    fly_lines_array = np.array([])  # Placeholder for fly lines
    fillets_array = np.array([])  # Placeholder for fillets
    
    return deposition_array, tool_moves_array, fly_lines_array, fillets_array

# ------- Hybrid2 V2
# def generate_hybrid2_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
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

    # Validate the polygon
    if not slice_polygon.is_valid:
        print(f"Invalid polygon: {explain_validity(slice_polygon)}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([])

    # Step 2: Extract and Debug Inliers with RANSAC
    inlier_points, _ = filter_points_ransac(boundary_points, degree=2, residual_threshold=1.0, visualize=False)
    
    # Fit a 2nd-degree polynomial
    x = inlier_points[:, 0]
    y = inlier_points[:, 1]
    poly_coeffs = np.polyfit(x, y, deg=2)

    # Generate smooth polynomial points
    fitted_x = np.linspace(boundary_points[:, 0].min(), boundary_points[:, 0].max(), num=500)
    fitted_y = np.polyval(poly_coeffs, fitted_x)
    ransac_contour = np.column_stack((fitted_x, fitted_y))

    # Step 3: Compute Deposition Lines
    deposition_lines = []
    zig_zag_lines = []
    offset = 0
    y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()

    while offset <= y_range:
        print(f"Generating raster line at offset: {offset}")
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
                deposition_lines.append(np.array(clipped_curve.coords))
            elif isinstance(clipped_curve, MultiLineString):
                for sub_curve in clipped_curve.geoms:
                    deposition_lines.append(np.array(sub_curve.coords))

        offset += hatch_distance

    # Step 4: Add Straight Connecting Lines
    for i in range(len(deposition_lines) - 1):
        prev_end = deposition_lines[i][-1]  # Endpoint of the current deposition line
        curr_start = deposition_lines[i + 1][0]  # Start point of the next deposition line
        
        # Create a straight connection
        straight_connection = np.array([prev_end, curr_start])
        zig_zag_lines.append(straight_connection)

    # Debug: Plot the Toolpath
    plt.figure(figsize=(8, 8))
    # Plot deposition lines
    for line in deposition_lines:
        plt.plot(line[:, 0], line[:, 1], label="Deposition Line", color="blue", alpha=0.7)
    # Plot zig-zag straight connections
    for line in zig_zag_lines:
        plt.plot(line[:, 0], line[:, 1], 'g--', label="Connection Line" if 'Connection Line' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot boundary
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', label="Boundary")
    plt.legend()
    plt.title("Hybrid Toolpath with Zig-Zag Guiding Lines")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    zig_zag_array = np.vstack(zig_zag_lines) if zig_zag_lines else np.array([])
    fly_lines_array = np.array([])  # Placeholder for fly lines
    fillets_array = np.array([])  # Placeholder for fillets

    return deposition_array, zig_zag_array, fly_lines_array, fillets_array

def generate_hybrid2_toolpath(boundary_points, tool_width, hatch_distance, fly_in_out, fly_distance, layer_index, raster_angle, corner_radius=0):
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

    # Validate the polygon
    if not slice_polygon.is_valid:
        print(f"Invalid polygon: {explain_validity(slice_polygon)}")
        slice_polygon = slice_polygon.buffer(0)
        if not slice_polygon.is_valid:
            print("Could not fix the polygon. Skipping this slice.")
            return np.array([]), np.array([]), np.array([])

    # Step 2: Extract and Debug Inliers with RANSAC
    inlier_points, _ = filter_points_ransac(boundary_points, degree=2, residual_threshold=1, visualize=True)
    
    # Fit a 2nd-degree polynomial
    x = inlier_points[:, 0]
    y = inlier_points[:, 1]
    poly_coeffs = np.polyfit(x, y, deg=2)

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
    
    
    # # Debug: Plot the Toolpath
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

    # Step 5: Return Toolpath Data
    deposition_array = np.vstack(deposition_lines) if deposition_lines else np.array([])
    connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
    fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
    fillets_array = np.array([])  # Placeholder for fillets
    connecting_array = np.array([])
    return deposition_array, connecting_array, fly_lines_array, fillets_array



def visualize_toolpath_3d(toolpaths, slice_heights):
    import matplotlib
    matplotlib.use('TkAgg')  # Ensure GUI-compatible backend

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, toolpath in enumerate(toolpaths):
        layer_height = slice_heights[idx]
        ax.scatter(toolpath[:, 0], toolpath[:, 1], [layer_height] * len(toolpath), label=f"Layer {idx + 1}")

    ax.set_title("3D Toolpath Visualization")
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_zlabel("Z-axis (mm)")
    plt.legend()
    plt.show(block=True)


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
            '-o', color='blue', label='Deposition'
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
                '-o', color='blue', label='Fillet Moves' if i == 0 else ""
            )   
   

    
    # Add labels, legend, and styling2
    plt.axis('equal')
    plt.legend()
    plt.title('2D Toolpath Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Visualization of both damaged model and repair volume using PyVista
def visualize_models(damaged_model, repair_volume):
    # Create a plotter
    plotter = pv.Plotter()

    # Visualize the damaged model and repair volume
    plotter.add_mesh(damaged_model, color="red", opacity=0.7, label="Damaged Model")
    plotter.add_mesh(repair_volume, color="blue", opacity=0.5, label="Repair Volume")

    # Add legend and display
    plotter.add_legend()
    plotter.show()

def visualize_point_clouds(point_clouds):
    """
    Visualizes a list of PointCloud objects.

    Parameters:
        point_clouds (list): List of PointCloud objects to visualize.
    """
    # Ask if the user wants to see all 2D NURBS curves
    visualize_all_pcd = input("Do you want to visualize all PCD(yes/no): ").strip().lower()

    if visualize_all_pcd == 'yes':
        
        o3d.visualization.draw_geometries(point_clouds, window_name="Point Clouds Visualization")
        
    else:
        # Ask for a specific layer to visualize
        while True:
            layer_idx = input("Enter the layer index (or 'q' to quit): ").strip()
            if layer_idx.lower() == 'q':
                break
            try:
                layer_idx = int(layer_idx)
                if 0 <= layer_idx < len(point_clouds):
                    print(f"Visualizing NURBS curve for slice {layer_idx}")
                    o3d.visualization.draw_geometries([point_clouds[layer_idx]], window_name="Point Clouds Visualization")
                else:
                    print("Invalid index, please enter a valid layer index.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")


# ================ Smoothing Functions ===================

def process_boundary_slices(slices, smooth_method="spline", window_size=11):
    """
    Processes boundary slices by ordering and smoothing the boundary points.

    Parameters:
        slices (list): List of open3d.geometry.PointCloud objects.
        smooth_method (str): Smoothing method, either 'moving_average' or 'spline'.
        window_size (int): Window size for moving average smoothing.

    Returns:
        list: List of processed open3d.geometry.PointCloud objects.
    """
    processed_slices = []

    for idx, pcd in enumerate(slices):
        points = np.asarray(pcd.points)

        if points.shape[0] < 3:
            print(f"Skipping slice {idx}: Not enough points to compute a convex hull.")
            processed_slices.append(pcd)  # Retain the original PointCloud
            continue

        # Step 1: Order points along the boundary
        ordered_points = order_boundary_points(points)

        # Step 2: Apply smoothing
        if smooth_method == "moving_average":
            smoothed_points = moving_average_smoothing_closed(ordered_points, window_size=window_size)
        elif smooth_method == "spline":
            smoothed_points = spline_smoothing_closed(ordered_points)
        else:
            raise ValueError(f"Unknown smoothing method: {smooth_method}")

        # Create a new PointCloud object with processed points
        smoothed_pcd = o3d.geometry.PointCloud()
        smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
        processed_slices.append(smoothed_pcd)

    return processed_slices


def order_boundary_points(points):
    """
    Orders the points of a closed boundary in a continuous manner.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.

    Returns:
        numpy.ndarray: Ordered Nx2 or Nx3 array of points.
    """
    from scipy.spatial import ConvexHull

    if points.shape[1] > 2:  # If 3D points, project to 2D for ordering
        points_2d = points[:, :2]
    else:
        points_2d = points

    # Compute convex hull to ensure boundary ordering
    hull = ConvexHull(points_2d)
    ordered_indices = hull.vertices
    return points[ordered_indices]


def moving_average_smoothing_closed(points, window_size=5):
    """
    Smooths points for a closed boundary using a moving average.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        window_size (int): Window size for moving average.

    Returns:
        numpy.ndarray: Smoothed Nx2 or Nx3 array of points.
    """
    n_points = points.shape[0]
    if n_points < window_size:
        print(f"Warning: Number of points ({n_points}) is less than the window size ({window_size}).")
        return points

    # Treat as periodic
    extended_points = np.vstack([points[-window_size:], points, points[:window_size]])
    smoothed_points = np.zeros_like(points)
    for i in range(points.shape[1]):  # Smooth each dimension
        smoothed = np.convolve(
            extended_points[:, i],
            np.ones(window_size) / window_size,
            mode='same'
        )
        smoothed_points[:, i] = smoothed[window_size:-window_size]
    return smoothed_points


def spline_smoothing_closed(points, degree=1, num_points=100):
    """
    Smooths points for a closed boundary using a spline fit.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        degree (int): Degree of the spline.
        num_points (int): Number of points in the fitted spline.

    Returns:
        numpy.ndarray: Smoothed Nx2 or Nx3 array of points.
    """

    if points.shape[0] <= degree:  # Not enough points for the spline
        # print(f"Warning: Not enough points ({points.shape[0]}) for spline of degree {degree}. Using moving average.")
        # return moving_average_smoothing_closed(points, window_size=5)
        return points

    is_3d = points.shape[1] == 3
    if is_3d:
        tck, _ = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0, k=degree, per=True)
        smoothed = np.array(splev(np.linspace(0, 1, num_points), tck)).T
    else:
        tck, _ = splprep([points[:, 0], points[:, 1]], s=0, k=degree, per=True)
        smoothed = np.array(splev(np.linspace(0, 1, num_points), tck)).T
    return smoothed


# ================ Feature Detection | Adaptive Smoothing | Simplification ===================

def calculate_curvature(points):
    """
    Calculates curvature for a closed boundary.
    Curvature is derived from the angle between consecutive points.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.

    Returns:
        numpy.ndarray: Array of curvature values for each point.
    """
    num_points = points.shape[0]
    curvature = np.zeros(num_points)

    for i in range(num_points):
        prev_point = points[i - 1]
        current_point = points[i]
        next_point = points[(i + 1) % num_points]  # Wrap around for closed boundary

        # Vectors between consecutive points
        vec1 = current_point - prev_point
        vec2 = next_point - current_point

        # Normalize the vectors
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)

        # Calculate the angle (dot product)
        cosine_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        curvature[i] = np.arccos(cosine_angle)

    
    return curvature

def detect_key_features(points, curvature_threshold=0.3):
    """
    Detects key features based on curvature.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        curvature_threshold (float): Threshold for detecting high-curvature points.

    Returns:
        numpy.ndarray: Indices of key points.
    """
    curvatures = []

    for i in range(len(points)):
        # Define vectors
        prev_idx = (i - 1) % len(points)
        next_idx = (i + 1) % len(points)
        vec1 = points[i] - points[prev_idx]
        vec2 = points[next_idx] - points[i]

        # Normalize vectors
        vec1 /= (np.linalg.norm(vec1) + 1e-8)  # Add small value to prevent division by zero
        vec2 /= (np.linalg.norm(vec2) + 1e-8)

        # Compute curvature using angle between vectors
        dot_product = np.dot(vec1, vec2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for acos
        curvature = np.arccos(dot_product)

        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    # Detect indices of key points
    key_point_indices = np.where(curvatures > curvature_threshold)[0]

    if len(key_point_indices) == 0:
        print("Warning: No key points detected. Returning empty array.")
    return key_point_indices

def simplify_points(points, min_distance=1.0):
    """
    Simplifies the point cloud by clustering close points.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        min_distance (float): Minimum distance to retain between points.

    Returns:
        numpy.ndarray: Simplified array of points.
    """
    simplified = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - simplified[-1]) > min_distance:
            simplified.append(points[i])
    return np.array(simplified)

def adaptive_smoothing(points, key_point_indices, degree=3, num_points=100):
    """
    Performs adaptive smoothing based on key points.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        key_point_indices (numpy.ndarray): Indices of key points to preserve.
        degree (int): Degree of the spline.
        num_points (int): Number of points in the fitted spline.

    Returns:
        numpy.ndarray: Smoothed points array.
    """
    from scipy.interpolate import splprep, splev

    # Check if there are enough key points for the spline
    if len(key_point_indices) <= degree or len(points) <= degree:
        print(f"Warning: Not enough key points ({len(key_point_indices)}) for spline of degree {degree}. Returning original points.")
        return points

    # Extract key points
    key_points = points[key_point_indices]

    # Handle edge cases with NaN values in key points
    if np.isnan(key_points).any():
        print("Warning: Detected NaN values in key points. Returning original points.")
        return points

    # Fit spline through key points
    tck, _ = splprep([key_points[:, 0], key_points[:, 1]], s=0, k=degree, per=True)
    smoothed = np.array(splev(np.linspace(0, 1, num_points), tck)).T

    return smoothed

def process_adaptive_slices(slices, curvature_threshold=0.3, min_distance=1.0, degree=3, num_points=100):
    """
    Processes slices by detecting features, simplifying, and adaptively smoothing.

    Parameters:
        slices (list): List of Open3D PointClouds.
        curvature_threshold (float): Curvature threshold for feature detection.
        min_distance (float): Minimum distance for point simplification.
        degree (int): Degree of spline for smoothing.
        num_points (int): Number of points for the smoothed spline.

    Returns:
        list: List of processed Open3D PointClouds.
    """
    processed_slices = []

    for idx, pcd in enumerate(slices):
        points = np.asarray(pcd.points)

        # Calculate curvature
        curvature = calculate_curvature(points)

        # Segment boundary into features
        segments = segment_boundary(points, curvature, curvature_threshold=curvature_threshold)

        # Visualize intermediate results
        visualize_segmentation(points, segments)  # Add this line for visualization

        # Simplify points first
        simplified_points = simplify_points(points, min_distance=min_distance)

        # Detect key features on the simplified points
        key_point_indices = detect_key_features(simplified_points, curvature_threshold=curvature_threshold)

        # Perform adaptive smoothing using simplified points and detected key points
        smoothed_points = adaptive_smoothing(simplified_points, key_point_indices, degree=degree, num_points=num_points)

        # Ensure smoothed_points is Nx3 (3D) or Nx2 (2D), and handle 2D by padding with zeros
        if smoothed_points.shape[1] == 2:  # If 2D, pad with zeros to make it 3D
            smoothed_points = np.hstack([smoothed_points, np.zeros((smoothed_points.shape[0], 1))])

        # Convert back to Open3D PointCloud
        smoothed_pcd = o3d.geometry.PointCloud()
        smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
        processed_slices.append(smoothed_pcd)

    return processed_slices

def segment_boundary(points, curvature, curvature_threshold=0.1, line_threshold=0.01):
    """
    Segments a boundary into lines, curves, and corners.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        curvature (numpy.ndarray): Curvature values for each point.
        curvature_threshold (float): Threshold for distinguishing curves from lines.
        line_threshold (float): Minimum curvature for considering as part of a line.

    Returns:
        list of tuples: Each tuple contains:
            - segment_type: "line", "curve", or "corner"
            - segment_indices: Indices of the segment in the points array.
    """
    num_points = len(curvature)
    segments = []

    # Identify groups of points for lines and curves
    current_segment = []
    current_type = None

    for i in range(num_points):
        curv = curvature[i]

        # Determine if this point belongs to a line, curve, or corner
        if curv < line_threshold:
            segment_type = "line"
        elif curv > curvature_threshold:
            segment_type = "curve"
        else:
            segment_type = "corner"

        # If type changes or we reach the end of points, finalize the current segment
        if current_type is None:
            current_type = segment_type
            current_segment = [i]
        elif segment_type == current_type:
            current_segment.append(i)
        else:
            # Add the current segment to segments list
            segments.append((current_type, current_segment))
            current_type = segment_type
            current_segment = [i]

    # Add the last segment
    if current_segment:
        segments.append((current_type, current_segment))

    return segments

def visualize_segmentation(points, segments):
    """
    Visualizes the segmentation of a boundary into lines, curves, and corners.

    Parameters:
        points (numpy.ndarray): Nx2 or Nx3 array of points.
        segments (list of tuples): Each tuple contains:
            - segment_type: "line", "curve", or "corner"
            - segment_indices: Indices of the segment in the points array.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # Plot all points
    plt.scatter(points[:, 0], points[:, 1], s=5, c='gray', label='Boundary Points', zorder=1)

    # Plot segments with different colors
    for segment_type, segment_indices in segments:
        segment_points = points[segment_indices]
        if segment_type == "line":
            plt.plot(segment_points[:, 0], segment_points[:, 1], 'b-', label='Line Segment', zorder=2)
        elif segment_type == "curve":
            plt.plot(segment_points[:, 0], segment_points[:, 1], 'r-', label='Curve Segment', zorder=3)
        elif segment_type == "corner":
            plt.scatter(segment_points[:, 0], segment_points[:, 1], c='g', s=20, label='Corner', zorder=4)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Boundary Segmentation: Lines, Curves, and Corners")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# def link_layers_and_visualize(toolpaths, layer_height):
    """
    Links all layers to create a continuous 3D trajectory and visualizes it.

    Parameters:
        toolpaths (list): List of tuples containing toolpaths for each layer.
                          Each tuple has (deposition, tool_moves, fly_lines, fillets).
        layer_height (float): Height increment between each layer.
    """
    continuous_trajectory = []  # To store the continuous 3D trajectory
    current_z = 0  # Starting Z-coordinate
    
    for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(toolpaths):
        # Convert 2D paths to 3D by adding the current Z-coordinate
        if deposition.size > 0:
            deposition_3d = np.hstack((deposition, np.full((deposition.shape[0], 1), current_z)))
            continuous_trajectory.append(deposition_3d)

        if tool_moves.size > 0:
            for move in tool_moves:
                move_3d = np.hstack((move, np.full((move.shape[0], 1), current_z)))
                continuous_trajectory.append(move_3d)

        if fly_lines.size > 0:
            for fly in fly_lines:
                fly_3d = np.hstack((fly, np.full((fly.shape[0], 1), current_z)))
                continuous_trajectory.append(fly_3d)

        # Link this layer to the next by adding a vertical connection in Z
        if i < len(toolpaths) - 1:  # If not the last layer
            next_z = current_z + layer_height
            # Add a vertical linking line from end of this layer to start of the next
            end_point = deposition_3d[-1] if deposition.size > 0 else None
            next_start_point = np.hstack((toolpaths[i + 1][0][0], next_z)) if toolpaths[i + 1][0].size > 0 else None

            if end_point is not None and next_start_point is not None:
                vertical_link = np.array([end_point, next_start_point])
                continuous_trajectory.append(vertical_link)

        # Increment Z-coordinate for the next layer
        current_z += layer_height

    # Flatten the trajectory into a single array for visualization
    flattened_trajectory = np.vstack(continuous_trajectory)

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for segment in continuous_trajectory:
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color='b')

    ax.set_title("3D Toolpath Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    return flattened_trajectory  # Return the full 3D trajectory for further use

# def link_layers_and_visualize(toolpaths, layer_height):
    """
    Links all layers to create a continuous 3D trajectory and visualizes it.

    Parameters:
        toolpaths (list): List of tuples containing toolpaths for each layer.
                          Each tuple has (deposition, tool_moves, fly_lines, fillets).
        layer_height (float): Height increment between each layer.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    continuous_trajectory = []  # To store the continuous 3D trajectory
    current_z = 0  # Starting Z-coordinate

    print("Debug: Starting to process layers...")
    for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(toolpaths):
        # Convert 2D paths to 3D by adding the current Z-coordinate
        if len(deposition) > 0:
            deposition_3d = np.hstack((deposition[:, :3], np.full((deposition.shape[0], 1), current_z)))  # Ensure deposition is 3D
            continuous_trajectory.append(deposition_3d)
            print(f"Layer {i}, Deposition Points (3D):\n{deposition_3d}")
        else:
            print(f"Layer {i}: No deposition points.")

        # Process tool moves, fly lines, and other components similarly
        if len(tool_moves) > 0:
            for move in tool_moves:
                move_3d = np.hstack((move[:, :3], np.full((move.shape[0], 1), current_z)))  # Ensure move is 3D
                continuous_trajectory.append(move_3d)

        if len(fly_lines) > 0:
            for fly in fly_lines:
                fly_3d = np.hstack((fly[:, :3], np.full((fly.shape[0], 1), current_z)))  # Ensure fly is 3D
                continuous_trajectory.append(fly_3d)

        # Link this layer to the next by adding a vertical connection in Z
        if i < len(toolpaths) - 1:  # If not the last layer
            next_z = current_z + layer_height

            end_point = deposition_3d[-1][:3] if len(deposition) > 0 else None  # Ensure end_point is 3D
            next_start_point = (
                np.hstack((toolpaths[i + 1][0][0][:3], next_z)) if len(toolpaths[i + 1][0]) > 0 else None
            )  # Ensure next_start_point is 3D

            if end_point is not None and next_start_point is not None:
                # Debug: Check and log shapes
                print(f"Linking Layer {i} to Layer {i + 1}")
                print(f"  End Point Shape: {end_point.shape}, End Point: {end_point}")
                print(f"  Next Start Point Shape: {next_start_point.shape}, Next Start Point: {next_start_point}")

                # Ensure both points are 3D
                if len(end_point) == 3 and len(next_start_point) == 3:
                    vertical_link = np.array([end_point, next_start_point])
                    continuous_trajectory.append(vertical_link)
                else:
                    print(f"Shape mismatch: end_point={end_point}, next_start_point={next_start_point}")
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

    return flattened_trajectory  # Return the full 3D trajectory for further use

def link_layers_and_visualize(toolpaths, layer_height):
    """
    Links all layers to create a continuous 3D trajectory and visualizes it.

    Parameters:
        toolpaths (list): List of tuples containing toolpaths for each layer.
                          Each tuple has (deposition, tool_moves, fly_lines, fillets).
        layer_height (float): Height increment between each layer.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    continuous_trajectory = []  # To store the continuous 3D trajectory
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
            print(f"Layer {i}, Deposition Points (3D):\n{deposition_3d}")
        else:
            print(f"Layer {i}: No deposition points.")

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

        # Link this layer to the next by adding a vertical connection in Z
        if i < len(toolpaths) - 1:  # If not the last layer
            next_z = current_z + layer_height

            end_point = deposition_3d[-1][:3] if len(deposition) > 0 else None  # Ensure end_point is 3D
            next_start_point = (
                np.hstack((toolpaths[i + 1][0][0][:3], next_z)) if len(toolpaths[i + 1][0]) > 0 else None
            )  # Ensure next_start_point is 3D

            if end_point is not None and next_start_point is not None:
                # Debug: Check and log shapes
                print(f"Linking Layer {i} to Layer {i + 1}")
                print(f"  End Point Shape: {end_point.shape}, End Point: {end_point}")
                print(f"  Next Start Point Shape: {next_start_point.shape}, Next Start Point: {next_start_point}")

                # Ensure both points are 3D
                if len(end_point) == 3 and len(next_start_point) == 3:
                    vertical_link = np.array([end_point, next_start_point])
                    continuous_trajectory.append(vertical_link)
                else:
                    print(f"Shape mismatch: end_point={end_point}, next_start_point={next_start_point}")
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

    return flattened_trajectory  # Return the full 3D trajectory for further use

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

def export_trajectory_to_csv(trajectory, filename="3d_trajectory.csv", simplify=True, epsilon=10):
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

# def add_missing_layers_to_toolpaths(toolpaths, slices, layers_with_insufficient_points):
    # """
    # Adds a single point (centroid) for layers with insufficient points to the deposition toolpath.

    # Parameters:
    #     toolpaths (list): Original toolpaths for valid layers.
    #     slices (list): Point cloud data for all slices.
    #     layers_with_insufficient_points (list): Indices of layers without valid boundaries.

    # Returns:
    #     list: Updated toolpaths with added centroids for missing layers.
    # """
    # # Find the first valid layer
    # first_valid_layer_index = min(set(range(len(slices))) - set(layers_with_insufficient_points))
    # first_valid_boundary = np.asarray(slices[first_valid_layer_index].points)

    # if first_valid_boundary.size == 0:
    #     raise ValueError(f"First valid boundary at layer {first_valid_layer_index} is empty.")

    # # Compute the centroid of the first valid layer
    # first_valid_centroid = np.mean(first_valid_boundary, axis=0)

    # for layer_idx in layers_with_insufficient_points:
    #     current_slice = np.asarray(slices[layer_idx].points)

    #     if current_slice.size == 0:
    #         raise ValueError(f"Slice {layer_idx} is empty and cannot compute a centroid.")

    #     # Compute the centroid of the current slice
    #     current_centroid = np.mean(current_slice, axis=0)

    #     # Add the centroid as a single deposition point for this layer
    #     toolpaths.insert(
    #         layer_idx,
    #         (
    #             np.array([current_centroid]),  # Deposition path with a single point
    #             [],  # No tool moves for missing layers
    #             [],  # No fly lines for missing layers
    #             []   # No fillets for missing layers
    #         )
    #     )

    # return toolpaths

def add_missing_layers_to_toolpaths(toolpaths, slices, layers_with_insufficient_points):
    """
    Adds a single point (centroid) of the first valid boundary layer as the deposition toolpath
    for layers with insufficient points.

    Parameters:
        toolpaths (list): Original toolpaths for valid layers.
        slices (list): Point cloud data for all slices.
        layers_with_insufficient_points (list): Indices of layers without valid boundaries.

    Returns:
        list: Updated toolpaths with added centroids for missing layers.
    """
    # Find the first valid layer
    first_valid_layer_index = min(set(range(len(slices))) - set(layers_with_insufficient_points))
    first_valid_boundary = np.asarray(slices[first_valid_layer_index].points)

    if first_valid_boundary.size == 0:
        raise ValueError(f"First valid boundary at layer {first_valid_layer_index} is empty.")

    # Compute the centroid of the first valid layer
    first_valid_centroid = np.mean(first_valid_boundary, axis=0)

    for layer_idx in layers_with_insufficient_points:
        # Add the centroid of the first valid layer as the deposition for this layer
        toolpaths.insert(
            layer_idx,
            (
                np.array([first_valid_centroid]),  # Deposition path with a single point
                [],  # No tool moves for missing layers
                [],  # No fly lines for missing layers
                []   # No fillets for missing layers
            )
        )

    return toolpaths

def extract_missing_region(nominal_path, damaged_path, voxel_size=1.0, threshold=3, k=1):
    # Load and sample point clouds from STL
    nominal_mesh = o3d.io.read_triangle_mesh(nominal_path).compute_vertex_normals()
    damaged_mesh = o3d.io.read_triangle_mesh(damaged_path).compute_vertex_normals()
    
    nominal_pcd = nominal_mesh.sample_points_poisson_disk(200000)
    damaged_pcd = damaged_mesh.sample_points_poisson_disk(200000)
    
    # Downsample point clouds
    voxel_grid_nominal = o3d.geometry.VoxelGrid.create_from_point_cloud(nominal_pcd, voxel_size)
    voxel_grid_damaged = o3d.geometry.VoxelGrid.create_from_point_cloud(damaged_pcd, voxel_size)
    
    downsampled_nominal = np.stack([vx.grid_index for vx in voxel_grid_nominal.get_voxels()])
    downsampled_damaged = np.stack([vx.grid_index for vx in voxel_grid_damaged.get_voxels()])
    
    nominal_pcd.points = o3d.utility.Vector3dVector(downsampled_nominal)
    damaged_pcd.points = o3d.utility.Vector3dVector(downsampled_damaged)
    
    # Apply ICP for alignment
    icp_result = o3d.pipelines.registration.registration_icp(
        nominal_pcd, damaged_pcd, 0.5, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    nominal_pcd.transform(icp_result.transformation)
    
    # Convert to numpy arrays
    nominal_points = np.asarray(nominal_pcd.points)
    damaged_points = np.asarray(damaged_pcd.points)
    
    # Build KD-Trees
    nominal_tree = cKDTree(nominal_points)
    damaged_tree = cKDTree(damaged_points)
    
    # Find missing points in damaged model (i.e., parts of nominal missing from damaged)
    missing_in_damaged = [i for i, point in enumerate(nominal_points)
                           if np.linalg.norm(point - damaged_points[damaged_tree.query(point, k)[1]]) > threshold]
    
    # Find missing points in nominal model (i.e., excess material in damaged model)
    missing_in_nominal = [i for i, point in enumerate(damaged_points)
                           if np.linalg.norm(point - nominal_points[nominal_tree.query(point, k)[1]]) > threshold]
    
    # Combine both sets of missing points
    missing_volume_points = np.vstack((nominal_points[missing_in_damaged], damaged_points[missing_in_nominal]))
    # Combine both sets of missing points
    # missing_volume_points = np.vstack((nominal_points[missing_in_damaged]))
    # missing_volume_points = np.vstack((damaged_points[missing_in_nominal]))

    # Create missing region point cloud
    missing_pcd = o3d.geometry.PointCloud()
    missing_pcd.points = o3d.utility.Vector3dVector(missing_volume_points)
    missing_pcd.paint_uniform_color([1, 0, 0])

    # Create missing region point cloud
    damaged_pcd = o3d.geometry.PointCloud()
    damaged_pcd.points = o3d.utility.Vector3dVector(damaged_points)
    damaged_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="ICP Alignment")

    # Cluster missing points
    labels = np.array(missing_pcd.cluster_dbscan(eps=35, min_points=100, print_progress=False))
    print(labels)
    if len(labels) > 0:
        largest_cluster = max(set(labels), key=list(labels).count)
        missing_pcd = missing_pcd.select_by_index(np.where(labels == largest_cluster)[0])
    
    # # Visualize
    # o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd, missing_pcd], window_name="Missing Region")
    return missing_pcd, damaged_pcd

def refine_nurbs_boundaries(boundaries, degree=2, smoothing=1):
    """
    Refines bad NURBS boundary fits using PCA alignment, Moving Least Squares (MLS), 
    and B-spline interpolation.

    Args:
        boundaries (list): List of NURBS boundary curves.
        degree (int): Degree of the B-spline.
        smoothing (float): Smoothing factor for MLS.

    Returns:
        list: A list of refined boundary curves.
    """
    refined_boundaries = []

    for i, boundary in enumerate(boundaries):
        if boundary is None:
            print(f"Skipping layer {i}: No curve available.")
            refined_boundaries.append(None)
            continue

        # Ensure the boundary is a valid curve object
        if not hasattr(boundary, "evaluate_single"):
            print(f"Skipping layer {i}: Invalid curve object.")
            refined_boundaries.append(None)
            continue

        # Convert curve to points for processing
        sample_points = np.array([boundary.evaluate_single(t) for t in np.linspace(0, 1, 100)])

        # Apply PCA alignment
        pca = PCA(n_components=2)
        aligned_points = pca.fit_transform(sample_points[:, :2])

        # Apply Moving Least Squares (MLS) smoothing
        smoothed_points = moving_least_squares_2d(aligned_points, poly_degree=2, smoothing_factor=smoothing)

        # Convert back to a NURBS curve
        refined_curve = fit_nurbs_to_boundary(smoothed_points, degree=degree)

        refined_boundaries.append(refined_curve)

    return refined_boundaries

def moving_least_squares_2d(points, poly_degree=2, smoothing_factor=0.01):
    """
    Applies Moving Least Squares (MLS) smoothing to 2D points.
    
    Args:
        points (np.array): (N, 2) array of 2D points.
        poly_degree (int): Polynomial degree for MLS.
        smoothing_factor (float): Weighting function for smoothing.
    
    Returns:
        np.array: Smoothed 2D points.
    """
    from scipy.spatial import KDTree

    tree = KDTree(points)
    smoothed_points = np.zeros_like(points)

    for i, p in enumerate(points):
        # Find neighboring points
        dists, idxs = tree.query(p, k=min(10, len(points)))  
        weights = np.exp(-dists ** 2 / (2 * smoothing_factor ** 2))

        # Weighted least squares fit (polynomial)
        X = np.vander(points[idxs, 0], N=poly_degree + 1, increasing=True)
        W = np.diag(weights)
        coeffs = np.linalg.lstsq(W @ X, W @ points[idxs, 1], rcond=None)[0]

        smoothed_points[i] = [p[0], np.polyval(coeffs[::-1], p[0])]

    return smoothed_points

def fit_bspline(points, degree=3):
    """
    Fits a B-spline curve to a set of 2D points.
    
    Args:
        points (np.array): (N, 2) array of 2D points.
        degree (int): Degree of the B-spline curve.

    Returns:
        np.array: Interpolated B-spline points.
    """
    tck, _ = splprep(points.T, k=degree, s=0.01)
    u_fine = np.linspace(0, 1, 100)  # High resolution for smooth curves
    spline_points = np.array(splev(u_fine, tck)).T
    return spline_points

def sample_nurbs_curve(nurbs_curve, num_points=100):
    """
    Sample a NURBS curve at evenly spaced parameter values.
    
    Parameters:
    - nurbs_curve: A NURBS curve object.
    - num_points: Number of points to sample along the curve.

    Returns:
    - np.array of shape (num_points, 2) containing sampled (x, y) coordinates.
    """
    param_values = np.linspace(nurbs_curve.knotvector[0], nurbs_curve.knotvector[-1], num_points)
    sampled_points = np.array([nurbs_curve.evaluate_single(t) for t in param_values])
    return sampled_points

def compute_convex_hull_ratio(boundary, num_samples=100):
    """
    Compute the convex hull ratio for a given NURBS boundary.

    Parameters:
    - boundary: NURBS curve object.
    - num_samples: Number of points to sample from the curve.

    Returns:
    - convex_hull_ratio: float (lower values indicate higher irregularity).
    """
    sampled_points = sample_nurbs_curve(boundary, num_samples)

    try:
        hull = ConvexHull(sampled_points)
        hull_area = hull.volume  # 2D convex hull area
        boundary_area = np.sum(np.abs(np.diff(sampled_points[:, 0]) * sampled_points[:-1, 1]))  # Approximate area

        convex_hull_ratio = boundary_area / hull_area if hull_area > 0 else 0
    except:
        convex_hull_ratio = 0  # If convex hull fails, assume it's highly irregular

    return convex_hull_ratio

def detect_bad_layers_by_convex_hull(boundaries_with_nurbs, ratio_threshold=None, debug=True):
    """
    Detect layers with low convex hull ratios (highly irregular boundaries).

    Parameters:
    - boundaries_with_nurbs: List of NURBS curve objects.
    - ratio_threshold: If None, uses the average ratio - 0.4 standard deviation.
    - debug: If True, visualizes the convex hull ratio distribution.

    Returns:
    - bad_layers: List of indices of problematic layers.
    """
    convex_hull_ratios = [compute_convex_hull_ratio(boundary) for boundary in boundaries_with_nurbs]
    print(convex_hull_ratios)

    # Compute automatic threshold if none provided
    if ratio_threshold is None:
        mean_ratio = np.mean(convex_hull_ratios)
        std_ratio = np.std(convex_hull_ratios)
        ratio_threshold = mean_ratio - (0.05 * std_ratio)  # Flag layers below this threshold

    bad_layers = [i for i, ratio in enumerate(convex_hull_ratios) if ratio > ratio_threshold]
    good_layers = [i for i, ratio in enumerate(convex_hull_ratios) if ratio < ratio_threshold]

    if debug:
        plt.figure(figsize=(8, 4))
        plt.plot(convex_hull_ratios, 'bo-', label="Convex Hull Ratio")
        plt.axhline(ratio_threshold, color='r', linestyle="--", label="Threshold")
        plt.xlabel("Layer Index")
        plt.ylabel("Convex Hull Ratio")
        plt.title("Boundary Irregularity (Convex Hull Ratio)")
        plt.legend()
        plt.grid()
        plt.show()
    
    return bad_layers, good_layers

def compute_scaling_factor_between_good_layers(good_layers, boundaries_with_nurbs):
    """
    Compute the average scaling factor between consecutive good layers based on convex hull ratios.
    
    Parameters:
    - good_layers: List of indices for the good layers.
    - boundaries_with_nurbs: List of NURBS curves (boundaries).
    
    Returns:
    - avg_scaling_factor: The average scaling factor between the good layers.
    """
    scaling_factors = []
    
    # Iterate through the good layers and compute scaling between consecutive good layers
    for i in range(1, len(good_layers)):
        layer_prev = boundaries_with_nurbs[good_layers[i - 1]]
        layer_next = boundaries_with_nurbs[good_layers[i]]

        # Compute convex hull ratios for the previous and next good layers
        ratio_prev = compute_convex_hull_ratio(layer_prev)
        ratio_next = compute_convex_hull_ratio(layer_next)
        
        # Compute scaling factor between these two layers
        if ratio_next != 0:
            scaling_factor = ratio_prev / ratio_next
            scaling_factors.append(scaling_factor)
    
    # Compute the average scaling factor between all consecutive good layers
    avg_scaling_factor = np.mean(scaling_factors) if scaling_factors else 1  # Avoid division by zero or empty list
    
    return avg_scaling_factor

def create_nurbs_from_points(points, degree=3):
    """
    Create a NURBS curve from a list of points.

    Parameters:
    - points: A list of points (2D or 3D) to create the NURBS curve from.
    - degree: Degree of the NURBS curve (default is 3, cubic).

    Returns:
    - nurbs_curve: A NURBS curve object.
    """
    # Number of points
    num_points = len(points)

    # Define control points (same as sampled points)
    # control_points = [(x, y, 0) for x, y in points]  # Convert 2D points to 3D (adding 0 for z)
    control_points = [(x, y, 0) for x, y, *_ in points]

    # Create a B-Spline curve
    nurbs_curve = BSpline.Curve()

    # Set the degree of the NURBS curve
    nurbs_curve.degree = degree

    # Set the control points
    nurbs_curve.ctrlpts = control_points

    # Generate the knot vector using the 'uniform' type, which is common for simple use cases
    nurbs_curve.knotvector = utilities.generate_knot_vector(degree, num_points)

    # Set weights for the control points (if you want a rational NURBS curve, set weights to 1 by default)
    nurbs_curve.weights = [1.0] * num_points

    # Optionally, you can refine the curve and evaluate it if you want
    nurbs_curve.evaluate()

    return nurbs_curve

def apply_scaling_to_layer(layer, scaling_factor):
    """
    Apply scaling to a NURBS layer based on the scaling factor.
    
    Parameters:
    - layer: NURBS curve to scale.
    - scaling_factor: The factor by which to scale the layer.
    
    Returns:
    - scaled_layer: The scaled NURBS curve.
    """
    # Get the points from the NURBS curve
    points = sample_nurbs_curve(layer, num_points=100)
    
    # Scale the points
    scaled_points = points * scaling_factor
    
    # Create a new NURBS curve from the scaled points
    scaled_layer = create_nurbs_from_points(scaled_points)
    
    return scaled_layer

def visualize_layers(corrected_boundaries, boundaries_with_nurbs, bad_layers):
    """
    Visualize bad layers, their closest good reference layers, and their corrected versions.

    Parameters:
    - corrected_boundaries: The list of updated (corrected) NURBS boundaries.
    - boundaries_with_nurbs: The original list of NURBS boundaries.
    - bad_layers: Indices of the bad layers being visualized.
    """
    plt.figure(figsize=(10, 10))

    for bad_layer in bad_layers:
        # Find the closest good layer above
        prev_good_layer = min([g for g in range(len(boundaries_with_nurbs)) if g > bad_layer], default=None)

        if prev_good_layer is None:
            print(f"Skipping visualization for layer {bad_layer}, no good layer found above.")
            continue

        # Sample points from NURBS curves
        bad_points = sample_nurbs_curve(boundaries_with_nurbs[bad_layer], num_points=100)
        good_points = sample_nurbs_curve(boundaries_with_nurbs[prev_good_layer], num_points=100)
        corrected_points = sample_nurbs_curve(corrected_boundaries[bad_layer], num_points=100)

        # Plot each layer with a distinct color
        plt.plot(bad_points[:, 0], bad_points[:, 1], label=f"Bad Layer {bad_layer}", color='red', linestyle='--')
        plt.plot(good_points[:, 0], good_points[:, 1], label=f"Good Layer {prev_good_layer}", color='green', linestyle='--')
        plt.plot(corrected_points[:, 0], corrected_points[:, 1], label=f"Corrected Layer {bad_layer}", color='blue', linestyle='-')

    # Labels, grid, and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Bad vs. Good vs. Corrected Layers")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")  # Keep the aspect ratio correct for better comparison
    plt.show()

def simplify_nurbs(nurbs_curve, num_control_points=10, degree=3):
    """
    Simplifies a NURBS curve by reducing the number of control points.

    Parameters:
    - nurbs_curve: The original NURBS curve.
    - num_control_points: The desired number of control points in the simplified curve.
    - degree: The degree of the new simplified NURBS curve.

    Returns:
    - simplified_nurbs: A new, simplified NURBS curve.
    """

    # Sample evenly spaced points from the original curve
    sampled_points = sample_nurbs_curve(nurbs_curve, num_points=num_control_points)

    # Convert sampled points into new control points
    control_points = [(x, y, 0) for x, y in sampled_points]
  

    # Create a new, simplified NURBS curve
    simplified_nurbs = BSpline.Curve()
    simplified_nurbs.degree = degree
    simplified_nurbs.ctrlpts = control_points
    simplified_nurbs.knotvector = utilities.generate_knot_vector(degree, num_control_points)
    simplified_nurbs.weights = [1.0] * num_control_points  # Uniform weights

    return simplified_nurbs

def deform_nurbs_to_fit(bad_layer, reference_layer):
    """
    Adjusts the bad layer's control points to better align with the reference good layer.
    
    Parameters:
    - bad_layer: The bad NURBS curve.
    - reference_layer: The good NURBS curve used as reference.
    
    Returns:
    - deformed_layer: The adjusted NURBS curve.
    """
    # Sample points from both NURBS curves
    bad_points = sample_nurbs_curve(bad_layer, num_points=50)[:, :2]  # Keep only x and y
    reference_points = sample_nurbs_curve(reference_layer, num_points=50)[:, :2]  # Already correct

    displacement_vectors = reference_points - bad_points  # Now both have shape (50,2)

    # Apply a weighted adjustment to move bad points toward reference points
    adjusted_points = bad_points + (displacement_vectors * 0.70)  # 0.7 is a smoothing factor

    # Reconstruct NURBS with the deformed points
    deformed_layer = create_nurbs_from_points(adjusted_points)

    return deformed_layer

def blend_nurbs_layers(bad_layer, reference_layer, blend_factor=0.5):
    """
    Smoothly blends the bad layer towards the reference good layer.

    Parameters:
    - bad_layer: The bad NURBS curve.
    - reference_layer: The good NURBS curve used as reference.
    - blend_factor: A value between 0 (no change) and 1 (fully adopt reference shape).

    Returns:
    - blended_layer: The blended NURBS curve.
    """
    # Sample points from both layers
    bad_points = sample_nurbs_curve(bad_layer, num_points=50)[:, :2]  # Keep only x and y
    reference_points = sample_nurbs_curve(reference_layer, num_points=50)[:, :2]  # Keep only x and y

    blended_points = (1 - blend_factor) * bad_points + blend_factor * reference_points  # Now they match


    # Reconstruct NURBS with the blended points
    blended_layer = create_nurbs_from_points(blended_points)

    return blended_layer


def update_bad_layers(boundaries_with_nurbs, good_layers, bad_layers, debug=True):
    """
    Update bad layers by adjusting them based on the previous good layers.

    Parameters:
    - boundaries_with_nurbs: List of NURBS curves representing the layers.
    - good_layers: Indices of the good layers to reference for updating bad layers.
    - bad_layers: Indices of the bad layers that need to be updated.
    - debug: If True, visualizes the layers and updates.
    
    Returns:
    - corrected_boundaries: List of updated NURBS curves for the layers.
    """

    print(bad_layers)
    print(good_layers)

    corrected_boundaries = boundaries_with_nurbs[:]  # Copy to avoid modifying original data

    # Reverse the order of bad layers to start with the last bad layer
    bad_layers = bad_layers[::-1]

    # Compute average scaling factor only once
    avg_scaling_factor = compute_scaling_factor_between_good_layers(good_layers, boundaries_with_nurbs)
    print(f"Average Scaling Factor: {avg_scaling_factor}")

    # Create a copy of bad_layers to iterate safely
    bad_layers_copy = bad_layers[:]

    # Use the closest good layer above for the first bad layer
    reference_layer = min([g for g in good_layers if g > bad_layers_copy[0]], default=None)

    for bad_layer in bad_layers_copy:
        if reference_layer is None:
            print(f"Warning: No good layer found for bad layer {bad_layer}. Skipping.")
            continue

        print(f"Updating bad layer {bad_layer} using good layer {reference_layer} as reference.")

        # OPTION 1: Apply scaling (Uncomment to use)
        print('Scaling Bad Layers')
        # corrected_layer = apply_scaling_to_layer(boundaries_with_nurbs[bad_layer], avg_scaling_factor)

        # OPTION 2: Apply deformation based on control points (Uncomment to use)
        print('Deforming Bad Layers')
        corrected_layer = deform_nurbs_to_fit(boundaries_with_nurbs[bad_layer], boundaries_with_nurbs[reference_layer])

        # OPTION 3: Apply smooth blending with the reference layer (Uncomment to use)
        print('Belnding Bad Layers')
        # corrected_layer = blend_nurbs_layers(boundaries_with_nurbs[bad_layer], boundaries_with_nurbs[reference_layer], blend_factor=0.6)

        # Update the corrected boundaries
        corrected_boundaries[bad_layer] = corrected_layer

        # Mark this layer as corrected
        good_layers.append(bad_layer)  
        good_layers.sort()  # Keep the order intact
        boundaries_with_nurbs[bad_layer] = corrected_layer  

        # Remove the corrected layer from the bad layers list
        bad_layers.remove(bad_layer)

        # Update reference layer for the next iteration
        reference_layer = bad_layer  # Now, the corrected bad layer is the new reference

    if debug:
        visualize_layers(corrected_boundaries, boundaries_with_nurbs, bad_layers_copy)

    return corrected_boundaries




# def sample_nurbs_curve(boundary, num_samples=100):
#     """
#     Sample a NURBS boundary at equidistant points.
    
#     Parameters:
#     - boundary: A NURBS curve object.
#     - num_samples: Number of samples to take from the boundary.
    
#     Returns:
#     - sampled_points: np.array of shape (num_samples, 2), sampled points on the curve.
#     """
#     param = np.linspace(0, 1, num_samples)  # Generate parameter values from 0 to 1
#     sampled_points = np.array([boundary.evaluate(u) for u in param])  # Evaluate the curve at each parameter u
#     return sampled_points

# def interpolate_boundary(boundary1, boundary2, num_samples=100):
#     """
#     Interpolate between two NURBS boundaries by averaging their sampled points.
    
#     Parameters:
#     - boundary1: First NURBS curve object.
#     - boundary2: Second NURBS curve object.
#     - num_samples: Number of points to sample from each curve.
    
#     Returns:
#     - interpolated_points: np.array of shape (num_samples, 2)
#     """
#     sampled_boundary1 = sample_nurbs_curve(boundary1, num_samples)
#     sampled_boundary2 = sample_nurbs_curve(boundary2, num_samples)

#     # Simple linear interpolation between the two sets of sampled points
#     interpolated_points = (sampled_boundary1 + sampled_boundary2) / 2.0

#     return interpolated_points

# def fit_nurbs_to_points(points, degree=3, num_ctrlpts=None):
#     """
#     Fit a NURBS curve to a set of points using least squares fitting.
    
#     Parameters:
#     - points: np.array of shape (num_samples, 2), representing boundary points.
#     - degree: Degree of the NURBS curve.
#     - num_ctrlpts: Number of control points for the NURBS fit.

#     Returns:
#     - fitted_nurbs: A new fitted NURBS curve object.
#     """
#     if num_ctrlpts is None:
#         num_ctrlpts = len(points) // 2  # Estimate control points based on sampled points

#     # Create and fit a NURBS curve using the points
#     curve = BSpline.Curve()
#     curve.degree = degree
#     curve.ctrlpts = [[pt[0], pt[1]] for pt in points]

#     # Fit the curve to the points
#     curve = fitting.fit_curve(curve, points.tolist(), num_ctrlpts)

#     return curve

# def get_boundary_size(boundary, num_samples=100):
#     """
#     Get the size (width, height) of the boundary based on the convex hull.
    
#     Parameters:
#     - boundary: A NURBS curve object.
#     - num_samples: Number of points to sample from the boundary.
    
#     Returns:
#     - width, height: Dimensions of the boundary.
#     """
#     sampled_points = sample_nurbs_curve(boundary, num_samples)
#     hull = ConvexHull(sampled_points)
#     width = np.max(sampled_points[:, 0]) - np.min(sampled_points[:, 0])
#     height = np.max(sampled_points[:, 1]) - np.min(sampled_points[:, 1])
#     return width, height

# def repair_bad_layers(boundaries_with_nurbs, bad_layers, num_samples=100, debug=False):
#     """
#     Corrects bad layers by interpolating from the closest good layers and adjusting size.
    
#     Parameters:
#     - boundaries_with_nurbs: List of NURBS curve objects.
#     - bad_layers: List of indices of bad layers.
#     - num_samples: Number of points to sample for interpolation.
#     - debug: If True, plots before and after correction.
    
#     Returns:
#     - corrected_boundaries: List of corrected NURBS curves.
#     """
#     corrected_boundaries = boundaries_with_nurbs.copy()

#     for i in bad_layers:
#         # Find the nearest good layers
#         prev_good, next_good = None, None
        
#         for j in range(i-1, -1, -1):  # Search backward for a good layer
#             if j not in bad_layers:
#                 prev_good = j
#                 break

#         for j in range(i+1, len(boundaries_with_nurbs)):  # Search forward for a good layer
#             if j not in bad_layers:
#                 next_good = j
#                 break

#         if prev_good is not None and next_good is not None:
#             # Interpolate between the two nearest good boundaries
#             interpolated_points = interpolate_boundary(
#                 boundaries_with_nurbs[prev_good], 
#                 boundaries_with_nurbs[next_good], 
#                 num_samples
#             )
            
#             # Get size of the boundary to be corrected
#             width_bad, height_bad = get_boundary_size(boundaries_with_nurbs[i], num_samples)
#             width_interp, height_interp = get_boundary_size(interpolated_points, num_samples)
            
#             # Scale interpolated boundary to match the bad layer's size
#             scale_x = width_bad / width_interp if width_interp > 0 else 1
#             scale_y = height_bad / height_interp if height_interp > 0 else 1
#             interpolated_points[:, 0] *= scale_x
#             interpolated_points[:, 1] *= scale_y

#             # Fit a NURBS curve to the scaled points
#             corrected_boundaries[i] = fit_nurbs_to_points(interpolated_points)
        
#         elif prev_good is not None:
#             # If only one good layer is available, use it as a reference
#             corrected_boundaries[i] = boundaries_with_nurbs[prev_good]
        
#         elif next_good is not None:
#             corrected_boundaries[i] = boundaries_with_nurbs[next_good]
    
#     if debug:
#         plt.figure(figsize=(8, 4))
#         for i, boundary in enumerate(boundaries_with_nurbs):
#             sampled_points = sample_nurbs_curve(boundary, num_samples)
#             plt.plot(sampled_points[:, 0], sampled_points[:, 1], 'gray', alpha=0.5)

#         for i in bad_layers:
#             sampled_points = sample_nurbs_curve(corrected_boundaries[i], num_samples)
#             plt.plot(sampled_points[:, 0], sampled_points[:, 1], 'r-', linewidth=2, label=f"Repaired Layer {i}")

#         plt.title("Boundary Repair Visualization")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     return corrected_boundaries


# Main function for user interaction and processing
def main():
    print("=== Additive Manufacturing Toolpath Generation for Repair ===")
    
    missing_volume_file = "/home/miguel-cruz/Documents/AS4RCSD/Dataset/Boolean Comparison/Case1_mv.pcd"  
    
    # Load the point cloud from the saved file
    missing_region = o3d.io.read_point_cloud(missing_volume_file)
    o3d.visualization.draw_geometries([missing_region], window_name="Point Cloud")

    # Compute the axis-aligned bounding box (AABB)
    aabb = missing_region.get_axis_aligned_bounding_box()
    aabb_extent = aabb.get_extent()  # Returns (width, height, depth) in mm

    # Compute the oriented bounding box (OBB) - may give more precise fitting
    obb = missing_region.get_oriented_bounding_box()
    obb_extent = obb.extent  # Returns (width, height, depth) in mm

    # Print bounding box dimensions
    print("Axis-Aligned Bounding Box (AABB) dimensions (mm):", aabb_extent)
    print("Oriented Bounding Box (OBB) dimensions (mm):", obb_extent)


    # Ask the user for the layer height
    layer_height = float(input("Enter the layer height (in mm): "))
    # layer_height = 1.0
    
    # Slice the voxelized repair volume into layers
    # slices, slice_heights = slice_voxelized_cloud(missing_region, layer_height)
    # print(f"Number of slices: {len(slices)}")

    slices, slice_heights = slice_point_cloud(missing_region, layer_height)
    print(f"Number of slices: {len(slices)}")
    print(slices)
    # Ask the user if they want to add extra layers
    add_layers_input = input("Do you want to add more layers? (yes/no): ").strip().lower()

    if add_layers_input == 'yes':
        try:
            num_extra_layers = int(input("How many extra layers do you want to add?: "))
            if num_extra_layers > 0:
                # Exclude the last slice, add extra layers equal to the penultimate slice
                penultimate_slice = slices[-2]  # The second-to-last slice
                for _ in range(num_extra_layers):
                    slices.append(penultimate_slice)
                    slice_heights = np.append(slice_heights, slice_heights[-3])  # Keep the height same as penultimate layer
                
                print(f"Added {num_extra_layers} extra layers.")
        except ValueError:
            print("Invalid number entered for extra layers.")
    
    # Visualize the slices
    print(f"Number of slices after addition: {len(slices)}")
    visualize_slices(slices, slice_heights)
    visualize_point_clouds(slices)


    # # Preprocess the slices
    # # slices_preprocessed = process_boundary_slices(slices, smooth_method="spline", window_size=5)
    # slices_preprocessed = process_adaptive_slices(
    # slices,
    # curvature_threshold=3,  # Adjust based on your boundary sharpness
    # min_distance=1.0,        # Adjust for point density
    # degree=2,                # Degree of the spline
    # num_points=100           # Number of points in the final smoothed curve
    # )

    # # Visualize the preprocessed slices
    # visualize_point_clouds(slices_preprocessed)


    # Generate NURBS boundaries from slices
    boundaries_with_nurbs, layers_with_insufficient_points = determine_bounds_with_nurbs(slices, degree=3) 

    # Detect bad and good layers based on convex hull analysis
    bad_layers, good_layers = detect_bad_layers_by_convex_hull(boundaries_with_nurbs)

    # 🔹 Simplify bad layers before correction
    for bad_layer in bad_layers:
        print(f"Simplifying bad layer {bad_layer} before correction.")
        boundaries_with_nurbs[bad_layer] = simplify_nurbs(boundaries_with_nurbs[bad_layer], num_control_points=12)

    # Now update bad layers using rescaling and reference from good layers
    # corrected_boundaries = update_bad_layers(boundaries_with_nurbs, good_layers, bad_layers, debug=True)


    # Ask if the user wants to see all 2D NURBS curves
    visualize_all = input("Do you want to visualize all 2D NURBS curves for each layer? (yes/no): ").strip().lower()
    
    if visualize_all == 'yes':
        # Visualize all layers' NURBS curves
        for i, curve in enumerate(boundaries_with_nurbs):
            print(f"Visualizing NURBS curve for slice {i}")
            visualize_nurbs_curve(curve)
    else:
        # Ask for a specific layer to visualize
        while True:
            layer_idx = input("Enter the layer index (or 'q' to quit): ").strip()
            if layer_idx.lower() == 'q':
                break
            try:
                layer_idx = int(layer_idx)
                if 0 <= layer_idx < len(boundaries_with_nurbs):
                    print(f"Visualizing NURBS curve for slice {layer_idx}")
                    visualize_nurbs_curve(boundaries_with_nurbs[layer_idx])
                else:
                    print("Invalid index, please enter a valid layer index.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")

    # Visualize all NURBS curves in 3D
    visualize_nurbs_in_3d(boundaries_with_nurbs)
    
    
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

    toolpaths = []
    for nurb in boundaries_with_nurbs:
        idx = boundaries_with_nurbs.index(nurb) + 1
        evaluated_points = np.array([nurb.evaluate_single(t) for t in np.linspace(0, 1, 100)])

        if strategy == 1:

            deposition, tool_moves, fly_lines, fillets = generate_raster_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        elif strategy == 2:
            
            deposition, tool_moves, fly_lines, fillets = generate_zigzag_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle, corner_radius)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        elif strategy == 3:

            deposition, tool_moves, fly_lines, fillets = generate_contour_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle, corner_radius)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        elif strategy == 4:
            
            deposition, tool_moves, fly_lines, fillets = generate_spiral_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle, corner_radius)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        elif strategy == 5:
            
            deposition, tool_moves, fly_lines, fillets = generate_hybrid1_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle, corner_radius)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        elif strategy == 6:
            
            deposition, tool_moves, fly_lines, fillets = generate_hybrid2_toolpath(evaluated_points[:, :2], tool_width, hatch_distance, fly_in_out, fly_distance, idx, raster_angle, corner_radius)
            toolpaths.append((deposition, tool_moves, fly_lines, fillets))

        else:
            print("Invalid choice, using raster toolpath by default.")
            toolpaths.append(generate_raster_toolpath(slice, tool_width))
    
  
    # toolpaths = add_missing_layers_to_toolpaths(toolpaths, slices, layers_with_insufficient_points) 

    full_3d_trajectory = link_layers_and_visualize(toolpaths, layer_height)
    
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



    # Visualize the toolpath in 3D
    # visualize_toolpath_3d(toolpaths, slice_heights)

if __name__ == "__main__":
    main()

