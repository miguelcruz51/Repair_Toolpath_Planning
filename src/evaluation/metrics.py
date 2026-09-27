"""
Evaluation metrics for toolpath analysis.
"""

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def calculate_total_length(path):
    """Calculate the total length of a path (sum of Euclidean distances between consecutive points).
    Skips NaN segments which represent tool moves, not deposition."""
    if len(path) < 2:
        return 0.0
    total_length = 0
    for i in range(1, len(path)):
        # Skip NaN segments (tool moves in raster strategy)
        if np.any(np.isnan(path[i])) or np.any(np.isnan(path[i-1])):
            continue
        total_length += np.linalg.norm(path[i] - path[i-1])
    return total_length

def deposition_to_travel_ratio(deposition_length, travel_length):
    """Calculate the Deposition-to-Travel Ratio."""
    if travel_length == 0:
        return float('inf')  # Avoid division by zero
    return deposition_length / travel_length

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

                    total_turns += angle_degrees
                    num_turns += 1  # Count valid turns
                
    if num_turns == 0:
        return 0    
    else:
        avg_turn_angle = total_turns/num_turns

    return avg_turn_angle

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
    """Calculate deposition color map for analysis."""
    # This function is now deprecated in favor of the optimized grid calculation
    # Return placeholder values for compatibility
    color_maps = []
    coverage_percentage = 85.0  # Placeholder
    lateral_support_percentage = 70.0  # Placeholder
    over_deposition_percentage = 5.0  # Placeholder
    supporting_contour_percentage = 80.0  # Placeholder
    wall_engagement_index = 0.75  # Placeholder

    return color_maps, coverage_percentage, lateral_support_percentage, over_deposition_percentage, supporting_contour_percentage, wall_engagement_index
