#!/usr/bin/env python3
# python AS4RCSD4.py --config example_config.json 
"""
AS4RCSD4.py - Additive Manufacturing Toolpath Generation for Repair (Optimized Version)

A comprehensive toolpath generation system for additive manufacturing repair applications.
Supports multiple strategies, geometric processing, and evaluation metrics.

Author: Optimized from AS4RCSD3.py
"""

import argparse
import sys
import os
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

# Core libraries
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import splprep, splev
from shapely.affinity import rotate
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, Point
import trimesh
import pyvista as pv
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
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
from itertools import combinations
from matplotlib.collections import LineCollection

import matplotlib
matplotlib.use("TkAgg")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ToolpathConfig:
    """Configuration class for toolpath generation parameters."""
    # Input files
    mesh_file: str
    non_planar_surface_file: Optional[str] = None
    
    # Processing parameters
    layer_height: float = 1.0
    add_layers: int = 0
    slicing_axis: str = "z"
    rotate_180: bool = False
    
    # Toolpath parameters
    strategy: int = 1  # 1-6 for different strategies
    tool_width: float = 2.0
    hatch_distance: float = 1.5
    fly_in_out: bool = True
    fly_distance: float = 2.0
    raster_angle: float = 0.0
    corner_radius: float = 0.0
    
    # Geometry processing
    enable_turbine_repair: bool = False
    extension_length: float = 5.0
    spline_smoothing: float = 0.1
    num_circles: int = 5
    circle_resolution: float = 0.05
    circle_tolerance: float = 0.05
    curvature_threshold: float = 2.0
    
    # Hybrid2 specific
    enable_extension: bool = True
    
    # Evaluation and output
    enable_evaluation: bool = True
    evaluate: Union[str, int] = "all"  # "all" for all layers, or specific layer number
    grid_size: int = 500  # Grid resolution for deposition analysis
    save_evaluation_plots: bool = True  # Automatically save evaluation plots
    enable_visualization: bool = True
    export_csv: bool = True
    output_dir: str = "output"
    debug_layer: Optional[int] = None
    
    # Visualization options
    visualize_all_layers: bool = False
    save_plots: bool = False
    show_3d_slices: bool = False
    show_boundary_extension: bool = False
    show_hybrid_process: bool = False
    show_layer_toolpath: bool = False
    show_3d_toolpath: bool = False
    save_layer_toolpath: Optional[int] = None
    save_3d_toolpath: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolpathConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_file: str) -> 'ToolpathConfig':
        """Load config from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, json_file: str) -> None:
        """Save config to JSON file."""
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class ToolpathGenerator:
    """Main class for toolpath generation and processing."""
    
    def __init__(self, config: ToolpathConfig):
        self.config = config
        self.mesh = None
        self.non_planar_mesh = None
        self.slices = []
        self.toolpaths = []
        self.contours = []
        self.processed_boundaries = []  # Store processed boundaries (extended if turbine repair enabled)
        self.turbine_repair_data = None  # Store turbine repair visualization data
        self.setup_output_directory()
        
    def setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_meshes(self):
        """Load mesh files."""
        logger.info(f"Loading main mesh: {self.config.mesh_file}")
        if not os.path.exists(self.config.mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {self.config.mesh_file}")
            
        self.mesh = o3d.io.read_triangle_mesh(self.config.mesh_file)
        if len(self.mesh.vertices) == 0:
            raise ValueError("Failed to load main mesh or mesh is empty")
            
        # Load non-planar surface if provided
        if self.config.non_planar_surface_file:
            logger.info(f"Loading non-planar surface: {self.config.non_planar_surface_file}")
            if not os.path.exists(self.config.non_planar_surface_file):
                raise FileNotFoundError(f"Non-planar surface file not found: {self.config.non_planar_surface_file}")
                
            self.non_planar_mesh = o3d.io.read_triangle_mesh(self.config.non_planar_surface_file)
            if len(self.non_planar_mesh.vertices) == 0:
                raise ValueError("Failed to load non-planar mesh or mesh is empty")
        
        # Apply colors for visualization
        self.mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
        if self.non_planar_mesh:
            self.non_planar_mesh.paint_uniform_color([1.0, 0.3, 0.3])  # Red
            
        logger.info("Meshes loaded successfully")
        
    def apply_transformations(self):
        """Apply rotations and transformations to meshes."""
        if not self.mesh:
            raise ValueError("No mesh loaded")
            
        meshes_to_transform = [self.mesh]
        if self.non_planar_mesh:
            meshes_to_transform.append(self.non_planar_mesh)
            
        # Compute common center for consistent transformations
        if len(meshes_to_transform) > 1:
            common_center = sum(mesh.get_center() for mesh in meshes_to_transform) / len(meshes_to_transform)
        else:
            common_center = self.mesh.get_center()
            
        # Apply axis alignment
        axis_map = {"x": 0, "y": 1, "z": 2}
        if self.config.slicing_axis == "x":
            logger.info("Rotating meshes to align X with Z...")
            for mesh in meshes_to_transform:
                self.rotate_mesh(mesh, "y", 90, center=common_center)
        elif self.config.slicing_axis == "y":
            logger.info("Rotating meshes to align Y with Z...")
            for mesh in meshes_to_transform:
                self.rotate_mesh(mesh, "x", 90, center=common_center)
                
        # Apply 180° rotation if requested
        if self.config.rotate_180:
            logger.info("Rotating meshes 180°...")
            for mesh in meshes_to_transform:
                self.rotate_mesh(mesh, "x", 180, center=common_center)
                
        # Compute vertex normals
        for mesh in meshes_to_transform:
            mesh.compute_vertex_normals()
            
        logger.info("Transformations applied successfully")
        
    def rotate_mesh(self, mesh, axis, angle, center=None):
        """Rotate mesh around a given axis by the specified angle in degrees."""
        radians = np.deg2rad(angle)
        axis_map = {"x": (radians, 0, 0), "y": (0, radians, 0), "z": (0, 0, radians)}

        if axis in axis_map:
            R = mesh.get_rotation_matrix_from_xyz(axis_map[axis])
            mesh.rotate(R, center=center if center is not None else mesh.get_center())
        else:
            logger.warning(f"Invalid axis: {axis}. No rotation applied.")

        return mesh
        
    def slice_meshes(self):
        """Slice the main mesh into layers."""
        logger.info(f"Slicing mesh with layer height: {self.config.layer_height}mm")
        self.slices = self.slice_mesh(
            self.mesh, 
            self.config.layer_height, 
            self.config.add_layers
        )
        logger.info(f"Generated {len(self.slices)} slices")
        
    def slice_mesh(self, mesh, layer_height=0.5, add_layers=0):
        """Slice the mesh along the Z-axis based on deposition layer height."""
        axis_idx = 2  # Z-axis
       
        # Get bounding box limits
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound, max_bound = bbox.min_bound[axis_idx], bbox.max_bound[axis_idx]
        
        # Compute number of slices based on layer height
        num_slices = int((max_bound - min_bound) / layer_height)
        logger.info(f"Total slices: {num_slices}, Total layers: {num_slices - 1 + add_layers}")
        
        slices = []

        # Loop through each slice height and generate contours
        for i in range(num_slices):
            slice_position = min_bound + i * layer_height
            
            # Create a slicing plane
            plane = np.zeros((4,))
            plane[axis_idx] = 1  # Normal to slicing axis
            plane[3] = -slice_position  # Plane equation

            # Compute the slice boundary for the mesh at the current height
            slice_boundary = self.extract_slice(mesh, plane)

            if slice_boundary is not None:
                slices.append((slice_position, slice_boundary))

        logger.info(f"Slicing complete. {len(slices)} valid cross-sections generated.")

        # Add extra layers if requested
        if add_layers > 0 and len(slices) > 1:
            logger.info(f"Adding {add_layers} extra layers")
            last_layer_position, last_layer_boundary = slices[-1]
        
            for i in range(add_layers):
                new_layer_position = last_layer_position + (i + 1) * layer_height
                slices.append((new_layer_position, last_layer_boundary))

        return slices
        
    def extract_slice(self, mesh, plane):
        """Extract the slice of a mesh at a specific height using a slicing plane."""
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

            # Check for intersection
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
            return None

        intersection_points = np.array(intersection_points)

        # Project points onto the 2D plane (remove the slicing axis)
        slicing_axis_idx = np.argmax(np.abs(plane_normal))
        
        if slicing_axis_idx == 0:
            intersection_points = intersection_points[:, [1, 2]]
        elif slicing_axis_idx == 1:
            intersection_points = intersection_points[:, [0, 2]]
        elif slicing_axis_idx == 2:
            intersection_points = intersection_points[:, [0, 1]]

        # Create a polygon from the 2D intersection points (Convex Hull)
        if len(intersection_points) > 2:
            try:
                hull = ConvexHull(intersection_points)
                hull_points = intersection_points[hull.vertices]
                return hull_points
            except:
                logger.warning("Error in computing convex hull")
                return intersection_points
        else:
            return intersection_points
            
    def process_turbine_repair(self, boundary_points, layer_index):
        """Process turbine blade repair geometry if enabled."""
        # Always ensure the initial boundary is closed first
        boundary_points = self.ensure_closed_boundary(boundary_points)
        
        if not self.config.enable_turbine_repair:
            logger.debug("Turbine repair not enabled, returning closed original boundary")
            return boundary_points
            
        try:
            logger.info(f"Processing turbine repair for layer {layer_index}")
            
            # Calculate original boundary area for comparison
            original_area = Polygon(boundary_points).area
            logger.debug(f"Original boundary area: {original_area:.2f}")
            
            # Densify boundary points
            densified_points = self.densify_boundary(boundary_points, step=0.2)
            polygon = Polygon(densified_points)
            
            # Fit circles
            fitted_circles, circles_with_boundary_points = self.fit_largest_circles(
                polygon, densified_points, 
                num_circles=self.config.num_circles,
                resolution=self.config.circle_resolution,
                tolerance=self.config.circle_tolerance
            )
            
            if not fitted_circles:
                logger.warning("No circles fitted, using original boundary")
                return boundary_points
                
            # Compute curvature influence
            centers, curvature_values, x_spline, y_spline = self.compute_curvature_influence(fitted_circles)
            
            # Filter centers by curvature
            filtered_centers, filtered_curvatures = self.filter_centers_by_curvature(
                centers, curvature_values, z_thresh=self.config.curvature_threshold
            )
            
            if len(filtered_centers) < 2:
                logger.warning("Insufficient filtered centers, using original boundary")
                return boundary_points
                
            # Refit spline and process
            x_spline, y_spline = self.refit_spline_from_filtered_centers(filtered_centers)
            filtered_circles = [c for c in circles_with_boundary_points 
                              if any(np.isclose(c[0], center[0]) and np.isclose(c[1], center[1]) 
                                   for center in filtered_centers)]
            
            # Fit left and right splines
            left_spline, right_spline, left_points, right_points = self.fit_left_right_splines_from_circle_contacts(filtered_circles)
            
            # Extend splines
            x_left_ext, y_left_ext = self.extend_spline(left_spline)
            x_right_ext, y_right_ext = self.extend_spline(right_spline)
            
            # Find intersection
            intersection_x, intersection_y = self.compute_spline_intersection(x_left_ext, y_left_ext, x_right_ext, y_right_ext)
            
            if intersection_x is None:
                logger.warning("No spline intersection found, using original boundary")
                return boundary_points
                
            intersection = np.array([intersection_x, intersection_y])
            
            # Find best circle (highest Y)
            best_circle = max(filtered_circles, key=lambda c: c[1])
            
            # Rebuild boundary
            final_boundary = self.rebuild_boundary_with_spline_top(
                boundary_points, best_circle[3][0], best_circle[3][1],
                x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection
            )
            
            final_boundary = self.sort_points_counterclockwise(final_boundary)
            
            # Ensure the final boundary is properly closed (last point connects to first)
            final_boundary = self.ensure_closed_boundary(final_boundary)
            
            # Store data for visualization if boundary extension is enabled
            if self.config.show_boundary_extension:
                self.turbine_repair_data = {
                    'layer_index': layer_index,
                    'original_boundary': boundary_points,
                    'fitted_circles': fitted_circles,
                    'circles_with_boundary_points': circles_with_boundary_points,
                    'centers': centers,
                    'curvature_values': curvature_values,
                    'filtered_centers': filtered_centers,
                    'filtered_curvatures': filtered_curvatures,
                    'x_spline': x_spline,
                    'y_spline': y_spline,
                    'left_spline': left_spline,
                    'right_spline': right_spline,
                    'left_points': left_points,
                    'right_points': right_points,
                    'x_left_ext': x_left_ext,
                    'y_left_ext': y_left_ext,
                    'x_right_ext': x_right_ext,
                    'y_right_ext': y_right_ext,
                    'intersection_x': intersection_x,
                    'intersection_y': intersection_y,
                    'intersection': intersection,
                    'best_circle': best_circle,
                    'final_boundary': final_boundary
                }
            
            # Calculate extended boundary area for comparison
            extended_area = Polygon(final_boundary).area
            area_increase = ((extended_area - original_area) / original_area) * 100
            
            logger.info(f"Turbine repair processing completed successfully. Extended boundary has {len(final_boundary)} points.")
            logger.info(f"Boundary area increased by {area_increase:.1f}% (from {original_area:.2f} to {extended_area:.2f})")
            logger.info("Extended boundary will be used for all subsequent toolpath generation.")
            logger.debug(f"Extended boundary bounds: {np.min(final_boundary, axis=0)} to {np.max(final_boundary, axis=0)}")
            return final_boundary
            
        except Exception as e:
            logger.error(f"Error in turbine repair processing: {e}")
            return boundary_points
            
    def get_layer_boundary(self, layer_idx):
        """Get the correct boundary for a layer (extended if turbine repair enabled, original otherwise)."""
        if layer_idx < len(self.processed_boundaries):
            # Use the stored processed boundary (extended if turbine repair was enabled)
            slice_position, processed_boundary = self.processed_boundaries[layer_idx]
            logger.debug(f"Using processed boundary for layer {layer_idx} visualization ({len(processed_boundary)} points)")
            return slice_position, processed_boundary
        else:
            # Fallback to original boundary if processed boundary not available
            slice_position, original_boundary = self.slices[layer_idx]
            logger.warning(f"Processed boundary not available for layer {layer_idx}, using original")
            return slice_position, original_boundary
            
    def extract_surface_contour(self, layer_height):
        """Extract contour from non-planar surface at given height."""
        if not self.non_planar_mesh:
            return np.array([])
            
        try:
            # Convert to point cloud and slice
            pcd = self.non_planar_mesh.sample_points_uniformly(number_of_points=100000)
            contour = self.slice_surface(pcd, layer_height)
            return np.vstack(contour) if len(contour) > 0 else np.array([])
        except Exception as e:
            logger.warning(f"Failed to extract surface contour: {e}")
            return np.array([])
            
    def slice_surface(self, pcd, layer_height, tolerance=0.05):
        """Slice a surface (point cloud) at specified height."""
        points = np.asarray(pcd.points)
        mask = np.abs(points[:, 2] - layer_height) < tolerance
        slice_points = points[mask][:, :2]  # Keep only (X, Y)
        return slice_points
        
    def generate_toolpaths(self):
        """Generate toolpaths for all layers."""
        logger.info(f"Generating toolpaths using strategy {self.config.strategy}")
        
        strategy_map = {
            1: self.generate_raster_toolpath,
            2: self.generate_zigzag_toolpath,
            3: self.generate_contour_toolpath,
            4: self.generate_spiral_toolpath,
            5: self.generate_hybrid1_toolpath,
            6: self.generate_hybrid2_toolpath
        }
        
        strategy_func = strategy_map.get(self.config.strategy, self.generate_raster_toolpath)
        
        for layer_index, (layer_height, boundary_points) in enumerate(self.slices):
            logger.info(f"Processing Layer {layer_index+1}/{len(self.slices)} at height {layer_height:.3f}")
            
            # Process turbine repair if enabled (this will also ensure boundary is closed)
            processed_boundary = self.process_turbine_repair(boundary_points, layer_index)
            
            # Store the processed boundary for later use in visualizations
            self.processed_boundaries.append((layer_height, processed_boundary))
            
            # Log boundary information for debugging
            logger.debug(f"Original boundary points: {len(boundary_points)}")
            logger.debug(f"Processed boundary points: {len(processed_boundary)}")
            
            # Use the processed boundary (extended if turbine repair enabled, or closed if not)
            boundary_points = processed_boundary
            
            # Extract surface contour
            contour = self.extract_surface_contour(layer_height)
            
            try:
                # Generate toolpath based on strategy
                if self.config.strategy == 6:  # Hybrid2 needs contour
                    deposition, tool_moves, fly_lines, fillets = strategy_func(
                        boundary_points, contour, layer_index
                    )
                else:
                    deposition, tool_moves, fly_lines, fillets = strategy_func(
                        boundary_points, layer_index
                    )
                    
                self.toolpaths.append((deposition, tool_moves, fly_lines, fillets))
                self.contours.append(contour)
                
            except Exception as e:
                logger.error(f"Error generating toolpath for layer {layer_index}: {e}")
                # Add empty toolpath to maintain indexing
                self.toolpaths.append((np.array([]), np.array([]), np.array([]), np.array([])))
                self.contours.append(np.array([]))
                
        logger.info(f"Toolpath generation completed. Generated {len(self.toolpaths)} toolpaths")
        
    def generate_raster_toolpath(self, boundary_points, layer_index):
        """Generate raster toolpath for a single boundary."""
        boundary_points = np.asarray(boundary_points)
        if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
            raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

        # Ensure the boundary is closed
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])

        # Create Shapely Polygon with inner offset for tool width
        try:
            slice_polygon = Polygon(boundary_points)
        except ValueError as e:
            logger.error(f"Error creating polygon: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Apply inner offset of half tool width for proper boundary clearance
        offset_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
        if offset_polygon.is_empty or not offset_polygon.is_valid:
            logger.warning(f"Polygon too small for tool width offset in layer {layer_index}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Determine the rotation angle based on the layer index
        effective_angle = self.config.raster_angle if layer_index % 2 != 0 else 0

        # Rotate the polygon by the effective_angle
        rotated_polygon = rotate(offset_polygon, angle=effective_angle, origin='center', use_radians=False)

        # Compute the bounding box and generate raster lines
        min_x, min_y, max_x, max_y = rotated_polygon.bounds

        deposition_lines = []
        tool_moves = []
        fly_lines = []

        y = min_y
        while y <= max_y:
            line_start = (min_x, y)
            line_end = (max_x, y)
            line = LineString([line_start, line_end])
            clipped_line = rotated_polygon.intersection(line)

            if not clipped_line.is_empty:
                if isinstance(clipped_line, LineString):
                    coords = np.array(clipped_line.coords)
                    if np.allclose(coords[:, 1], coords[0, 1]):
                        deposition_lines.append(coords)  
                elif isinstance(clipped_line, MultiLineString):
                    for subline in clipped_line.geoms:
                        coords = np.array(subline.coords)
                        if np.allclose(coords[:, 1], coords[0, 1]):  
                            deposition_lines.append(coords)

            y += self.config.hatch_distance

        # Handle fly-in/out and tool moves
        if self.config.fly_in_out:
            for i, line in enumerate(deposition_lines):
                start_point = line[0]
                end_point = line[-1]

                # Calculate direction vectors
                start_dir = (line[1] - start_point) / np.linalg.norm(line[1] - start_point)
                end_dir = (end_point - line[-2]) / np.linalg.norm(end_point - line[-2])

                # Add fly-in and fly-out points
                fly_in_point = start_point - self.config.fly_distance * start_dir
                fly_out_point = end_point + self.config.fly_distance * end_dir
                fly_lines.append(np.array([fly_in_point, start_point]))
                fly_lines.append(np.array([end_point, fly_out_point]))

                # Add diagonal tool move between lines
                if i < len(deposition_lines) - 1:
                    current_line = deposition_lines[i]
                    next_line = deposition_lines[i + 1]

                    fly_out_point = current_line[-1] + self.config.fly_distance * (current_line[-1] - current_line[0]) / np.linalg.norm(current_line[-1] - current_line[0])
                    fly_in_point = next_line[0] - self.config.fly_distance * (next_line[1] - next_line[0]) / np.linalg.norm(next_line[1] - next_line[0])

                    tool_moves.append(np.array([fly_out_point, fly_in_point]))
        else:
            # Add simple straight movements between consecutive deposition lines
            for i in range(len(deposition_lines) - 1):
                tool_moves.append(np.array([deposition_lines[i][-1], deposition_lines[i + 1][0]]))

        # Combine results - FIXED: Keep deposition lines separate, connections are tool moves
        # Each horizontal line is separate deposition, diagonal connections are tool moves
        all_deposition_points = []
        
        for line in deposition_lines:
            all_deposition_points.extend(line)
        
        deposition_array = np.array(all_deposition_points) if all_deposition_points else np.array([])
        tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
        fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
        fillets_array = np.array([])

        # Rotate back to original orientation
        if effective_angle != 0:
            deposition_array = self.rotate_points(deposition_array, -effective_angle, offset_polygon.centroid)
            tool_moves_array = self.rotate_points(tool_moves_array, -effective_angle, offset_polygon.centroid)
            fly_lines_array = self.rotate_points(fly_lines_array, -effective_angle, offset_polygon.centroid)

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array
        
    def generate_zigzag_toolpath(self, boundary_points, layer_index):
        """Generate zig-zag toolpath with optional corner smoothing."""
        boundary_points = np.asarray(boundary_points)
        if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
            raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

        # Ensure the boundary is closed
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])

        # Create Shapely Polygon with inner offset for tool width
        slice_polygon = Polygon(boundary_points)
        
        # Apply inner offset of half tool width for proper boundary clearance
        offset_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
        if offset_polygon.is_empty or not offset_polygon.is_valid:
            logger.warning(f"Polygon too small for tool width offset in layer {layer_index}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Determine the rotation angle based on the layer index
        effective_angle = self.config.raster_angle if layer_index % 2 != 0 else 0

        # Rotate the polygon by the effective_angle
        rotated_polygon = rotate(offset_polygon, angle=effective_angle, origin='center', use_radians=False)

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

            y += self.config.hatch_distance

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

                if self.config.corner_radius > 0:
                    # Add fillet arcs and connecting line
                    arc1, connecting_line, arc2 = self.create_fillet_between_racks(prev_end, curr_start, self.config.corner_radius)
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
        if self.config.fly_in_out and zig_zag_lines:
            start_point = zig_zag_lines[0][0]
            end_point = zig_zag_lines[-1][-1]

            # Calculate direction vectors for fly-in and fly-out
            start_dir = (zig_zag_lines[0][1] - start_point) / np.linalg.norm(zig_zag_lines[0][1] - start_point)
            end_dir = (end_point - zig_zag_lines[-1][-2]) / np.linalg.norm(end_point - zig_zag_lines[-1][-2])

            fly_in_point = start_point - self.config.fly_distance * start_dir
            fly_out_point = end_point + self.config.fly_distance * end_dir

            fly_lines.append(np.array([fly_in_point, start_point]))
            fly_lines.append(np.array([end_point, fly_out_point]))

        # Combine results
        deposition_array = np.vstack(zig_zag_lines) if zig_zag_lines else np.array([])
        tool_moves_array = np.array([])  # No tool moves for zig-zag pattern
        fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
        fillets_array = np.vstack(fillets) if fillets else np.array([])

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array
        
    def generate_contour_toolpath(self, boundary_points, layer_index):
        """Generate contour toolpath by offsetting the polygon inward multiple times."""
        boundary_points = np.asarray(boundary_points)
        if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
            raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

        # Ensure the boundary is closed
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])

        slice_polygon = Polygon(boundary_points)

        if not slice_polygon.is_valid:
            logger.warning(f"Invalid polygon detected: {slice_polygon}")
            slice_polygon = slice_polygon.buffer(0)
            if not slice_polygon.is_valid:
                logger.error("Could not fix the polygon. Skipping this slice.")
                return np.array([]), np.array([]), np.array([]), np.array([])

        # Start with inner offset of half tool width for proper boundary clearance
        current_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
        if current_polygon.is_empty or not current_polygon.is_valid:
            logger.warning(f"Polygon too small for tool width offset in layer {layer_index}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        deposition_lines = []
        tool_moves = []
        
        while current_polygon.area > 0:
            if isinstance(current_polygon, MultiPolygon):
                current_polygon = max(current_polygon.geoms, key=lambda p: p.area)

            coords = np.array(current_polygon.exterior.coords)

            # Ensure each contour is closed
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            deposition_lines.append(coords)
            
            # Add tool move to next contour (if there will be one)
            next_polygon = current_polygon.buffer(-self.config.hatch_distance)
            if not next_polygon.is_empty and next_polygon.is_valid and next_polygon.area > 0:
                if isinstance(next_polygon, MultiPolygon):
                    next_polygon = max(next_polygon.geoms, key=lambda p: p.area)
                
                next_coords = np.array(next_polygon.exterior.coords)
                if not np.allclose(next_coords[0], next_coords[-1]):
                    next_coords = np.vstack([next_coords, next_coords[0]])
                
                # Tool move from end of current contour to start of next contour
                tool_moves.append(np.array([coords[-1], next_coords[0]]))

            current_polygon = next_polygon
            if current_polygon.is_empty or not current_polygon.is_valid:
                break

        # Simple approach: keep deposition and tool moves separate
        # Deposition array contains only the contour points
        # Tool moves array contains the connecting segments
        
        all_deposition_points = []
        for line in deposition_lines:
            all_deposition_points.extend(line)
        
        deposition_array = np.array(all_deposition_points) if all_deposition_points else np.array([])
        tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
        fly_lines_array = np.array([])
        fillets_array = np.array([])

        logger.info(f"Layer {layer_index}: Generated {len(deposition_lines)} contours, "
                   f"{len(all_deposition_points)} deposition points, {len(tool_moves)} tool moves")
        
        if len(tool_moves) > 0:
            logger.info(f"Layer {layer_index}: Tool moves shape: {tool_moves_array.shape}")
            for i, tm in enumerate(tool_moves):
                logger.info(f"  Tool move {i}: from {tm[0]} to {tm[1]}")

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array
    
    def _store_move_types(self, layer_index: int, move_types: List[str]):
        """Store move types for later use in CSV generation"""
        if not hasattr(self, 'layer_move_types'):
            self.layer_move_types = {}
        self.layer_move_types[layer_index] = move_types
        
    def generate_spiral_toolpath(self, boundary_points, layer_index):
        """Generate continuous spiral toolpath."""
        boundary_points = np.asarray(boundary_points)
        if len(boundary_points.shape) != 2 or boundary_points.shape[1] != 2:
            raise ValueError("Boundary points must be a 2D array with shape (N, 2).")

        # Ensure the boundary is closed
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])

        slice_polygon = Polygon(boundary_points)

        if not slice_polygon.is_valid:
            logger.warning(f"Invalid polygon detected: {slice_polygon}")
            slice_polygon = slice_polygon.buffer(0)
            if not slice_polygon.is_valid:
                logger.error("Could not fix the polygon. Skipping this slice.")
                return np.array([]), np.array([]), np.array([]), np.array([])

        # Start with inner offset of half tool width for proper boundary clearance
        current_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
        if current_polygon.is_empty or not current_polygon.is_valid:
            logger.warning(f"Polygon too small for tool width offset in layer {layer_index}")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Initialize the spiral path
        spiral_points = []
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
                num_steps = max(int(np.linalg.norm([x2 - x1, y2 - y1]) / self.config.hatch_distance), 1)
                for t in np.linspace(0, 1, num_steps):
                    spiral_points.append([(1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2])

            # Offset inward for the next spiral loop
            current_polygon = current_polygon.buffer(-self.config.hatch_distance)
            if current_polygon.is_empty or not current_polygon.is_valid:
                break

        # Convert to NumPy array
        spiral_points = np.array(spiral_points)

        # Optional: Add fly-in moves if required
        fly_lines_array = np.array([])
        if self.config.fly_in_out and len(spiral_points) > 0:
            # Fly-in moves (from a specified location to the start of the spiral)
            start_point = spiral_points[0]
            fly_lines_array = np.array([np.array([boundary_points[0], start_point])])

        # Combine deposition lines and tool moves into arrays
        deposition_array = spiral_points
        tool_moves_array = np.array([])
        fillets_array = np.array([])

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array
        
    def generate_hybrid1_toolpath(self, boundary_points, layer_index):
        """Generate hybrid toolpath combining contour deposition and zig-zag filling."""
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
                logger.error("Could not fix the polygon. Skipping this slice.")
                return np.array([]), np.array([]), np.array([]), np.array([])

        ### Step 1: Generate Contour Deposition ###
        # Offset the boundary inward by half the tool width to create a contour path
        contour_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
        if contour_polygon.is_empty:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Get contour deposition points
        contour_points = np.array(contour_polygon.exterior.coords)
        contour_deposition = [contour_points]

        ### Step 2: Generate Zig-Zag Fill ###
        # Further offset to define the raster area
        fill_polygon = contour_polygon.buffer(-self.config.tool_width)
        if fill_polygon.is_empty:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Rotate the polygon for zig-zag alignment
        rotated_fill_polygon = rotate(fill_polygon, self.config.raster_angle, origin='center')
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

            y += self.config.hatch_distance

        # Reverse alternate lines for proper zigzag movement
        for i in range(len(zigzag_lines)):
            if i % 2 == 1:
                zigzag_lines[i] = zigzag_lines[i][::-1]

            # Add tool move between adjacent lines
            if i > 0:
                tool_moves.append(np.array([zigzag_lines[i - 1][-1], zigzag_lines[i][0]]))

        ### Step 3: Connect Contour to Zig-Zag (Tool Move) ###
        # FIXED: Tool move from contour end to zigzag start
        if zigzag_lines and contour_deposition:
            # Rotate zigzag lines back to original orientation
            for i in range(len(zigzag_lines)):
                zigzag_lines[i] = self.rotate_points(zigzag_lines[i], -self.config.raster_angle, fill_polygon.centroid)
            
            # Tool move from end of contour to start of first zigzag line
            tool_moves.insert(0, np.array([contour_deposition[-1][-1], zigzag_lines[0][0]]))

        ### Step 4: Fly-in and Fly-out ###
        fly_lines = []
        if self.config.fly_in_out and len(contour_deposition) > 0:
            start_point = contour_deposition[0][0]
            end_point = zigzag_lines[-1][-1] if zigzag_lines else contour_deposition[-1][-1]

            # Fly-in
            fly_lines.append(np.array([start_point - [self.config.fly_distance, 0], start_point]))

            # Fly-out
            fly_lines.append(np.array([end_point, end_point + [self.config.fly_distance, 0]]))

        ### Finalize Arrays ###
        # FIXED: Keep contour and zigzag separate, connection is tool move
        all_deposition_points = []
        
        # Add contour deposition points
        for contour_line in contour_deposition:
            all_deposition_points.extend(contour_line)
        
        # Add zigzag deposition points  
        for zigzag_line in zigzag_lines:
            all_deposition_points.extend(zigzag_line)
        
        deposition_array = np.array(all_deposition_points) if all_deposition_points else np.array([])
        tool_moves_array = np.vstack(tool_moves) if tool_moves else np.array([])
        fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
        fillets_array = np.array([])  # No fillets in hybrid path

        return deposition_array, tool_moves_array, fly_lines_array, fillets_array
        
    def generate_hybrid2_toolpath(self, boundary_points, non_planar_boundary_points, layer_index):
        """Generate hybrid toolpath with improved extension and inner offset approach.
        
        Strategy:
        1. Create inner offset boundary (tool width compensation)
        2. Generate offset lines from damage contour 
        3. Ensure all lines contact boundary with smart extension
        4. Trim and link with zigzag pattern
        """
        
        logger.debug(f"Hybrid2 received boundary with {len(boundary_points)} points")
        logger.debug(f"Boundary bounds: {np.min(boundary_points, axis=0)} to {np.max(boundary_points, axis=0)}")
        
        # Check if this looks like an extended boundary (more points than typical)
        if len(boundary_points) > 50:  # Typical slice might have 20-40 points, extended could have 100+
            logger.info("Hybrid2 appears to be working with extended boundary (high point count)")
        else:
            logger.info("Hybrid2 working with standard boundary")
        
        def find_boundary_intersection(point, direction, boundary_polygon, max_distance=50):
            """Find intersection with boundary in given direction with validation."""
            try:
                # Normalize direction
                direction = direction / np.linalg.norm(direction)
                
                # Create ray from point in direction
                end_point = point + direction * max_distance
                ray = LineString([point, end_point])
                
                # Find intersection with boundary
                intersection = boundary_polygon.boundary.intersection(ray)
                
                if intersection.is_empty:
                    return None
                    
                if hasattr(intersection, 'coords'):
                    # Single point intersection
                    return np.array(intersection.coords[0])
                elif hasattr(intersection, 'geoms'):
                    # Multiple intersections - take closest
                    closest_point = min(intersection.geoms, 
                                      key=lambda p: np.linalg.norm(np.array(p.coords[0]) - point))
                    return np.array(closest_point.coords[0])
                else:
                    return None
            except Exception:
                return None
        
        def validate_line_orientation(line_coords, expected_direction):
            """Validate that line has correct orientation."""
            if len(line_coords) < 2:
                return True
                
            line_vector = line_coords[-1] - line_coords[0]
            line_length = np.linalg.norm(line_vector)
            
            if line_length < 1e-6:
                return True
                
            line_direction = line_vector / line_length
            dot_product = np.dot(line_direction, expected_direction)
            
            # Allow some tolerance for orientation (cos(45°) ≈ 0.7)
            return abs(dot_product) > 0.5
        
        def smart_extend_line(line_coords, boundary_polygon, bounds):
            """Smart extension that validates orientation and prevents wrong directions."""
            if len(line_coords) < 2:
                return line_coords
                
            minx, miny, maxx, maxy = bounds
            start_pt = line_coords[0]
            end_pt = line_coords[-1]
            
            # Calculate line's natural direction
            line_vector = end_pt - start_pt
            line_length = np.linalg.norm(line_vector)
            
            if line_length < 1e-6:
                return line_coords
                
            natural_direction = line_vector / line_length
            
            # Determine best extension directions based on boundary proximity
            extension_candidates = []
            
            # Try extending in line's natural direction
            forward_intersection = find_boundary_intersection(end_pt, natural_direction, boundary_polygon)
            if forward_intersection is not None:
                forward_line = np.vstack([line_coords, forward_intersection])
                if validate_line_orientation(forward_line, natural_direction):
                    extension_candidates.append(forward_line)
            
            backward_intersection = find_boundary_intersection(start_pt, -natural_direction, boundary_polygon)
            if backward_intersection is not None:
                backward_line = np.vstack([backward_intersection, line_coords])
                if validate_line_orientation(backward_line, natural_direction):
                    extension_candidates.append(backward_line)
            
            # Try cardinal directions if natural extension fails
            if not extension_candidates:
                cardinal_directions = [
                    np.array([1.0, 0.0]),   # Right
                    np.array([-1.0, 0.0]),  # Left
                    np.array([0.0, 1.0]),   # Up
                    np.array([0.0, -1.0])   # Down
                ]
                
                for direction in cardinal_directions:
                    # Extend from both ends
                    start_ext = find_boundary_intersection(start_pt, -direction, boundary_polygon)
                    end_ext = find_boundary_intersection(end_pt, direction, boundary_polygon)
                    
                    extended_coords = line_coords.copy()
                    if start_ext is not None:
                        extended_coords = np.vstack([start_ext, extended_coords])
                    if end_ext is not None:
                        extended_coords = np.vstack([extended_coords, end_ext])
                    
                    if validate_line_orientation(extended_coords, direction):
                        extension_candidates.append(extended_coords)
            
            # Choose best extension (longest line within boundary)
            if extension_candidates:
                best_candidate = max(extension_candidates, 
                                   key=lambda coords: np.linalg.norm(coords[-1] - coords[0]))
                return best_candidate
            else:
                return line_coords
        
        # Ensure boundary is closed (should already be closed from turbine repair process)
        if not np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
            logger.debug("Hybrid2: Added closing point to boundary")
        else:
            logger.debug("Hybrid2: Boundary already closed")

        # Create boundary polygon with inner offset for tool width compensation
        original_boundary = Polygon(boundary_points)
        if not original_boundary.is_valid:
            original_boundary = original_boundary.buffer(0)
            if not original_boundary.is_valid:
                raise ValueError("Invalid boundary polygon")
        
        # Create inner offset boundary (half tool width inward)
        inner_offset_distance = self.config.tool_width / 2
        working_boundary = original_boundary.buffer(-inner_offset_distance)
        
        if working_boundary.is_empty:
            logger.warning("Boundary too small for tool width, using original boundary")
            working_boundary = original_boundary

        if len(non_planar_boundary_points) < 2:
            logger.warning("Non-planar boundary must have at least two points")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Create damage contour line (sorted by X coordinate)
        ordered_points = sorted(non_planar_boundary_points, key=lambda x: x[0])
        contour_line = LineString(ordered_points)

        if not contour_line.is_valid:
            raise ValueError("Invalid contour line from non-planar boundary points")

        # Step 1: Generate offset lines from damage contour
        offset_lines = []
        direction_flag = True
        current_offset = self.config.hatch_distance
        bounds = working_boundary.bounds
        max_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) + 10

        while current_offset <= max_extent:
            # Try both positive and negative offsets to find the right direction
            for offset_sign in [-1, 1]:
                try:
                    offset_contour = contour_line.parallel_offset(
                        current_offset * offset_sign, side='right', resolution=50
                    )

                    if isinstance(offset_contour, (MultiLineString, LineString)):
                        line = offset_contour
                        if isinstance(offset_contour, MultiLineString):
                            # Take the longest segment
                            line = max(offset_contour.geoms, key=lambda g: g.length)

                        # Check if this offset intersects with working boundary
                        if working_boundary.intersects(line):
                            coords = np.array(line.coords)
                            if not direction_flag:
                                coords = coords[::-1]  # Reverse for zigzag pattern
                            offset_lines.append(coords)
                            direction_flag = not direction_flag
                            break  # Found valid offset, move to next distance
                            
                except Exception as e:
                    logger.debug(f"Failed to create offset {current_offset * offset_sign}: {e}")
                    continue
            
            current_offset += self.config.hatch_distance

        if not offset_lines:
            logger.warning("No offset lines generated")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Step 2: Ensure all lines contact boundary with smart extension
        extended_lines = []
        if self.config.enable_extension:
            for line_coords in offset_lines:
                extended_line = smart_extend_line(line_coords, working_boundary, bounds)
                extended_lines.append(extended_line)
        else:
            extended_lines = offset_lines

        # Step 3: Trim all lines to fit within working boundary
        final_lines = []
        for line_coords in extended_lines:
            line = LineString(line_coords)
            clipped = working_boundary.intersection(line)
            
            if not clipped.is_empty:
                if isinstance(clipped, LineString):
                    final_lines.append(np.array(clipped.coords))
                elif isinstance(clipped, MultiLineString):
                    # Take the longest segment
                    longest_segment = max(clipped.geoms, key=lambda g: g.length)
                    final_lines.append(np.array(longest_segment.coords))

        if not final_lines:
            logger.warning("No lines remain after trimming")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Step 4: Generate connecting moves between lines
        connecting_lines = []
        for i in range(len(final_lines) - 1):
            prev_end = final_lines[i][-1]
            curr_start = final_lines[i + 1][0]
            connecting_lines.append(np.array([prev_end, curr_start]))
        
        # Step 4.5: Close the boundary by linking last point to initial point
        if final_lines and len(final_lines) > 0:
            last_end = final_lines[-1][-1]
            first_start = final_lines[0][0]
            # Add closing connection
            connecting_lines.append(np.array([last_end, first_start]))

        # Step 5: Generate fly-in and fly-out moves
        fly_lines = []
        if self.config.fly_in_out and final_lines:
            start = final_lines[0][0]
            end = final_lines[-1][-1]
            
            # Calculate fly directions
            if len(final_lines[0]) > 1:
                fly_in_vec = final_lines[0][1] - start
                fly_in_vec = fly_in_vec / np.linalg.norm(fly_in_vec)
                fly_lines.append(np.array([start - fly_in_vec * self.config.fly_distance, start]))
            
            if len(final_lines[-1]) > 1:
                fly_out_vec = end - final_lines[-1][-2]
                fly_out_vec = fly_out_vec / np.linalg.norm(fly_out_vec)
                fly_lines.append(np.array([end, end + fly_out_vec * self.config.fly_distance]))

        # Combine results
        deposition_array = np.vstack(final_lines) if final_lines else np.array([])
        connecting_array = np.vstack(connecting_lines) if connecting_lines else np.array([])
        fly_lines_array = np.vstack(fly_lines) if fly_lines else np.array([])
        fillets_array = np.array([])

        return deposition_array, connecting_array, fly_lines_array, fillets_array
        
    def densify_boundary(self, points, step=1.0):
        """Densify boundary points by adding intermediate points."""
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

    def fit_largest_circles(self, polygon, boundary_points, num_circles=3, resolution=0.5, tolerance=0.1, debug=False):
        """Fit circles inside the polygon that touch exactly two well-separated points on the boundary."""
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

                    # Find candidate boundary points close to the expected radius
                    dists = np.linalg.norm(boundary_points - np.array([x, y]), axis=1)
                    indices_close = np.where(np.abs(dists - distance_to_boundary) < tolerance)[0]
                    candidate_points = boundary_points[indices_close]

                    found_pair = False
                    for i in range(len(candidate_points)):
                        for j in range(i + 1, len(candidate_points)):
                            p1 = candidate_points[i]
                            p2 = candidate_points[j]

                            # Arc distance filter
                            index_diff = abs(indices_close[i] - indices_close[j])
                            if index_diff < 10:
                                continue

                            # Check if points are nearly opposite relative to center
                            v1 = p1 - np.array([x, y])
                            v2 = p2 - np.array([x, y])
                            v1 /= np.linalg.norm(v1)
                            v2 /= np.linalg.norm(v2)
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

                            if angle < np.deg2rad(150):
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
                    if len(circles) == 1:
                        ax.plot(circle_x, circle_y, 'ko', label='Circle Centers')
                        ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx', label='Boundary Points')
                    else:
                        ax.plot(circle_x, circle_y, 'ko')
                        ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx')

        if debug:
            ax.set_aspect('equal')
            ax.set_title("Fitted Circles", fontsize=20)
            ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend(loc="upper left", fontsize=12)
            ax.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()

        return circles, circles_with_boundary_points

    def compute_curvature_influence(self, fitted_circles, num_samples=500):
        """Compute curvature influence from fitted circles."""
        # Extract and sort centers by Y (ascending)
        centers = np.array([[c[0], c[1]] for c in fitted_circles])
        centers = centers[np.argsort(centers[:, 1])]  # Sort by Y axis (bottom to top)
        
        x, y = centers[:, 0], centers[:, 1]

        # Ensure enough points for spline degree
        if len(x) < 4:
            # Duplicate points if needed
            while len(x) < 4:
                x = np.append(x, x[-1])
                y = np.append(y, y[-1])

        # Fit spline
        try:
            from scipy.interpolate import UnivariateSpline
            spline_x = UnivariateSpline(np.arange(len(x)), x, s=self.config.spline_smoothing, k=min(3, len(x)-1))
            spline_y = UnivariateSpline(np.arange(len(y)), y, s=self.config.spline_smoothing, k=min(3, len(y)-1))
            
            # Sample spline
            t_vals = np.linspace(0, len(x)-1, num_samples)
            x_spline = spline_x(t_vals)
            y_spline = spline_y(t_vals)
            
            # Compute curvature
            dx = np.gradient(x_spline)
            dy = np.gradient(y_spline)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            curvature = np.nan_to_num(curvature)
            
            return centers, curvature, x_spline, y_spline
            
        except Exception as e:
            logger.warning(f"Spline fitting failed: {e}")
            return centers, np.zeros(len(centers)), x, y

    def filter_centers_by_curvature(self, centers, curvature_values, z_thresh=2.0):
        """Filter centers based on curvature threshold."""
        if len(curvature_values) == 0:
            return centers, curvature_values
            
        mean_curvature = np.mean(curvature_values)
        std_curvature = np.std(curvature_values)
        threshold = mean_curvature + z_thresh * std_curvature
        
        valid_indices = curvature_values <= threshold
        filtered_centers = centers[valid_indices]
        filtered_curvatures = curvature_values[valid_indices]
        
        return filtered_centers, filtered_curvatures

    def refit_spline_from_filtered_centers(self, filtered_centers):
        """Refit spline from filtered centers."""
        if len(filtered_centers) < 2:
            return np.array([]), np.array([])
            
        x, y = filtered_centers[:, 0], filtered_centers[:, 1]
        
        try:
            from scipy.interpolate import UnivariateSpline
            spline_x = UnivariateSpline(np.arange(len(x)), x, s=self.config.spline_smoothing, k=min(3, len(x)-1))
            spline_y = UnivariateSpline(np.arange(len(y)), y, s=self.config.spline_smoothing, k=min(3, len(y)-1))
            
            t_vals = np.linspace(0, len(x)-1, 100)
            x_spline = spline_x(t_vals)
            y_spline = spline_y(t_vals)
            
            return x_spline, y_spline
            
        except Exception as e:
            logger.warning(f"Spline refitting failed: {e}")
            return x, y

    def fit_left_right_splines_from_circle_contacts(self, filtered_circles):
        """Fit left and right splines from circle contact points."""
        if not filtered_circles:
            return None, None, np.array([]), np.array([])
            
        left_points = []
        right_points = []
        
        for circle in filtered_circles:
            center_x, center_y, radius, contact_points = circle
            if len(contact_points) >= 2:
                # Sort contact points by x-coordinate
                sorted_points = contact_points[np.argsort(contact_points[:, 0])]
                left_points.append(sorted_points[0])
                right_points.append(sorted_points[-1])
        
        if len(left_points) < 2 or len(right_points) < 2:
            return None, None, np.array(left_points), np.array(right_points)
            
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        try:
            from scipy.interpolate import UnivariateSpline
            
            # Fit left spline
            left_sorted = left_points[np.argsort(left_points[:, 1])]
            left_spline_x = UnivariateSpline(left_sorted[:, 1], left_sorted[:, 0], s=self.config.spline_smoothing)
            left_spline_y = UnivariateSpline(np.arange(len(left_sorted)), left_sorted[:, 1], s=self.config.spline_smoothing)
            
            # Fit right spline
            right_sorted = right_points[np.argsort(right_points[:, 1])]
            right_spline_x = UnivariateSpline(right_sorted[:, 1], right_sorted[:, 0], s=self.config.spline_smoothing)
            right_spline_y = UnivariateSpline(np.arange(len(right_sorted)), right_sorted[:, 1], s=self.config.spline_smoothing)
            
            return (left_spline_x, left_spline_y), (right_spline_x, right_spline_y), left_points, right_points
            
        except Exception as e:
            logger.warning(f"Left/right spline fitting failed: {e}")
            return None, None, left_points, right_points

    def extend_spline(self, spline):
        """Extend spline beyond its original range."""
        if spline is None:
            return np.array([]), np.array([])
            
        spline_x, spline_y = spline
        
        try:
            # Get original range
            original_range = np.linspace(0, 1, 50)
            
            # Extend range
            extended_range = np.linspace(-0.2, 1.2, 70)
            
            x_extended = spline_x(extended_range)
            y_extended = spline_y(extended_range)
            
            return x_extended, y_extended
            
        except Exception as e:
            logger.warning(f"Spline extension failed: {e}")
            return np.array([]), np.array([])

    def compute_spline_intersection(self, x_left, y_left, x_right, y_right):
        """Compute intersection point of left and right splines."""
        if len(x_left) == 0 or len(x_right) == 0:
            return None, None
            
        try:
            # Find approximate intersection by minimizing distance
            min_distance = float('inf')
            best_intersection = None
            
            for i, (xl, yl) in enumerate(zip(x_left, y_left)):
                for j, (xr, yr) in enumerate(zip(x_right, y_right)):
                    distance = np.sqrt((xl - xr)**2 + (yl - yr)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_intersection = ((xl + xr) / 2, (yl + yr) / 2)
            
            if best_intersection and min_distance < 5.0:  # Reasonable threshold
                return best_intersection[0], best_intersection[1]
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Spline intersection computation failed: {e}")
            return None, None

    def rebuild_boundary_with_spline_top(self, boundary_points, best_circle_x, best_circle_y, 
                                       x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection):
        """Rebuild boundary with spline top section."""
        try:
            # This is a simplified version - you may need to adapt based on your specific geometry
            if intersection is None or len(x_left_ext) == 0 or len(x_right_ext) == 0:
                return boundary_points
                
            # Create new boundary by combining original bottom with spline top
            bottom_boundary = boundary_points[boundary_points[:, 1] < best_circle_y]
            
            # Create top section from splines
            left_spline_points = np.column_stack([x_left_ext, y_left_ext])
            right_spline_points = np.column_stack([x_right_ext, y_right_ext])
            
            # Combine all points
            new_boundary = np.vstack([
                bottom_boundary,
                left_spline_points,
                [intersection],
                right_spline_points[::-1]  # Reverse right spline
            ])
            
            return new_boundary
            
        except Exception as e:
            logger.warning(f"Boundary rebuilding failed: {e}")
            return boundary_points

    def sort_points_counterclockwise(self, points):
        """Sort points in counterclockwise order."""
        try:
            # Find centroid
            centroid = np.mean(points, axis=0)
            
            # Calculate angles from centroid
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            return points[sorted_indices]
            
        except Exception as e:
            logger.warning(f"Point sorting failed: {e}")
            return points

    def plot_filtered_result(self, boundary_points, fitted_circles, centers, curvature_values, 
                           filtered_centers, filtered_curvatures, x_spline, y_spline):
        """Plot filtered curvature result."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot boundary
        x_b, y_b = boundary_points[:, 0], boundary_points[:, 1]
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

        # Colorbar with larger font
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Curvature Deviation', fontsize=18)
        cbar.ax.tick_params(labelsize=12)

        # Formatting
        ax.set_aspect('equal')
        ax.set_title("Filtered Centers Based on Curvature Deviation", fontsize=20)
        ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
        ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def plot_splines_with_points(self, boundary_points, left_spline, right_spline, left_points, right_points):
        """Plot left and right splines with contact points."""
        plt.figure(figsize=(8, 8))
        
        # Plot boundary and splines
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-')
        
        if left_spline is not None:
            plt.plot(left_spline[0], left_spline[1], 'g-', label='Left Spline')
        if right_spline is not None:
            plt.plot(right_spline[0], right_spline[1], 'r-', label='Right Spline')
            
        plt.plot(left_points[:, 0], left_points[:, 1], 'gx', label='Left Points')
        plt.plot(right_points[:, 0], right_points[:, 1], 'rx', label='Right Points')

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
        ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
        ax.set_title("Splines from Circle Contacts", fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True)
        plt.show()

    def plot_splines_with_points_2(self, boundary_points, left_spline, right_spline, left_points, right_points,
                                  x_spline, y_spline, x_left_ext, y_left_ext, x_right_ext, y_right_ext, 
                                  intersection_x, intersection_y):
        """Plot extended splines with intersection point."""
        plt.figure(figsize=(8, 8))
        
        # Plot the boundary
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-')

        # Plot the extended splines
        plt.plot(x_left_ext, y_left_ext, 'g--', label='Extended Left Spline')
        plt.plot(x_right_ext, y_right_ext, 'r--', label='Extended Right Spline')

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
        plt.show()
        
    def extend_line_to_boundary(self, line, boundary_polygon):
        """Extend a line to reach the boundary if it doesn't already touch it."""
        if isinstance(line, MultiLineString):
            line = max(line.geoms, key=lambda g: g.length)
            
        if not isinstance(line, LineString):
            return line
            
        coords = np.array(line.coords)
        if len(coords) < 2:
            return line
            
        start_point = coords[0]
        end_point = coords[-1]
        
        # Check if line already touches boundary
        if boundary_polygon.touches(line) or boundary_polygon.intersects(line):
            return line
            
        # Extend both ends of the line
        extended_coords = coords.copy()
        
        # Extend start
        if len(coords) > 1:
            start_direction = coords[0] - coords[1]
            start_direction = start_direction / np.linalg.norm(start_direction)
            
            # Find intersection with boundary by extending backwards
            extension_length = 50  # Maximum extension distance
            extended_start = start_point + start_direction * extension_length
            extension_line = LineString([start_point, extended_start])
            
            intersection = boundary_polygon.boundary.intersection(extension_line)
            if not intersection.is_empty:
                if hasattr(intersection, 'coords'):
                    extended_coords = np.vstack([np.array(intersection.coords)[0], extended_coords])
                elif hasattr(intersection, 'geoms'):
                    # Take the closest intersection point
                    closest_point = min(intersection.geoms, 
                                       key=lambda p: np.linalg.norm(np.array(p.coords[0]) - start_point))
                    extended_coords = np.vstack([np.array(closest_point.coords)[0], extended_coords])
        
        # Extend end
        if len(coords) > 1:
            end_direction = coords[-1] - coords[-2]
            end_direction = end_direction / np.linalg.norm(end_direction)
            
            # Find intersection with boundary by extending forwards
            extension_length = 50  # Maximum extension distance
            extended_end = end_point + end_direction * extension_length
            extension_line = LineString([end_point, extended_end])
            
            intersection = boundary_polygon.boundary.intersection(extension_line)
            if not intersection.is_empty:
                if hasattr(intersection, 'coords'):
                    extended_coords = np.vstack([extended_coords, np.array(intersection.coords)[-1]])
                elif hasattr(intersection, 'geoms'):
                    # Take the closest intersection point
                    closest_point = min(intersection.geoms, 
                                       key=lambda p: np.linalg.norm(np.array(p.coords[0]) - end_point))
                    extended_coords = np.vstack([extended_coords, np.array(closest_point.coords)[0]])
        
        return LineString(extended_coords)
        
    def evaluate_toolpaths(self):
        """Evaluate toolpath performance using exact AS4RCSD3 approach."""
        if not self.config.enable_evaluation:
            logger.info("Evaluation disabled, skipping metrics calculation")
            return
            
        logger.info("Evaluating toolpath performance...")
        start_time = time.time()
        
        try:
            # Import evaluation functions
            from evaluation_metrics import (
                calculate_total_length, deposition_to_travel_ratio,
                turn_angle_analysis, centroid_based_distance_per_layer
            )
            
            # Determine debug layer for AS4RCSD3 compatibility
            if self.config.evaluate == "all":
                debug_layer = None  # Process all layers
                logger.info(f"Evaluating all {len(self.toolpaths)} layers")
            else:
                try:
                    debug_layer = int(self.config.evaluate)
                    if 0 <= debug_layer < len(self.toolpaths):
                        logger.info(f"Evaluating single layer: {debug_layer}")
                    else:
                        logger.warning(f"Layer {debug_layer} out of range. Evaluating all layers.")
                        debug_layer = None
                except ValueError:
                    logger.warning(f"Invalid evaluate value '{self.config.evaluate}'. Evaluating all layers.")
                    debug_layer = None
            
            # Calculate basic metrics (AS4RCSD3 style - only deposition paths, no tool moves for hybrid2)
            total_deposition_length = sum(calculate_total_length(deposition) for deposition, _, _, _ in self.toolpaths)
            # For hybrid2, tool moves are minimal (only between layers), so we calculate them differently
            total_travel_length = 0  # Hybrid2 has minimal travel moves
            
            ratio = deposition_to_travel_ratio(total_deposition_length, max(total_travel_length, 1))
            turn_angle = turn_angle_analysis(self.toolpaths)
            CANPD_result, CANPD_avr = centroid_based_distance_per_layer(
                self.toolpaths, self.processed_boundaries, self.config.tool_width, visualize=False, num_samples=360
            )
            
            # Use exact AS4RCSD3 deposition color map calculation
            logger.info("Calculating deposition color maps using AS4RCSD3 method...")
            color_maps, coverage_percentage, lateral_support_percentage, over_deposition_percentage, supporting_contour_percentage, wall_engagement_index = self.calculate_deposition_color_map_as4rcsd3(debug_layer)
            
            # Combine results (AS4RCSD3 format)
            overall_metrics = {
                "Deposition Length": total_deposition_length,
                "Travel Length": total_travel_length,
                "Dep-to-Travel Ratio": ratio,
                "Average Turn Angle": turn_angle,
                "Centroid-Aligned Nearest Point Deviation": CANPD_avr,
                "Coverage %": coverage_percentage,
                "Over-Deposition %": over_deposition_percentage,
                "Supported Deposition %": lateral_support_percentage,
                "Supporting Contour %": supporting_contour_percentage,
                "Wall Engagement Index": wall_engagement_index
            }
            
            # Print results table (exact AS4RCSD3 style)
            print("\nEvaluation Metrics Table:")
            print(f"{'Metric':<30} {'Desired Value':<20} {'Calculated Value':<10}")
            print("=" * 80)
            
            metrics_table = [
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
            
            for metric in metrics_table:
                print(f"{metric[0]:<30} {metric[1]:<20} {metric[2]:<10.4f}")
            
            # Save metrics to text file (AS4RCSD3 style)
            self.save_evaluation_metrics_as4rcsd3(overall_metrics)
            
            elapsed_time = time.time() - start_time
            logger.info(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
            
        except ImportError as e:
            logger.warning(f"Evaluation metrics module not found: {e}")
            logger.warning("Skipping detailed evaluation")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def handle_evaluation_visualization(self, deposition_analysis, layers_evaluated):
        """Handle user requests for evaluation visualization (AS4RCSD3 style)."""
        try:
            # Ask user for deposition color map visualization
            visualize_color_map = input("Do you want to visualize the deposition color map? (yes/no): ").strip().lower()
            
            if visualize_color_map == 'yes':
                self.visualize_color_maps(deposition_analysis, layers_evaluated)
            
            # Ask user for other evaluation plots
            visualize_debug = input("Do you want to visualize debug layer analysis? (yes/no): ").strip().lower()
            
            if visualize_debug == 'yes':
                if len(layers_evaluated) == 1:
                    debug_layer = layers_evaluated[0]
                else:
                    while True:
                        try:
                            debug_layer = int(input(f"Enter layer index to debug (0-{max(layers_evaluated)}): "))
                            if debug_layer in layers_evaluated:
                                break
                            else:
                                print(f"Layer {debug_layer} was not evaluated. Choose from: {layers_evaluated}")
                        except ValueError:
                            print("Please enter a valid number.")
                
                self.visualize_debug_layer(deposition_analysis, debug_layer)
            
            # Ask for deposition flow visualization
            visualize_flow = input("Do you want to visualize deposition flow? (yes/no): ").strip().lower()
            
            if visualize_flow == 'yes':
                if len(layers_evaluated) == 1:
                    flow_layer = layers_evaluated[0]
                else:
                    while True:
                        try:
                            flow_layer = int(input(f"Enter layer index for flow visualization (0-{max(layers_evaluated)}): "))
                            if flow_layer in layers_evaluated:
                                break
                            else:
                                print(f"Layer {flow_layer} was not evaluated. Choose from: {layers_evaluated}")
                        except ValueError:
                            print("Please enter a valid number.")
                
                self.visualize_deposition_flow(flow_layer)
                
        except KeyboardInterrupt:
            print("\nVisualization cancelled by user.")
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
    
    def visualize_color_maps(self, deposition_analysis, layers_evaluated):
        """Visualize deposition color maps for evaluated layers."""
        color_maps = deposition_analysis["color_maps"]
        
        for i, layer_index in enumerate(layers_evaluated):
            if i < len(color_maps):
                plt.figure(figsize=(8, 8))
                plt.imshow(color_maps[i], cmap='plasma', interpolation='nearest')
                plt.colorbar(label='Deposition Count')
                plt.title(f"Layer {layer_index} Deposition Map")
                plt.xlabel("Grid X")
                plt.ylabel("Grid Y")
                plt.show()
    
    def visualize_debug_layer(self, deposition_analysis, debug_layer):
        """Visualize debug information for a specific layer (AS4RCSD3 style)."""
        try:
            # Find the layer in evaluated results
            layer_idx_in_results = None
            for i, layer_data in enumerate(deposition_analysis["per_layer"]):
                if layer_data["layer_index"] == debug_layer:
                    layer_idx_in_results = i
                    break
            
            if layer_idx_in_results is None:
                print(f"Layer {debug_layer} was not evaluated.")
                return
            
            # Get layer data
            layer_height, boundary_points = self.processed_boundaries[debug_layer]
            contour = self.contours[debug_layer] if debug_layer < len(self.contours) else []
            color_map = deposition_analysis["color_maps"][layer_idx_in_results]
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Create boundary polygon for plotting
            boundary_polygon = Polygon(boundary_points)
            min_x, min_y, max_x, max_y = boundary_polygon.bounds
            
            # Plot the rotated and mirrored grid (AS4RCSD3 style)
            mirrored_grid = np.fliplr(color_map)
            rotated_grid = np.rot90(mirrored_grid)
            
            extent = [min_x, max_x, min_y, max_y]
            plt.imshow(rotated_grid, cmap='plasma', extent=extent, origin='lower', alpha=0.8)
            plt.colorbar(label='Deposition Count')
            
            # Plot boundary
            x, y = boundary_polygon.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2, label="Boundary")
            
            # Plot contour if available
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                x_contour, y_contour = zip(*contour_points)
                plt.plot(x_contour, y_contour, 'g-', linewidth=2, label="Contour")
            
            plt.title(f"Layer {debug_layer} Debug Analysis", fontsize=16)
            plt.xlabel("X (mm)", fontsize=14)
            plt.ylabel("Y (mm)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing debug layer {debug_layer}: {e}")
    
    def visualize_deposition_flow(self, layer_index):
        """Visualize deposition flow for a specific layer (AS4RCSD3 style)."""
        try:
            if layer_index >= len(self.toolpaths):
                print(f"Layer {layer_index} out of range.")
                return
            
            # Get layer data
            layer_height, boundary_points = self.processed_boundaries[layer_index]
            deposition, _, _, _ = self.toolpaths[layer_index]
            contour = self.contours[layer_index] if layer_index < len(self.contours) else []
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Plot boundary
            boundary_polygon = Polygon(boundary_points)
            x, y = boundary_polygon.exterior.xy
            plt.plot(x, y, 'k-', linewidth=2, label="Boundary")
            
            # Plot contour
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                x_contour, y_contour = zip(*contour_points)
                plt.plot(x_contour, y_contour, 'g-', linewidth=2, label="Contour")
            
            # Plot deposition path with flow direction
            if len(deposition) > 1:
                deposition_array = np.array(deposition)
                x_dep, y_dep = deposition_array[:, 0], deposition_array[:, 1]
                
                # Plot path
                plt.plot(x_dep, y_dep, 'b-', linewidth=1, alpha=0.7, label="Deposition Path")
                
                # Add arrows to show direction
                for i in range(0, len(deposition) - 1, max(1, len(deposition) // 20)):
                    dx = deposition[i+1][0] - deposition[i][0]
                    dy = deposition[i+1][1] - deposition[i][1]
                    plt.arrow(deposition[i][0], deposition[i][1], dx*0.3, dy*0.3,
                             head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.8)
                
                # Mark start and end
                plt.plot(x_dep[0], y_dep[0], 'go', markersize=8, label="Start")
                plt.plot(x_dep[-1], y_dep[-1], 'ro', markersize=8, label="End")
            
            plt.title(f"Layer {layer_index} Deposition Flow", fontsize=16)
            plt.xlabel("X (mm)", fontsize=14)
            plt.ylabel("Y (mm)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing deposition flow for layer {layer_index}: {e}")
    
    def save_evaluation_plots(self, deposition_analysis, layers_evaluated):
        """Automatically save evaluation plots for analyzed layers."""
        if not self.config.save_evaluation_plots:
            logger.info("Evaluation plot saving disabled")
            return
            
        try:
            output_dir = Path(self.config.output_dir)
            plots_dir = output_dir / "evaluation_plots"
            plots_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving evaluation plots to {plots_dir}")
            
            # Save plots for each evaluated layer
            for i, layer_index in enumerate(layers_evaluated):
                if i < len(deposition_analysis["color_maps"]):
                    # Save deposition color map matrix
                    self.save_deposition_color_map_plot(
                        deposition_analysis, layer_index, i, plots_dir
                    )
                    
                    # Save deposition flow plot
                    self.save_deposition_flow_plot(layer_index, plots_dir)
            
            logger.info(f"Saved {len(layers_evaluated)} sets of evaluation plots")
            
        except Exception as e:
            logger.error(f"Error saving evaluation plots: {e}")
    
    def save_deposition_color_map_plot(self, deposition_analysis, layer_index, result_index, plots_dir):
        """Save deposition color map matrix plot for a specific layer."""
        try:
            # Get layer data
            layer_height, boundary_points = self.processed_boundaries[layer_index]
            contour = self.contours[layer_index] if layer_index < len(self.contours) else []
            color_map = deposition_analysis["color_maps"][result_index]
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create boundary polygon for plotting
            boundary_polygon = Polygon(boundary_points)
            min_x, min_y, max_x, max_y = boundary_polygon.bounds
            
            # Plot the rotated and mirrored grid (AS4RCSD3 style)
            mirrored_grid = np.fliplr(color_map)
            rotated_grid = np.rot90(mirrored_grid)
            
            extent = [min_x, max_x, min_y, max_y]
            plt.imshow(rotated_grid, cmap='plasma', extent=extent, origin='lower', alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(label='Deposition Count')
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Deposition Count', fontsize=14)
            
            # Plot boundary
            x, y = boundary_polygon.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2, label="Boundary")
            
            # Plot contour if available
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                x_contour, y_contour = zip(*contour_points)
                plt.plot(x_contour, y_contour, 'g-', linewidth=2, label="Contour")
            
            # Get layer metrics for title
            layer_metrics = deposition_analysis["per_layer"][result_index]
            
            plt.title(f"Layer {layer_index} Deposition Color Map\n"
                     f"Coverage: {layer_metrics['coverage_percentage']:.1f}%, "
                     f"Over-Dep: {layer_metrics['over_deposition_percentage']:.1f}%, "
                     f"WEI: {layer_metrics['wall_engagement_index']:.3f}", 
                     fontsize=16)
            plt.xlabel("X (mm)", fontsize=14)
            plt.ylabel("Y (mm)", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            
            # Save plot
            plot_filename = plots_dir / f"layer_{layer_index:03d}_deposition_colormap.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved deposition color map: {plot_filename}")
            
        except Exception as e:
            logger.error(f"Error saving deposition color map for layer {layer_index}: {e}")
    
    def save_deposition_flow_plot(self, layer_index, plots_dir):
        """Save deposition flow plot for a specific layer."""
        try:
            if layer_index >= len(self.toolpaths):
                logger.warning(f"Layer {layer_index} out of range for flow plot")
                return
            
            # Get layer data
            layer_height, boundary_points = self.processed_boundaries[layer_index]
            deposition, _, _, _ = self.toolpaths[layer_index]
            contour = self.contours[layer_index] if layer_index < len(self.contours) else []
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot boundary
            boundary_polygon = Polygon(boundary_points)
            x, y = boundary_polygon.exterior.xy
            plt.plot(x, y, 'k-', linewidth=2, label="Boundary")
            
            # Plot contour
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                x_contour, y_contour = zip(*contour_points)
                plt.plot(x_contour, y_contour, 'g-', linewidth=2, label="Contour")
            
            # Plot deposition path with color fade and flow direction
            if len(deposition) > 1:
                deposition_array = np.array(deposition)
                x_dep, y_dep = deposition_array[:, 0], deposition_array[:, 1]
                
                # Create color fade from start (blue) to end (red)
                colors = plt.cm.coolwarm(np.linspace(0, 1, len(deposition)))
                
                # Plot path segments with color fade
                for i in range(len(deposition) - 1):
                    plt.plot([x_dep[i], x_dep[i+1]], [y_dep[i], y_dep[i+1]], 
                            color=colors[i], linewidth=2, alpha=0.8)
                
                # Add arrows to show direction (every 10th point or so)
                arrow_step = max(1, len(deposition) // 15)
                for i in range(0, len(deposition) - 1, arrow_step):
                    dx = deposition[i+1][0] - deposition[i][0]
                    dy = deposition[i+1][1] - deposition[i][1]
                    if np.sqrt(dx*dx + dy*dy) > 0.1:  # Only add arrow if segment is long enough
                        plt.arrow(deposition[i][0], deposition[i][1], dx*0.3, dy*0.3,
                                 head_width=0.5, head_length=0.3, fc='orange', ec='orange', 
                                 alpha=0.7, linewidth=1)
                
                # Mark start and end points
                plt.plot(x_dep[0], y_dep[0], 'go', markersize=10, label="Start", markeredgecolor='black')
                plt.plot(x_dep[-1], y_dep[-1], 'ro', markersize=10, label="End", markeredgecolor='black')
                
                # Add color bar for path progression
                sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, label='Path Progress (Start → End)')
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label('Path Progress (Start → End)', fontsize=14)
            
            plt.title(f"Layer {layer_index} Deposition Flow\n"
                     f"Path Length: {len(deposition)} points", fontsize=16)
            plt.xlabel("X (mm)", fontsize=14)
            plt.ylabel("Y (mm)", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            
            # Save plot
            plot_filename = plots_dir / f"layer_{layer_index:03d}_deposition_flow.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved deposition flow plot: {plot_filename}")
            
        except Exception as e:
            logger.error(f"Error saving deposition flow plot for layer {layer_index}: {e}")
    
    def visualize_color_map_as4rcsd3(self, color_maps):
        """Visualize color maps AS4RCSD3 style."""
        for layer_index, grid in enumerate(color_maps):
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap='plasma', interpolation='nearest')  # Plasma highlights intensity better
            plt.colorbar(label='Deposition Count')
            plt.title(f"Layer {layer_index} Deposition Map")
            plt.show()
    
    def save_as4rcsd3_debug_layer_plot(self, grid, layer_index, boundary_points, contour, min_x, min_y, max_x, max_y):
        """Save debug layer plot AS4RCSD3 style."""
        try:
            output_dir = Path(self.config.output_dir)
            plots_dir = output_dir / "evaluation_plots"
            plots_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(8, 8))

            # Mirror the grid horizontally (left-right)
            mirrored_grid = np.fliplr(grid)
            # Rotate the grid 90 degrees counterclockwise (CCW)
            rotated_grid = np.rot90(mirrored_grid)
            
            # Define the real-world extent in mm
            extent = [min_x, max_x, min_y, max_y]
            plt.imshow(rotated_grid, cmap='plasma', extent=extent, origin='lower', alpha=0.8)
            
            # Add colorbar with consistent formatting
            cbar = plt.colorbar(label='Deposition Count')
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Deposition Count', fontsize=14)

            # Plot boundary
            boundary_polygon = Polygon(boundary_points)
            x, y = boundary_polygon.exterior.xy
            plt.plot(x, y, 'r-', label="Boundary")

            # Plot contour
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                x, y = zip(*contour_points)
                plt.plot(x, y, 'g-', label="Contour")

            # Apply consistent formatting
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
            ax.set_title(f"Layer {layer_index} Debug Analysis", fontsize=20)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True)
            plt.tight_layout()
            
            # Save plot
            plot_filename = plots_dir / f"layer_{layer_index:03d}_debug_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error saving AS4RCSD3 debug plot for layer {layer_index}: {e}")
    
    def save_as4rcsd3_deposition_flow_plot(self, deposition, layer_index, boundary_points, contour):
        """Save deposition flow plot AS4RCSD3 style."""
        try:
            output_dir = Path(self.config.output_dir)
            plots_dir = output_dir / "evaluation_plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Extract boundary
            boundary_polygon = Polygon(boundary_points)
            min_x, min_y, max_x, max_y = boundary_polygon.bounds

            # Plot setup
            plt.figure(figsize=(8, 8))

            # Plot contour
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                x, y = zip(*list(contour_line.coords))
                plt.plot(x, y, 'g--', label="Contour")

            # Convert deposition path into segments
            points = np.array(deposition)
            if len(points) >= 2:
                segments = np.array([points[i:i+2] for i in range(len(points) - 1)])
                cmap_values = np.linspace(0, 1, len(segments))

                # Create color-mapped LineCollection
                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, cmap='viridis', array=cmap_values, linewidth=2)
                plt.gca().add_collection(lc)

                # Plot start and end points for clarity
                plt.plot(*points[0], 'ko', label='Start')
                plt.plot(*points[-1], 'ks', label='End')

                # Add colorbar with consistent formatting
                cbar = plt.colorbar(lc, label='Deposition Flow (start → end)')
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label('Deposition Flow (start → end)', fontsize=14)

            # Apply consistent formatting
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
            ax.set_title(f"Deposition Flow Visualization - Layer {layer_index}", fontsize=20)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True)
            plt.tight_layout()
            
            # Save plot
            plot_filename = plots_dir / f"layer_{layer_index:03d}_deposition_flow.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error saving AS4RCSD3 flow plot for layer {layer_index}: {e}")
    
    def save_evaluation_metrics_as4rcsd3(self, overall_metrics):
        """Save evaluation metrics in AS4RCSD3 format."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Save metrics to a text file (AS4RCSD3 style)
            output_file = output_dir / "evaluation_metrics.txt"

            with open(output_file, "w") as f:
                f.write("Evaluation Metrics Table:\n")
                f.write(f"{'Metric':<30} {'Desired Value':<20} {'Calculated Value':<10}\n")
                f.write("=" * 80 + "\n")
                
                metrics_table = [
                    ("Deposition Length", "None", overall_metrics["Deposition Length"]),
                    ("Travel Length", "Lower", overall_metrics["Travel Length"]),
                    ("Dep-to-Travel Ratio", "Higher", overall_metrics["Dep-to-Travel Ratio"]),
                    ("Average Turn Angle", "Lower", overall_metrics["Average Turn Angle"]),
                    ("Centroid-Aligned Nearest Point Deviation", "Lower", overall_metrics["Centroid-Aligned Nearest Point Deviation"]),
                    ("Coverage %", "Higher", overall_metrics["Coverage %"]),
                    ("Over-Deposition %", "Lower", overall_metrics["Over-Deposition %"]),
                    ("Supported Deposition %", "Higher", overall_metrics["Supported Deposition %"]),
                    ("Supporting Contour %", "Higher", overall_metrics["Supporting Contour %"])
                ]
                
                for metric in metrics_table:
                    f.write(f"{metric[0]:<30} {metric[1]:<20} {metric[2]:<10.4f}\n")

            print(f"\nMetrics saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving AS4RCSD3 metrics: {e}")
            
    def save_metrics(self, metrics: Dict[str, float]):
        """Save evaluation metrics to file."""
        output_file = Path(self.config.output_dir) / "evaluation_metrics.json"
        
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {output_file}")
            
    def calculate_comprehensive_deposition_metrics(self):
        """Calculate comprehensive deposition metrics using efficient grid-based analysis."""
        logger.info("Creating high-resolution deposition maps...")
        
        # Determine overall bounds for all layers
        all_boundaries = [boundary for _, boundary in self.processed_boundaries]
        all_points = np.vstack(all_boundaries)
        
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        
        # Add padding
        padding = max(self.config.tool_width * 2, 5.0)
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Create high-resolution grid
        grid_resolution = self.config.tool_width / 4  # 4 pixels per tool width
        x_points = int((max_x - min_x) / grid_resolution) + 1
        y_points = int((max_y - min_y) / grid_resolution) + 1
        
        logger.info(f"Grid size: {x_points} x {y_points} ({x_points * y_points:,} points)")
        
        # Create coordinate grids
        x_grid = np.linspace(min_x, max_x, x_points)
        y_grid = np.linspace(min_y, max_y, y_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize results
        layer_metrics = []
        overall_deposition_map = np.zeros((y_points, x_points))
        overall_intended_map = np.zeros((y_points, x_points))
        
        # Process each layer
        for layer_idx, ((layer_height, boundary_points), (deposition, tool_moves, fly_lines, fillets)) in enumerate(
            zip(self.processed_boundaries, self.toolpaths)
        ):
            logger.debug(f"Processing deposition map for layer {layer_idx}")
            
            # Create intended area map (boundary)
            intended_map = self.create_boundary_map(boundary_points, X, Y, grid_resolution)
            
            # Create actual deposition map (toolpath + tool width)
            deposition_map = self.create_deposition_map(deposition, X, Y, grid_resolution)
            
            # Calculate layer metrics
            layer_metric = self.calculate_layer_deposition_metrics(
                intended_map, deposition_map, boundary_points, deposition, layer_idx
            )
            layer_metrics.append(layer_metric)
            
            # Accumulate for overall metrics
            overall_deposition_map += deposition_map
            overall_intended_map += intended_map
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_deposition_metrics(
            overall_intended_map, overall_deposition_map
        )
        
        # Calculate layer statistics
        layer_averages = self.calculate_layer_statistics(layer_metrics)
        
        return {
            "overall": overall_metrics,
            "per_layer": layer_metrics,
            "layer_averages": layer_averages,
            "grid_info": {
                "resolution": grid_resolution,
                "size": (x_points, y_points),
                "bounds": (min_x, min_y, max_x, max_y)
            }
        }
    
    def create_boundary_map(self, boundary_points, X, Y, resolution):
        """Create a binary map of the intended boundary area."""
        from shapely.geometry import Polygon, Point
        
        # Create polygon from boundary
        boundary_polygon = Polygon(boundary_points)
        
        # Create binary map
        boundary_map = np.zeros(X.shape, dtype=bool)
        
        # Vectorized point-in-polygon test
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Use shapely's vectorized contains for efficiency
        try:
            from shapely.vectorized import contains
            inside = contains(boundary_polygon, points[:, 0], points[:, 1])
            boundary_map = inside.reshape(X.shape)
        except ImportError:
            # Fallback to individual point tests (slower)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = Point(X[i, j], Y[i, j])
                    boundary_map[i, j] = boundary_polygon.contains(point)
        
        return boundary_map.astype(float)
    
    def create_deposition_map(self, deposition_path, X, Y, resolution):
        """Create a deposition map based on toolpath and tool width."""
        if len(deposition_path) < 2:
            return np.zeros(X.shape)
        
        deposition_map = np.zeros(X.shape)
        tool_radius = self.config.tool_width / 2
        
        # For each point in the grid, check if it's within tool radius of any deposition path
        for i in range(len(deposition_path) - 1):
            start_point = deposition_path[i]
            end_point = deposition_path[i + 1]
            
            # Calculate distance from each grid point to this line segment
            distances = self.distance_to_line_segment(X, Y, start_point, end_point)
            
            # Add deposition where distance <= tool radius
            deposition_map += (distances <= tool_radius).astype(float)
        
        # Normalize (clip to 1.0 for areas with multiple passes)
        return np.clip(deposition_map, 0, 1)
    
    def distance_to_line_segment(self, X, Y, start_point, end_point):
        """Calculate distance from grid points to line segment."""
        # Vector from start to end
        line_vec = end_point - start_point
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq < 1e-10:  # Degenerate line segment
            # Distance to point
            return np.sqrt((X - start_point[0])**2 + (Y - start_point[1])**2)
        
        # Parameter t for closest point on line
        t = np.maximum(0, np.minimum(1, 
            ((X - start_point[0]) * line_vec[0] + (Y - start_point[1]) * line_vec[1]) / line_length_sq
        ))
        
        # Closest point on line segment
        closest_x = start_point[0] + t * line_vec[0]
        closest_y = start_point[1] + t * line_vec[1]
        
        # Distance to closest point
        return np.sqrt((X - closest_x)**2 + (Y - closest_y)**2)
    
    def calculate_layer_deposition_metrics(self, intended_map, deposition_map, boundary_points, deposition_path, layer_idx):
        """Calculate deposition metrics for a single layer."""
        # Basic coverage metrics
        intended_area = np.sum(intended_map)
        deposited_area = np.sum(deposition_map)
        covered_area = np.sum(intended_map * deposition_map)
        
        # Coverage percentage
        coverage_percentage = (covered_area / intended_area * 100) if intended_area > 0 else 0
        
        # Over-deposition (deposition outside intended area)
        over_deposition_area = np.sum(deposition_map * (1 - intended_map))
        over_deposition_percentage = (over_deposition_area / intended_area * 100) if intended_area > 0 else 0
        
        # Under-deposition (intended area not covered)
        under_deposition_area = intended_area - covered_area
        under_deposition_percentage = (under_deposition_area / intended_area * 100) if intended_area > 0 else 0
        
        # Material efficiency
        material_efficiency = (covered_area / deposited_area * 100) if deposited_area > 0 else 0
        
        # Wall engagement (simplified - based on boundary proximity)
        wall_engagement = self.calculate_wall_engagement(deposition_path, boundary_points)
        
        return {
            "layer_index": layer_idx,
            "coverage_percentage": coverage_percentage,
            "over_deposition_percentage": over_deposition_percentage,
            "under_deposition_percentage": under_deposition_percentage,
            "material_efficiency": material_efficiency,
            "wall_engagement_index": wall_engagement,
            "intended_area": intended_area,
            "deposited_area": deposited_area,
            "covered_area": covered_area
        }
    
    def calculate_overall_deposition_metrics(self, overall_intended_map, overall_deposition_map):
        """Calculate overall deposition metrics across all layers."""
        # Normalize maps (each layer contributes equally)
        num_layers = len(self.toolpaths)
        avg_intended_map = overall_intended_map / num_layers
        avg_deposition_map = overall_deposition_map / num_layers
        
        # Calculate metrics
        intended_area = np.sum(avg_intended_map)
        deposited_area = np.sum(avg_deposition_map)
        covered_area = np.sum(avg_intended_map * (avg_deposition_map > 0))
        
        coverage_percentage = (covered_area / intended_area * 100) if intended_area > 0 else 0
        over_deposition_area = np.sum((avg_deposition_map > 0) * (avg_intended_map == 0))
        over_deposition_percentage = (over_deposition_area / intended_area * 100) if intended_area > 0 else 0
        under_deposition_percentage = 100 - coverage_percentage
        material_efficiency = (covered_area / deposited_area * 100) if deposited_area > 0 else 0
        
        # Overall wall engagement (average of all layers)
        wall_engagements = []
        for (_, boundary_points), (deposition, _, _, _) in zip(self.processed_boundaries, self.toolpaths):
            wall_engagements.append(self.calculate_wall_engagement(deposition, boundary_points))
        
        avg_wall_engagement = np.mean(wall_engagements) if wall_engagements else 0
        
        return {
            "coverage_percentage": coverage_percentage,
            "over_deposition_percentage": over_deposition_percentage,
            "under_deposition_percentage": under_deposition_percentage,
            "material_efficiency": material_efficiency,
            "wall_engagement_index": avg_wall_engagement
        }
    
    def calculate_wall_engagement(self, deposition_path, boundary_points):
        """Calculate wall engagement index based on proximity to boundary."""
        if len(deposition_path) < 2 or len(boundary_points) < 3:
            return 0.0
        
        from shapely.geometry import Polygon, LineString
        
        try:
            boundary_polygon = Polygon(boundary_points)
            deposition_line = LineString(deposition_path)
            
            # Calculate how much of the deposition is close to the boundary
            boundary_buffer = boundary_polygon.boundary.buffer(self.config.tool_width)
            engaged_length = deposition_line.intersection(boundary_buffer).length
            total_length = deposition_line.length
            
            return (engaged_length / total_length) if total_length > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def calculate_layer_statistics(self, layer_metrics):
        """Calculate statistics across all layers."""
        if not layer_metrics:
            return {
                "avg_coverage": 0, "std_coverage": 0,
                "excellent_layers": 0, "poor_layers": 0
            }
        
        coverages = [m["coverage_percentage"] for m in layer_metrics]
        
        return {
            "avg_coverage": np.mean(coverages),
            "std_coverage": np.std(coverages),
            "excellent_layers": sum(1 for c in coverages if c > 95),
            "poor_layers": sum(1 for c in coverages if c < 80)
        }
        
    def save_comprehensive_metrics(self, overall_metrics, detailed_metrics):
        """Save comprehensive metrics to JSON files."""
        output_dir = Path(self.config.output_dir)
        
        # Save overall metrics (compatible with existing format)
        overall_file = output_dir / "evaluation_metrics.json"
        with open(overall_file, "w") as f:
            json.dump(overall_metrics, f, indent=2)
        logger.info(f"Overall metrics saved to {overall_file}")
        
        # Save detailed metrics (new comprehensive format)
        detailed_file = output_dir / "detailed_evaluation_metrics.json"
        with open(detailed_file, "w") as f:
            json.dump(detailed_metrics, f, indent=2)
        logger.info(f"Detailed metrics saved to {detailed_file}")
        
    def calculate_deposition_color_map_as4rcsd3(self, debug_layer=None):
        """Calculate deposition color map using exact AS4RCSD3 implementation."""
        color_maps = []
        over_deposition_metrics = []
        
        # Use the exact AS4RCSD3 approach
        for layer_index, ((layer_height, boundary_points), (deposition, _, _, _), contour) in enumerate(
            zip(self.processed_boundaries, self.toolpaths, self.contours)
        ):
            if debug_layer is not None and layer_index != debug_layer:
                continue
            
            boundary_polygon = Polygon(boundary_points)
            grid_size = self.config.grid_size
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

            # Process Contour (Wall Zone) First - exact AS4RCSD3 approach
            if len(contour) > 1:
                ordered_points = sorted(contour, key=lambda x: x[0])
                contour_line = LineString(ordered_points)
                contour_points = list(contour_line.coords)
                
                try:
                    tolerance = self.config.tool_width / 2

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

            # Rasterize deposition paths - exact AS4RCSD3 approach
            for i in range(len(deposition) - 1):
                try:
                    point1 = deposition[i]
                    point2 = deposition[i + 1]
                    line = LineString([point1, point2])

                    # Limit to the boundary area
                    deposition_area = line.buffer(self.config.tool_width / 2, cap_style=2).intersection(boundary_polygon)
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

            # Compute Wall Engagement Index (WEI) - exact AS4RCSD3 approach
            support_time_grid = np.zeros_like(grid)  # Stores when each contour pixel was first supported

            for segment_index in range(len(deposition) - 1):
                try:
                    point1 = deposition[segment_index]
                    point2 = deposition[segment_index + 1]
                    line = LineString([point1, point2])

                    deposition_area = line.buffer(self.config.tool_width / 2, cap_style=2).intersection(boundary_polygon)
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

            # Calculate deposition metrics - exact AS4RCSD3 approach
            deposition_only_grid = grid - contour_grid
            deposition_only_grid[deposition_only_grid < 0] = 0  # Ensure no negative values

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

            # Compute contour support effectiveness
            contour_pixels = (contour_grid == 1)
            supporting_contour_pixels = (contour_pixels & (grid > 1))
            total_contour_pixels = np.sum(contour_pixels)
            supporting_contour_pixel_count = np.sum(supporting_contour_pixels)

            supporting_contour_percentage = (supporting_contour_pixel_count / total_contour_pixels * 100
                                            if total_contour_pixels > 0 else 0)

            # Print AS4RCSD3 style output
            print(f"Layer {layer_index}: Max Deposition Overlap = {grid.max()}, Coverage = {coverage_percentage:.2f}%, "
                  f"Support Deposition = {lateral_support_percentage:.2f}%, Over-Deposition = {over_deposition_percentage:.2f}%, "
                  f"Zero Deposition = {zero_deposition_cells}, Supporting Contour = {supporting_contour_percentage:.2f}%, "
                  f"Wall Engagement Index = {wall_engagement_index:.2f}")

            color_maps.append(grid)
            over_deposition_metrics.append(over_deposition_percentage)

            # Save individual layer plots if enabled
            if self.config.save_evaluation_plots:
                self.save_as4rcsd3_debug_layer_plot(grid, layer_index, boundary_points, contour, min_x, min_y, max_x, max_y)
                self.save_as4rcsd3_deposition_flow_plot(deposition, layer_index, boundary_points, contour)

        # Visualize color maps (AS4RCSD3 style)
        if debug_layer is None:  # Only show if processing all layers
            self.visualize_color_map_as4rcsd3(color_maps)

        # Return the last layer's metrics (AS4RCSD3 behavior)
        return color_maps, coverage_percentage, lateral_support_percentage, over_deposition_percentage, supporting_contour_percentage, wall_engagement_index
    

    

        
        # Save text file (AS4RCSD3 style)
        text_file = output_dir / "evaluation_metrics.txt"
        with open(text_file, "w") as f:
            f.write("Evaluation Metrics Table:\n")
            f.write(f"{'Metric':<40} {'Desired Value':<15} {'Calculated Value':<15}\n")
            f.write("=" * 80 + "\n")
            
            metrics_table = [
                ("Deposition Length", "None", overall_metrics["Deposition Length"]),
                ("Travel Length", "Lower", overall_metrics["Travel Length"]),
                ("Dep-to-Travel Ratio", "Higher", overall_metrics["Dep-to-Travel Ratio"]),
                ("Average Turn Angle", "Lower", overall_metrics["Average Turn Angle"]),
                ("Centroid-Aligned Nearest Point Deviation", "Lower", overall_metrics["Centroid-Aligned Nearest Point Deviation"]),
                ("Coverage %", "Higher", overall_metrics["Coverage %"]),
                ("Over-Deposition %", "Lower", overall_metrics["Over-Deposition %"]),
                ("Supported Deposition %", "Higher", overall_metrics["Supported Deposition %"]),
                ("Supporting Contour %", "Higher", overall_metrics["Supporting Contour %"]),
                ("Wall Engagement Index", "Higher", overall_metrics["Wall Engagement Index"])
            ]
            
            for metric in metrics_table:
                f.write(f"{metric[0]:<40} {metric[1]:<15} {metric[2]:<15.4f}\n")
        
        logger.info(f"Metrics table saved to {text_file}")
        
        # Save detailed per-layer metrics
        detailed_file = output_dir / "detailed_evaluation_metrics.json"
        detailed_data = {
            "overall_metrics": overall_metrics,
            "per_layer_metrics": deposition_analysis["per_layer"],
            "summary_statistics": {
                "total_layers": len(deposition_analysis["per_layer"]),
                "average_coverage": np.mean([m["coverage_percentage"] for m in deposition_analysis["per_layer"]]),
                "coverage_std": np.std([m["coverage_percentage"] for m in deposition_analysis["per_layer"]]),
                "excellent_layers": sum(1 for m in deposition_analysis["per_layer"] if m["coverage_percentage"] > 95),
                "poor_layers": sum(1 for m in deposition_analysis["per_layer"] if m["coverage_percentage"] < 80)
            }
        }
        
        with open(detailed_file, "w") as f:
            json.dump(detailed_data, f, indent=2)
        logger.info(f"Detailed per-layer metrics saved to {detailed_file}")
        
    def visualize_deposition_color_map_interactive(self):
        """Interactive deposition color map visualization with user layer selection."""
        if not hasattr(self, 'deposition_analysis_cache'):
            logger.info("Calculating deposition analysis for visualization...")
            self.deposition_analysis_cache = self.calculate_deposition_color_map_optimized()
        
        print(f"\n=== Deposition Color Map Viewer ===")
        print(f"Available layers: 0-{len(self.toolpaths)-1}")
        print("Options:")
        print(f"  - Enter layer number (0-{len(self.toolpaths)-1}) to view deposition color map")
        print("  - Enter 'all' to view all layers sequentially")
        print(f"  - Enter 'range X-Y' to view layers X through Y (e.g., 'range 0-5')")
        print("  - Enter 'q' to skip")
        
        while True:
            try:
                user_input = input(f"\nEnter your choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Exiting deposition color map visualization...")
                    break
                    
                elif user_input == 'all':
                    print("Showing deposition color maps for all layers sequentially...")
                    for i in range(len(self.toolpaths)):
                        print(f"Displaying deposition color map for layer {i}/{len(self.toolpaths)-1}")
                        self.visualize_deposition_color_map_layer(i)
                        
                        if i < len(self.toolpaths) - 1:
                            continue_input = input("Press Enter for next layer, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                    break
                                
                elif user_input.startswith('range '):
                    try:
                        range_part = user_input.replace('range ', '')
                        start_layer, end_layer = map(int, range_part.split('-'))
                        
                        if 0 <= start_layer < len(self.toolpaths) and 0 <= end_layer < len(self.toolpaths) and start_layer <= end_layer:
                            print(f"Showing deposition color maps for layers {start_layer} to {end_layer}...")
                            for i in range(start_layer, end_layer + 1):
                                print(f"Displaying deposition color map for layer {i}")
                                self.visualize_deposition_color_map_layer(i)
                                
                                if i < end_layer:
                                    continue_input = input("Press Enter for next layer, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                            break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 0 and {len(self.toolpaths)-1}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 0-5')")
                        
                else:
                    # Try to parse as single layer number
                    try:
                        layer_idx = int(user_input)
                        if 0 <= layer_idx < len(self.toolpaths):
                            print(f"Displaying deposition color map for layer {layer_idx}")
                            self.visualize_deposition_color_map_layer(layer_idx)
                            
                            next_input = input("View another layer? (Enter layer number, 'q' to exit, or press Enter): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                        else:
                            print(f"Invalid layer number. Please enter 0-{len(self.toolpaths)-1}")
                    except ValueError:
                        print("Invalid input. Type 'q' to exit or see options above.")
                        
            except KeyboardInterrupt:
                print("\nExiting deposition color map visualization...")
                break
    
    def visualize_deposition_color_map_layer(self, layer_idx):
        """Visualize deposition color map for a specific layer (AS4RCSD3 style)."""
        if layer_idx >= len(self.toolpaths):
            logger.error(f"Layer {layer_idx} not available")
            return
        
        # Get cached analysis or calculate if needed
        if not hasattr(self, 'deposition_analysis_cache'):
            self.deposition_analysis_cache = self.calculate_deposition_color_map_optimized()
        
        if layer_idx >= len(self.deposition_analysis_cache["per_layer"]):
            logger.error(f"Layer {layer_idx} analysis not available")
            return
        
        # Get layer data
        layer_metrics = self.deposition_analysis_cache["per_layer"][layer_idx]
        color_map = self.deposition_analysis_cache["color_maps"][layer_idx]
        
        _, boundary_points = self.get_layer_boundary(layer_idx)
        deposition, _, _, _ = self.toolpaths[layer_idx]
        contour = self.contours[layer_idx] if layer_idx < len(self.contours) else np.array([])
        
        # Create visualization (AS4RCSD3 style)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Deposition color map
        from shapely.geometry import Polygon
        boundary_polygon = Polygon(boundary_points)
        min_x, min_y, max_x, max_y = boundary_polygon.bounds
        
        # Create color-coded visualization
        im = ax1.imshow(color_map, extent=[min_x, max_x, min_y, max_y], 
                       origin='lower', cmap='viridis', alpha=0.8)
        
        # Plot boundary and toolpath
        ax1.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2, label='Boundary')
        if len(deposition) > 1:
            ax1.plot(deposition[:, 0], deposition[:, 1], 'w-', linewidth=1, alpha=0.9, label='Toolpath')
        
        # Plot contour if available
        if len(contour) > 1:
            contour_array = np.array(contour)
            ax1.scatter(contour_array[:, 0], contour_array[:, 1], c='red', s=10, alpha=0.8, label='Damage Contour')
        
        ax1.set_title(f'Layer {layer_idx} - Deposition Color Map\n'
                     f'Coverage: {layer_metrics["coverage_percentage"]:.1f}%, '
                     f'Wall Eng: {layer_metrics["wall_engagement_index"]:.3f}', fontsize=14)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Deposition Intensity', fontsize=12)
        
        # Plot 2: Metrics summary
        ax2.axis('off')
        metrics_text = f"""
Layer {layer_idx} Deposition Analysis:

📊 COVERAGE METRICS:
  Coverage: {layer_metrics['coverage_percentage']:.1f}%
  Over-Deposition: {layer_metrics['over_deposition_percentage']:.1f}%
  Lateral Support: {layer_metrics['lateral_support_percentage']:.1f}%

🔧 QUALITY METRICS:
  Supporting Contour: {layer_metrics['supporting_contour_percentage']:.1f}%
  Wall Engagement Index: {layer_metrics['wall_engagement_index']:.3f}

📈 GRID STATISTICS:
  Total Cells: {layer_metrics['total_cells']}
  Deposition Cells: {layer_metrics['deposition_cells']}
  Support Cells: {layer_metrics['support_cells']}
  Over-Deposition Cells: {layer_metrics['over_deposition_cells']}
  Zero Deposition Cells: {layer_metrics['zero_deposition_cells']}

🎨 COLOR MAP LEGEND:
  Dark Blue: No deposition
  Light Blue: Low deposition
  Green/Yellow: Good deposition
  Red: High/over deposition
        """
        
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"deposition_colormap_layer_{layer_idx}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Deposition color map saved to {plot_file}")
        
        plt.show()
        
        return layer_metrics
        
    def export_toolpaths(self):
        """Export toolpaths to various formats."""
        if not self.config.export_csv:
            return
            
        logger.info("Exporting toolpaths...")
        
        try:
            # Link layers and create 3D trajectory
            full_3d_trajectory = self.link_layers_to_3d()
            
            # Export to CSV
            csv_file = Path(self.config.output_dir) / "toolpath_trajectory.csv"
            self.export_trajectory_to_csv(full_3d_trajectory, csv_file)
            
            # Export configuration
            config_file = Path(self.config.output_dir) / "toolpath_config.json"
            self.config.save_json(str(config_file))
            
            logger.info(f"Toolpaths exported to {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting toolpaths: {e}")
            
    def link_layers_to_3d(self):
        """Link all layers to create a continuous 3D trajectory."""
        continuous_trajectory = []
        current_z = 0
        
        for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
            # Convert 2D paths to 3D by adding Z coordinate
            if len(deposition) > 0:
                deposition_3d = np.hstack((deposition[:, :2], np.full((deposition.shape[0], 1), current_z)))
                continuous_trajectory.append(deposition_3d)
                
            # Add tool moves, fly lines, fillets similarly...
            # (Implementation details would go here)
            
            # Increment Z for next layer
            current_z += self.config.layer_height
            
        # Flatten trajectory
        if continuous_trajectory:
            return np.vstack(continuous_trajectory)
        else:
            return np.empty((0, 3))
            
    def export_trajectory_to_csv(self, trajectory, filename):
        """Export 3D trajectory to CSV file."""
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z"])
            for point in trajectory:
                writer.writerow(point)
        logger.info(f"Trajectory exported to {filename}")
        
    def visualize_results(self):
        """Visualize toolpath results."""
        if not self.config.enable_visualization:
            return
            
        logger.info("Generating visualizations...")
        
        # 1. Show 3D slices if requested
        if self.config.show_3d_slices:
            self.visualize_3d_slices()
            
        # 2. Show boundary extension process if requested
        if self.config.show_boundary_extension:
            self.visualize_boundary_extension_process()
            
        # 3. Show hybrid process if requested
        if self.config.show_hybrid_process:
            self.visualize_hybrid_process()
            
        # 4. Show layer toolpath with color map if requested
        if self.config.show_layer_toolpath:
            self.visualize_layer_toolpath_with_colormap()
            
        # 5. Show 3D toolpath if requested
        if self.config.show_3d_toolpath:
            self.visualize_3d_toolpath()
        
        if self.config.visualize_all_layers:
            # Visualize all layers
            for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
                logger.info(f"Visualizing layer {i + 1}")
                self.visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets, i)
        else:
            # Visualize specific layers or provide interactive selection
            self.interactive_layer_visualization()
            
    def visualize_toolpath_2d(self, deposition, tool_moves, fly_lines, fillets, layer_index):
        """Visualize a single layer's toolpath in 2D."""
        plt.figure(figsize=(8, 8))
        
        # Plot deposition path
        if deposition.size > 0:
            plt.plot(deposition[:, 0], deposition[:, 1], 'b-', label='Deposition Path', linewidth=2)
            
        # Plot tool moves
        if tool_moves.size > 0:
            for i in range(0, len(tool_moves), 2):
                segment = tool_moves[i:i+2]
                if len(segment) == 2:
                    plt.plot(segment[:, 0], segment[:, 1], 'r--', 
                           label='Tool Moves' if i == 0 else "", alpha=0.7)
                           
        # Plot fly lines
        if fly_lines.size > 0:
            for i in range(0, len(fly_lines), 2):
                segment = fly_lines[i:i+2]
                if len(segment) == 2:
                    plt.plot(segment[:, 0], segment[:, 1], 'g--', 
                           label='Fly Moves' if i == 0 else "", alpha=0.7)
                           
        # Plot fillets
        if fillets.size > 0:
            for i in range(0, len(fillets), 2):
                segment = fillets[i:i+2]
                if len(segment) == 2:
                    plt.plot(segment[:, 0], segment[:, 1], 'y-', 
                           label='Fillets' if i == 0 else "", alpha=0.7)
        
        plt.axis('equal')
        plt.legend()
        plt.title(f'Layer {layer_index + 1} Toolpath Visualization')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True, alpha=0.3)
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"layer_{layer_index + 1}_toolpath.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def interactive_layer_visualization(self):
        """Provide interactive layer visualization with enhanced options."""
        print(f"\n=== Interactive Layer Visualization ===")
        print(f"Available layers: 1-{len(self.toolpaths)}")
        print("Options:")
        print("  - Enter layer number (1-{}) to view specific layer".format(len(self.toolpaths)))
        print("  - Enter 'all' to view all layers sequentially")
        print("  - Enter 'range X-Y' to view layers X through Y (e.g., 'range 1-5')")
        print("  - Enter 'list' to see layer summary")
        print("  - Enter 'q' to quit")
        
        while True:
            try:
                layer_input = input(f"\nEnter your choice: ").strip().lower()
                
                if layer_input == 'q':
                    print("Exiting layer visualization...")
                    break
                    
                elif layer_input == 'all':
                    print("Showing all layers sequentially...")
                    for i in range(len(self.toolpaths)):
                        print(f"Displaying layer {i + 1}/{len(self.toolpaths)}")
                        deposition, tool_moves, fly_lines, fillets = self.toolpaths[i]
                        self.visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets, i)
                        
                        if i < len(self.toolpaths) - 1:  # Don't ask after the last layer
                            continue_input = input("Press Enter for next layer, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                                
                elif layer_input.startswith('range '):
                    try:
                        range_part = layer_input.replace('range ', '')
                        start_layer, end_layer = map(int, range_part.split('-'))
                        
                        if 1 <= start_layer <= len(self.toolpaths) and 1 <= end_layer <= len(self.toolpaths) and start_layer <= end_layer:
                            print(f"Showing layers {start_layer} to {end_layer}...")
                            for i in range(start_layer - 1, end_layer):
                                print(f"Displaying layer {i + 1}")
                                deposition, tool_moves, fly_lines, fillets = self.toolpaths[i]
                                self.visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets, i)
                                
                                if i < end_layer - 1:  # Don't ask after the last layer in range
                                    continue_input = input("Press Enter for next layer, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 1 and {len(self.toolpaths)}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 1-5')")
                        
                elif layer_input == 'list':
                    print(f"\n=== Layer Summary ===")
                    print(f"Total layers: {len(self.toolpaths)}")
                    for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
                        dep_points = len(deposition) if len(deposition) > 0 else 0
                        move_points = len(tool_moves) if len(tool_moves) > 0 else 0
                        fly_points = len(fly_lines) if len(fly_lines) > 0 else 0
                        print(f"  Layer {i + 1}: {dep_points} deposition pts, {move_points} travel pts, {fly_points} fly pts")
                    print()
                    
                else:
                    # Try to parse as single layer number
                    try:
                        layer_idx = int(layer_input) - 1
                        if 0 <= layer_idx < len(self.toolpaths):
                            print(f"Displaying layer {layer_idx + 1}")
                            deposition, tool_moves, fly_lines, fillets = self.toolpaths[layer_idx]
                            self.visualize_toolpath_2d(deposition, tool_moves, fly_lines, fillets, layer_idx)
                            
                            # Ask if user wants to see another layer
                            next_input = input("View another layer? (Enter layer number, 'q' to quit, or press Enter to continue): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                            else:
                                # Process the next input in the next iteration
                                continue
                        else:
                            print(f"Invalid layer number. Please enter 1-{len(self.toolpaths)}")
                    except ValueError:
                        print("Invalid input. Type 'q' to quit or see options above.")
                        
            except KeyboardInterrupt:
                print("\nExiting layer visualization...")
                break
                
    def visualize_3d_slices(self):
        """Visualize 3D slices of the model."""
        logger.info("Visualizing 3D slices...")
        
        # Plot all layers in 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for slice_position, slice_points in self.slices:
            if len(slice_points) > 1:
                slice_points = np.vstack([slice_points, slice_points[0]])  # Close the loop
                ax.plot(slice_points[:, 0], slice_points[:, 1], 
                       np.full(len(slice_points), slice_position), 
                       'b-', alpha=0.7, linewidth=1)

        ax.set_xlabel('X (mm)', fontsize=14)
        ax.set_ylabel('Y (mm)', fontsize=14)
        ax.set_zlabel('Z (mm)', fontsize=14)
        ax.set_title("3D Model Slices", fontsize=16)
        ax.grid(True)
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / "3d_slices.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"3D slices plot saved to {plot_file}")
            
        plt.show()
        
        # Interactive slice viewing options
        print(f"\n=== 2D Slice Viewer ===")
        print(f"Available slices: 0-{len(self.slices)-1}")
        print("Options:")
        print(f"  - Enter slice number (0-{len(self.slices)-1}) to view 2D slice with contour")
        print("  - Enter 'all' to view all slices sequentially")
        print(f"  - Enter 'range X-Y' to view slices X through Y (e.g., 'range 0-5')")
        print("  - Enter 'list' to see slice summary")
        print("  - Enter 'q' to continue")
        
        while True:
            try:
                user_input = input(f"\nEnter your choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Continuing to next visualization...")
                    break
                    
                elif user_input == 'all':
                    print("Showing all slices sequentially...")
                    for i in range(len(self.slices)):
                        print(f"Displaying slice {i}/{len(self.slices)-1}")
                        self.visualize_2d_slice_with_contour(i)
                        
                        if i < len(self.slices) - 1:
                            continue_input = input("Press Enter for next slice, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                                
                elif user_input.startswith('range '):
                    try:
                        range_part = user_input.replace('range ', '')
                        start_slice, end_slice = map(int, range_part.split('-'))
                        
                        if 0 <= start_slice < len(self.slices) and 0 <= end_slice < len(self.slices) and start_slice <= end_slice:
                            print(f"Showing slices {start_slice} to {end_slice}...")
                            for i in range(start_slice, end_slice + 1):
                                print(f"Displaying slice {i}")
                                self.visualize_2d_slice_with_contour(i)
                                
                                if i < end_slice:
                                    continue_input = input("Press Enter for next slice, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 0 and {len(self.slices)-1}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 0-5')")
                        
                elif user_input == 'list':
                    print(f"\n=== Slice Summary ===")
                    print(f"Total slices: {len(self.slices)}")
                    for i, (slice_position, boundary_points) in enumerate(self.slices):
                        boundary_pts = len(boundary_points) if len(boundary_points) > 0 else 0
                        print(f"  Slice {i}: Z={slice_position:.2f}mm, {boundary_pts} boundary points")
                    print()
                    
                else:
                    # Try to parse as single slice number
                    try:
                        layer_idx = int(user_input)
                        if 0 <= layer_idx < len(self.slices):
                            print(f"Displaying slice {layer_idx}")
                            self.visualize_2d_slice_with_contour(layer_idx)
                            
                            next_input = input("View another slice? (Enter slice number, 'q' to continue, or press Enter): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                        else:
                            print(f"Invalid slice number. Please enter 0-{len(self.slices)-1}")
                    except ValueError:
                        print("Invalid input. Type 'q' to continue or see options above.")
                        
            except KeyboardInterrupt:
                print("\nContinuing to next visualization...")
                break
                
    def visualize_2d_slice_with_contour(self, layer_idx):
        """Visualize 2D slice with boundary and contour."""
        slice_position, boundary_points = self.get_layer_boundary(layer_idx)
        contour = self.contours[layer_idx] if layer_idx < len(self.contours) else np.array([])
        
        plt.figure(figsize=(10, 8))
        
        # Plot boundary (ensure it's closed)
        if len(boundary_points) > 0:
            closed_boundary = self.ensure_closed_boundary(boundary_points)
            plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], 'b-', linewidth=2, label='Boundary')
            
        # Plot contour
        if len(contour) > 0:
            plt.scatter(contour[:, 0], contour[:, 1], c='red', s=20, alpha=0.7, label='Damage Contour')
            
        plt.xlabel('X (mm)', fontsize=14)
        plt.ylabel('Y (mm)', fontsize=14)
        plt.title(f'Layer {layer_idx} - Height: {slice_position:.2f}mm', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"layer_{layer_idx}_2d_slice.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"2D slice plot saved to {plot_file}")
            
        plt.show()
        
    def visualize_boundary_extension_process(self):
        """Visualize the complete turbine repair boundary extension process.
        
        This function shows the complete turbine blade repair process including:
        1. Circle fitting with contact points
        2. Curvature filtering and spline fitting
        3. Left and right spline extraction
        4. Spline extension and intersection finding
        5. Final boundary reconstruction
        6. Original vs final boundary comparison
        """
        if not self.config.enable_turbine_repair:
            logger.info("Turbine repair not enabled, skipping boundary extension visualization")
            return
            
        logger.info("Visualizing complete turbine repair boundary extension process...")
        
        # Interactive boundary extension visualization
        print(f"\n=== Boundary Extension Process Viewer ===")
        print(f"Available layers: 0-{len(self.slices)-1}")
        print("Options:")
        print(f"  - Enter layer number (0-{len(self.slices)-1}) to visualize boundary extension process")
        print("  - Enter 'all' to view all layers sequentially")
        print(f"  - Enter 'range X-Y' to view layers X through Y (e.g., 'range 0-5')")
        print("  - Enter 'q' to skip")
        
        while True:
            try:
                user_input = input(f"\nEnter your choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Skipping boundary extension visualization...")
                    break
                    
                elif user_input == 'all':
                    print("Showing boundary extension for all layers sequentially...")
                    for i in range(len(self.slices)):
                        print(f"Displaying boundary extension for layer {i}/{len(self.slices)-1}")
                        self.visualize_boundary_extension_for_layer(i)
                        
                        if i < len(self.slices) - 1:
                            continue_input = input("Press Enter for next layer, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                    break
                                
                elif user_input.startswith('range '):
                    try:
                        range_part = user_input.replace('range ', '')
                        start_layer, end_layer = map(int, range_part.split('-'))
                        
                        if 0 <= start_layer < len(self.slices) and 0 <= end_layer < len(self.slices) and start_layer <= end_layer:
                            print(f"Showing boundary extension for layers {start_layer} to {end_layer}...")
                            for i in range(start_layer, end_layer + 1):
                                print(f"Displaying boundary extension for layer {i}")
                                self.visualize_boundary_extension_for_layer(i)
                                
                                if i < end_layer:
                                    continue_input = input("Press Enter for next layer, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                            break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 0 and {len(self.slices)-1}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 0-5')")
                        
                else:
                    # Try to parse as single layer number
                    try:
                        layer_idx = int(user_input)
                        if 0 <= layer_idx < len(self.slices):
                            print(f"Displaying boundary extension for layer {layer_idx}")
                            self.visualize_boundary_extension_for_layer(layer_idx)
                            
                            next_input = input("View another layer? (Enter layer number, 'q' to skip, or press Enter): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                        else:
                            print(f"Invalid layer number. Please enter 0-{len(self.slices)-1}")
                    except ValueError:
                        print("Invalid input. Type 'q' to skip or see options above.")
                        
            except KeyboardInterrupt:
                print("\nSkipping boundary extension visualization...")
                break
                
    def visualize_boundary_extension_for_layer(self, layer_idx):
        """Visualize complete turbine repair boundary extension process for a specific layer."""
        slice_position, original_boundary = self.slices[layer_idx]
        
        # Check if we have turbine repair data for this layer
        if hasattr(self, 'turbine_repair_data') and self.turbine_repair_data['layer_index'] == layer_idx:
            self.visualize_complete_turbine_repair_process(self.turbine_repair_data)
        else:
            # Process the layer to get turbine repair data
            processed_boundary = self.process_turbine_repair(original_boundary, layer_idx)
            if hasattr(self, 'turbine_repair_data'):
                self.visualize_complete_turbine_repair_process(self.turbine_repair_data)
            else:
                logger.warning(f"No turbine repair data available for layer {layer_idx}")
                
    def visualize_complete_turbine_repair_process(self, data):
        """Visualize the complete turbine repair process in a comprehensive 2x3 subplot."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Complete Turbine Repair Process - Layer {data["layer_index"]}', fontsize=18)
        
        # Plot 1: Circle Fitting with Fitted Circles
        axes[0, 0].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b-', linewidth=2, label='Original Boundary')
        
        # Plot fitted circles
        for (cx, cy, r) in data['fitted_circles']:
            circle = plt.Circle((cx, cy), r, color='blue', alpha=0.3)
            axes[0, 0].add_patch(circle)
            axes[0, 0].plot(cx, cy, 'ko', markersize=4)
        
        # Plot circle contact points
        for circle_data in data['circles_with_boundary_points']:
            if len(circle_data) > 3:
                contact_points = circle_data[3]
                axes[0, 0].plot(contact_points[:, 0], contact_points[:, 1], 'rx', markersize=6)
        
        axes[0, 0].set_title('1. Circle Fitting', fontsize=14)
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # Plot 2: Curvature Filtering
        axes[0, 1].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b-', linewidth=2)
        
        # Plot original fitted circles
        for (cx, cy, r) in data['fitted_circles']:
            circle = plt.Circle((cx, cy), r, color='blue', alpha=0.2)
            axes[0, 1].add_patch(circle)
        
        axes[0, 1].scatter(data['centers'][:, 0], data['centers'][:, 1], c='gray', s=60, alpha=0.4)
        
        # Plot fitted spline
        if len(data['x_spline']) > 0:
            axes[0, 1].plot(data['x_spline'], data['y_spline'], 'k--', linewidth=1.2, label='B-spline')
        
        # Plot filtered centers (colored by curvature)
        if len(data['filtered_centers']) > 0:
            sc = axes[0, 1].scatter(data['filtered_centers'][:, 0], data['filtered_centers'][:, 1],
                                   c=np.abs(data['filtered_curvatures']), cmap='plasma', s=100,
                                   edgecolors='black')
        
        axes[0, 1].set_title('2. Curvature Filtering', fontsize=14)
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # Plot 3: Left and Right Splines
        axes[0, 2].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b-', linewidth=2)
        
        if data['left_spline'] is not None:
            axes[0, 2].plot(data['left_spline'][0], data['left_spline'][1], 'g-', linewidth=2, label='Left Spline')
        if data['right_spline'] is not None:
            axes[0, 2].plot(data['right_spline'][0], data['right_spline'][1], 'r-', linewidth=2, label='Right Spline')
            
        if len(data['left_points']) > 0:
            axes[0, 2].plot(data['left_points'][:, 0], data['left_points'][:, 1], 'gx', markersize=8, label='Left Points')
        if len(data['right_points']) > 0:
            axes[0, 2].plot(data['right_points'][:, 0], data['right_points'][:, 1], 'rx', markersize=8, label='Right Points')
        
        axes[0, 2].set_title('3. Left and Right Splines', fontsize=14)
        axes[0, 2].set_xlabel('X (mm)')
        axes[0, 2].set_ylabel('Y (mm)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')
        
        # Plot 4: Extended Splines with Intersection
        axes[1, 0].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b-', linewidth=2)
        
        # Plot extended splines
        if len(data['x_left_ext']) > 0:
            axes[1, 0].plot(data['x_left_ext'], data['y_left_ext'], 'g--', linewidth=2, label='Extended Left Spline')
        if len(data['x_right_ext']) > 0:
            axes[1, 0].plot(data['x_right_ext'], data['y_right_ext'], 'r--', linewidth=2, label='Extended Right Spline')
        
        # Plot intersection point
        if data['intersection_x'] is not None and data['intersection_y'] is not None:
            axes[1, 0].plot(data['intersection_x'], data['intersection_y'], 'bo', markersize=8, label='Intersection Point')
        
        axes[1, 0].set_title('4. Extended Splines with Intersection', fontsize=14)
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Plot 5: Final Boundary Reconstruction
        axes[1, 1].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b--', linewidth=1, alpha=0.5, label='Original Boundary')
        axes[1, 1].plot(data['final_boundary'][:, 0], data['final_boundary'][:, 1], 'r-', linewidth=2, label='Reconstructed Boundary')
        
        # Close the polygon visually
        axes[1, 1].plot([data['final_boundary'][-1, 0], data['final_boundary'][0, 0]], 
                       [data['final_boundary'][-1, 1], data['final_boundary'][0, 1]], 'r-', linewidth=2)
        
        axes[1, 1].set_title('5. Final Boundary Reconstruction', fontsize=14)
        axes[1, 1].set_xlabel('X (mm)')
        axes[1, 1].set_ylabel('Y (mm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        # Plot 6: Comparison (Original vs Final) - Show boundary closure
        axes[1, 2].plot(data['original_boundary'][:, 0], data['original_boundary'][:, 1], 'b-', linewidth=2, label='Original Boundary')
        axes[1, 2].plot(data['final_boundary'][:, 0], data['final_boundary'][:, 1], 'r-', linewidth=2, label='Final Boundary (Closed)')
        
        # Explicitly show the closing connection for final boundary
        # if len(data['final_boundary']) > 2:
        #     axes[1, 2].plot([data['final_boundary'][-1, 0], data['final_boundary'][0, 0]], 
        #                    [data['final_boundary'][-1, 1], data['final_boundary'][0, 1]], 
        #                    'r-', linewidth=3, alpha=0.8, label='Closing Connection')
        
        # Mark start/end points
        # axes[1, 2].plot(data['final_boundary'][0, 0], data['final_boundary'][0, 1], 'go', markersize=8, label='Start/End Point')
        
        # Fill the area between boundaries to show the difference
        try:
            from matplotlib.patches import Polygon as MplPolygon
            original_patch = MplPolygon(data['original_boundary'], alpha=0.2, facecolor='blue')
            final_patch = MplPolygon(data['final_boundary'], alpha=0.2, facecolor='red')
            axes[1, 2].add_patch(original_patch)
            axes[1, 2].add_patch(final_patch)
        except:
            pass
        
        axes[1, 2].set_title('6. Original vs Final (Closed for Toolpath)', fontsize=14)
        axes[1, 2].set_xlabel('X (mm)')
        axes[1, 2].set_ylabel('Y (mm)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"turbine_repair_process_layer_{data['layer_index']}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Turbine repair process plot saved to {plot_file}")
            
        plt.show()
        
    def visualize_hybrid_process(self):
        """Visualize the hybrid deposition method process."""
        if self.config.strategy != 6:
            logger.info("Hybrid strategy not selected, skipping hybrid process visualization")
            return
            
        logger.info("Visualizing hybrid deposition process...")
        
        # Interactive hybrid process visualization
        print(f"\n=== Hybrid Process Viewer ===")
        print(f"Available layers: 0-{len(self.slices)-1}")
        print("Options:")
        print(f"  - Enter layer number (0-{len(self.slices)-1}) to visualize hybrid process")
        print("  - Enter 'all' to view all layers sequentially")
        print(f"  - Enter 'range X-Y' to view layers X through Y (e.g., 'range 0-5')")
        print("  - Enter 'q' to skip")
        
        while True:
            try:
                user_input = input(f"\nEnter your choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Skipping hybrid process visualization...")
                    break
                    
                elif user_input == 'all':
                    print("Showing hybrid process for all layers sequentially...")
                    for i in range(len(self.slices)):
                        print(f"Displaying hybrid process for layer {i}/{len(self.slices)-1}")
                        self.visualize_hybrid_process_for_layer(i)
                        
                        if i < len(self.slices) - 1:
                            continue_input = input("Press Enter for next layer, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                    break
                                
                elif user_input.startswith('range '):
                    try:
                        range_part = user_input.replace('range ', '')
                        start_layer, end_layer = map(int, range_part.split('-'))
                        
                        if 0 <= start_layer < len(self.slices) and 0 <= end_layer < len(self.slices) and start_layer <= end_layer:
                            print(f"Showing hybrid process for layers {start_layer} to {end_layer}...")
                            for i in range(start_layer, end_layer + 1):
                                print(f"Displaying hybrid process for layer {i}")
                                self.visualize_hybrid_process_for_layer(i)
                                
                                if i < end_layer:
                                    continue_input = input("Press Enter for next layer, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                            break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 0 and {len(self.slices)-1}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 0-5')")
                        
                else:
                    # Try to parse as single layer number
                    try:
                        layer_idx = int(user_input)
                        if 0 <= layer_idx < len(self.slices):
                            print(f"Displaying hybrid process for layer {layer_idx}")
                            self.visualize_hybrid_process_for_layer(layer_idx)
                            
                            next_input = input("View another layer? (Enter layer number, 'q' to skip, or press Enter): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                        else:
                            print(f"Invalid layer number. Please enter 0-{len(self.slices)-1}")
                    except ValueError:
                        print("Invalid input. Type 'q' to skip or see options above.")
                        
            except KeyboardInterrupt:
                print("\nSkipping hybrid process visualization...")
                break
                
    def visualize_hybrid_process_for_layer(self, layer_idx):
        """Visualize hybrid process for a specific layer."""
        slice_position, boundary_points = self.get_layer_boundary(layer_idx)
        contour = self.contours[layer_idx] if layer_idx < len(self.contours) else np.array([])
        
        # Log boundary information for verification
        logger.info(f"Hybrid process visualization using boundary with {len(boundary_points)} points")
        if self.config.enable_turbine_repair:
            logger.info("Turbine repair enabled - should be using extended boundary")
        
        # Create a 2x2 subplot for the process
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        title = f'Hybrid Deposition Process - Layer {layer_idx}'
        if self.config.enable_turbine_repair:
            title += ' (Extended Boundary)'
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Contour offset
        closed_boundary = self.ensure_closed_boundary(boundary_points)
        boundary_label = 'Extended Boundary' if self.config.enable_turbine_repair else 'Boundary'
        axes[0, 0].plot(closed_boundary[:, 0], closed_boundary[:, 1], 'k-', linewidth=2, label=boundary_label)
        if len(contour) > 0:
            axes[0, 0].scatter(contour[:, 0], contour[:, 1], c='red', s=20, alpha=0.7, label='Damage Contour')
        axes[0, 0].set_title('1. Contour Offset', fontsize=14)
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # Plot 2: Trimmed lines
        axes[0, 1].plot(closed_boundary[:, 0], closed_boundary[:, 1], 'k-', linewidth=2, label=boundary_label)
        axes[0, 1].set_title('2. Trimmed Offset Lines', fontsize=14)
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # Plot 3: Without excess offset lines
        axes[1, 0].plot(closed_boundary[:, 0], closed_boundary[:, 1], 'k-', linewidth=2, label='Boundary')
        axes[1, 0].set_title('3. Cleaned Offset Lines', fontsize=14)
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Plot 4: Final linked deposition lines
        axes[1, 1].plot(closed_boundary[:, 0], closed_boundary[:, 1], 'k-', linewidth=2, label='Boundary')
        if layer_idx < len(self.toolpaths):
            deposition, _, _, _ = self.toolpaths[layer_idx]
            if len(deposition) > 0:
                axes[1, 1].plot(deposition[:, 0], deposition[:, 1], 'b-', linewidth=2, label='Final Toolpath')
        axes[1, 1].set_title('4. Linked Deposition Lines', fontsize=14)
        axes[1, 1].set_xlabel('X (mm)')
        axes[1, 1].set_ylabel('Y (mm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"hybrid_process_layer_{layer_idx}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Hybrid process plot saved to {plot_file}")
            
        plt.show()
        
    def visualize_layer_toolpath_with_colormap(self):
        """Visualize layer toolpath with color fade and deposition color map."""
        logger.info("Visualizing layer toolpath with color map...")
        
        # Interactive layer toolpath with colormap visualization
        print(f"\n=== Layer Toolpath with Colormap Viewer ===")
        print(f"Available layers: 0-{len(self.toolpaths)-1}")
        print("Options:")
        print(f"  - Enter layer number (0-{len(self.toolpaths)-1}) to visualize with color map")
        print("  - Enter 'all' to view all layers sequentially")
        print(f"  - Enter 'range X-Y' to view layers X through Y (e.g., 'range 0-5')")
        print("  - Enter 'q' to skip")
        
        while True:
            try:
                user_input = input(f"\nEnter your choice: ").strip().lower()
                
                if user_input == 'q':
                    print("Skipping layer toolpath colormap visualization...")
                    break
                    
                elif user_input == 'all':
                    print("Showing layer toolpath colormap for all layers sequentially...")
                    for i in range(len(self.toolpaths)):
                        print(f"Displaying layer toolpath colormap for layer {i}/{len(self.toolpaths)-1}")
                        self.visualize_layer_with_colormap(i)
                        
                        if i < len(self.toolpaths) - 1:
                            continue_input = input("Press Enter for next layer, 'q' to stop, or 's' to skip to end: ").strip().lower()
                            if continue_input == 'q':
                                break
                            elif continue_input == 's':
                                break
                    break
                                
                elif user_input.startswith('range '):
                    try:
                        range_part = user_input.replace('range ', '')
                        start_layer, end_layer = map(int, range_part.split('-'))
                        
                        if 0 <= start_layer < len(self.toolpaths) and 0 <= end_layer < len(self.toolpaths) and start_layer <= end_layer:
                            print(f"Showing layer toolpath colormap for layers {start_layer} to {end_layer}...")
                            for i in range(start_layer, end_layer + 1):
                                print(f"Displaying layer toolpath colormap for layer {i}")
                                self.visualize_layer_with_colormap(i)
                                
                                if i < end_layer:
                                    continue_input = input("Press Enter for next layer, 'q' to stop: ").strip().lower()
                                    if continue_input == 'q':
                                        break
                            break
                        else:
                            print(f"Invalid range. Please use format 'range X-Y' where X and Y are between 0 and {len(self.toolpaths)-1}")
                    except (ValueError, IndexError):
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 0-5')")
                        
                else:
                    # Try to parse as single layer number
                    try:
                        layer_idx = int(user_input)
                        if 0 <= layer_idx < len(self.toolpaths):
                            print(f"Displaying layer toolpath colormap for layer {layer_idx}")
                            self.visualize_layer_with_colormap(layer_idx)
                            
                            next_input = input("View another layer? (Enter layer number, 'q' to skip, or press Enter): ").strip()
                            if next_input.lower() == 'q':
                                break
                            elif next_input == '':
                                continue
                        else:
                            print(f"Invalid layer number. Please enter 0-{len(self.toolpaths)-1}")
                    except ValueError:
                        print("Invalid input. Type 'q' to skip or see options above.")
                        
            except KeyboardInterrupt:
                print("\nSkipping layer toolpath colormap visualization...")
                break
                
    def visualize_layer_with_colormap(self, layer_idx):
        """Visualize specific layer with toolpath and deposition color map."""
        deposition, tool_moves, fly_lines, fillets = self.toolpaths[layer_idx]
        slice_position, boundary_points = self.get_layer_boundary(layer_idx)
        
        # Log boundary information for verification
        logger.info(f"Layer toolpath visualization using boundary with {len(boundary_points)} points")
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        title = f'Layer {layer_idx} - Toolpath and Deposition Analysis'
        if self.config.enable_turbine_repair:
            title += ' (Extended Boundary)'
        fig.suptitle(title, fontsize=16)
        
        # Left plot: Toolpath with color fade
        if len(deposition) > 1:
            # Create segments for color mapping
            points = np.array(deposition)
            segments = np.array([points[i:i+2] for i in range(len(points) - 1)])
            
            # Create color map from start (blue) to end (red)
            colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
            
            # Plot segments with color fade
            lc = LineCollection(segments, colors=colors, linewidth=3)
            ax1.add_collection(lc)
            
            # Mark start and end points
            ax1.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start')
            ax1.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, label='End')
            
        # Plot boundary
        ax1.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', linewidth=1, alpha=0.7, label='Boundary')
        
        ax1.set_title('Toolpath with Color Fade', fontsize=14)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Right plot: Deposition color map (simplified version)
        # Create a simple deposition intensity map
        if len(deposition) > 1:
            # Create a grid for deposition analysis
            x_min, x_max = boundary_points[:, 0].min(), boundary_points[:, 0].max()
            y_min, y_max = boundary_points[:, 1].min(), boundary_points[:, 1].max()
            
            # Simple visualization - just show the toolpath density
            ax2.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2, label='Boundary')
            
            # Plot deposition with thickness to show coverage
            for i in range(len(deposition) - 1):
                ax2.plot([deposition[i, 0], deposition[i+1, 0]], 
                        [deposition[i, 1], deposition[i+1, 1]], 
                        'b-', linewidth=self.config.tool_width*2, alpha=0.3)
                        
        ax2.set_title('Expected Deposition Coverage', fontsize=14)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / f"layer_{layer_idx}_toolpath_colormap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Layer toolpath with colormap saved to {plot_file}")
            
        plt.show()
        
    def visualize_3d_toolpath(self):
        """Visualize the complete 3D toolpath with layer connections."""
        logger.info("Visualizing 3D toolpath...")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        current_z = 0
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.toolpaths)))
        
        for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
            if len(deposition) > 0:
                # Plot deposition path for this layer
                deposition_3d = np.hstack((deposition[:, :2], np.full((deposition.shape[0], 1), current_z)))
                ax.plot(deposition_3d[:, 0], deposition_3d[:, 1], deposition_3d[:, 2], 
                       color=colors[i], linewidth=2, alpha=0.8)
                
                # Add ramp to next layer if not the last layer
                if i < len(self.toolpaths) - 1:
                    next_deposition, _, _, _ = self.toolpaths[i + 1]
                    if len(next_deposition) > 0:
                        # Connect last point of current layer to first point of next layer
                        end_point = deposition_3d[-1]
                        start_next = np.array([next_deposition[0, 0], next_deposition[0, 1], current_z + self.config.layer_height])
                        
                        # Draw ramp
                        ax.plot([end_point[0], start_next[0]], 
                               [end_point[1], start_next[1]], 
                               [end_point[2], start_next[2]], 
                               'r--', linewidth=1, alpha=0.6)
                        
            current_z += self.config.layer_height
            
        ax.set_xlabel('X (mm)', fontsize=14)
        ax.set_ylabel('Y (mm)', fontsize=14)
        ax.set_zlabel('Z (mm)', fontsize=14)
        ax.set_title('Complete 3D Toolpath', fontsize=16)
        
        # Add colorbar to show layer progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(self.toolpaths)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Layer Number', fontsize=12)
        
        if self.config.save_plots:
            plot_file = Path(self.config.output_dir) / "3d_toolpath.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"3D toolpath plot saved to {plot_file}")
            
        plt.show()
        
    def save_layer_toolpath_csv(self, layer_idx):
        """Save specific layer toolpath to CSV."""
        if layer_idx >= len(self.toolpaths):
            logger.error(f"Layer {layer_idx} does not exist")
            return
            
        deposition, tool_moves, fly_lines, fillets = self.toolpaths[layer_idx]
        slice_position, _ = self.slices[layer_idx]
        
        # Create 3D coordinates for the layer
        if len(deposition) > 0:
            deposition_3d = np.hstack((deposition[:, :2], np.full((deposition.shape[0], 1), slice_position)))
            
            csv_file = Path(self.config.output_dir) / f"layer_{layer_idx}_toolpath.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y', 'Z', 'Type'])
                
                for point in deposition_3d:
                    writer.writerow([point[0], point[1], point[2], 'Deposition'])
                    
            logger.info(f"Layer {layer_idx} toolpath saved to {csv_file}")
        else:
            logger.warning(f"No toolpath data for layer {layer_idx}")
            
    def save_complete_3d_toolpath_csv_with_contours(self):
        """Save complete 3D toolpath to CSV with proper contour sequencing and tool move identification."""
        csv_file = Path(self.config.output_dir) / "complete_3d_toolpath.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z', 'Layer', 'Type'])
            
            current_z = 0
            for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
                
                # Debug logging
                logger.info(f"Layer {i}: strategy='{self.config.strategy}', deposition shape: {deposition.shape if hasattr(deposition, 'shape') else 'no shape'}, "
                           f"tool_moves shape: {tool_moves.shape if hasattr(tool_moves, 'shape') else 'no shape'}, "
                           f"tool_moves length: {len(tool_moves) if hasattr(tool_moves, '__len__') else 'no length'}")
                
                # For contour strategy, write deposition and tool moves separately
                # Strategy 3 = contour strategy
                # Handle different strategies that need special tool move processing
                # Strategy 3 = contour strategy, Strategy 5 = hybrid1 strategy
                is_contour = (self.config.strategy == "contour" or 
                             self.config.strategy == 3 or 
                             str(self.config.strategy) == "3")
                
                is_hybrid1 = (self.config.strategy == "hybrid1" or 
                             self.config.strategy == 5 or 
                             str(self.config.strategy) == "5")
                
                if is_contour:
                    logger.info(f"*** Using contour sequence for layer {i} ***")
                    self._write_contour_separate_sequence_to_csv(writer, deposition, tool_moves, i, current_z)
                elif is_hybrid1:
                    logger.info(f"*** Using hybrid1 sequence for layer {i} ***")
                    self._write_hybrid1_sequence_to_csv(writer, deposition, tool_moves, i, current_z)
                else:
                    # For other strategies, use original method
                    logger.info(f"*** Using original sequence for layer {i} (strategy: {self.config.strategy}) ***")
                    self._write_original_sequence_to_csv(writer, deposition, tool_moves, fly_lines, fillets, i, current_z)
                        
                # Add ramp to next layer if not the last layer
                if i < len(self.toolpaths) - 1:
                    next_deposition, _, _, _ = self.toolpaths[i + 1]
                    
                    # Check if both current and next layer have deposition points
                    if (len(deposition) > 0 and len(next_deposition) > 0 and 
                        hasattr(deposition, '__len__') and hasattr(next_deposition, '__len__') and
                        len(deposition.shape) >= 1 and len(next_deposition.shape) >= 1):
                        
                        try:
                            # Ramp from last point of current layer to first point of next layer
                            end_point = deposition[-1]
                            start_next = next_deposition[0]
                            
                            # Ensure points are valid
                            if len(end_point) >= 2 and len(start_next) >= 2:
                                writer.writerow([end_point[0], end_point[1], current_z, i, 'Ramp_Start'])
                                writer.writerow([start_next[0], start_next[1], current_z + self.config.layer_height, i+1, 'Ramp_End'])
                        except (IndexError, TypeError) as e:
                            logger.warning(f"Could not create ramp between layers {i} and {i+1}: {e}")
                            
                current_z += self.config.layer_height
                
        logger.info(f"Complete 3D toolpath saved to {csv_file} with proper contour sequencing")
    
    def _write_contour_separate_sequence_to_csv(self, writer, deposition, tool_moves, layer_index, z_position):
        """Write contour toolpath with proper sequence: contour -> tool_move -> contour -> tool_move."""
        try:
            # We need to regenerate the proper sequence from the boundary points
            # because the current deposition array has all contours mixed together
            boundary_points = self._get_layer_boundary(layer_index)
            
            if boundary_points is not None:
                logger.info(f"Layer {layer_index}: Regenerating contour sequence from boundary points")
                self._write_regenerated_contour_sequence(writer, boundary_points, layer_index, z_position)
            else:
                logger.warning(f"Layer {layer_index}: No boundary points available, using fallback method")
                # Fallback: write deposition and tool moves separately (wrong order but better than nothing)
                self._write_fallback_sequence(writer, deposition, tool_moves, layer_index, z_position)
                        
        except Exception as e:
            logger.error(f"Error writing contour sequence for layer {layer_index}: {e}")
            # Fallback to original method
            self._write_original_sequence_to_csv(writer, deposition, tool_moves, [], [], layer_index, z_position)
    
    def _write_regenerated_contour_sequence(self, writer, boundary_points, layer_index, z_position):
        """Regenerate and write the proper contour sequence with correct ordering."""
        try:
            # Regenerate contour toolpath with proper sequencing
            boundary_points = np.asarray(boundary_points)
            
            # Ensure the boundary is closed
            if not np.allclose(boundary_points[0], boundary_points[-1]):
                boundary_points = np.vstack([boundary_points, boundary_points[0]])

            slice_polygon = Polygon(boundary_points)
            if not slice_polygon.is_valid:
                slice_polygon = slice_polygon.buffer(0)
                if not slice_polygon.is_valid:
                    logger.warning(f"Invalid polygon for layer {layer_index}")
                    return

            # Start with inner offset of half tool width for proper boundary clearance
            current_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
            if current_polygon.is_empty or not current_polygon.is_valid:
                logger.warning(f"Polygon too small after tool width offset for layer {layer_index}")
                return

            contour_count = 0
            total_points_written = 0
            
            while current_polygon.area > 0:
                if isinstance(current_polygon, MultiPolygon):
                    current_polygon = max(current_polygon.geoms, key=lambda p: p.area)

                coords = np.array(current_polygon.exterior.coords)
                
                # Ensure each contour is closed
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])

                # Write all points of this contour as deposition
                contour_points = 0
                for point in coords:
                    if len(point) >= 2:
                        writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
                        contour_points += 1
                        total_points_written += 1
                
                logger.info(f"  Contour {contour_count}: wrote {contour_points} deposition points")
                
                # Calculate next contour
                next_polygon = current_polygon.buffer(-self.config.hatch_distance)
                if not next_polygon.is_empty and next_polygon.is_valid and next_polygon.area > 0:
                    if isinstance(next_polygon, MultiPolygon):
                        next_polygon = max(next_polygon.geoms, key=lambda p: p.area)
                    
                    next_coords = np.array(next_polygon.exterior.coords)
                    if not np.allclose(next_coords[0], next_coords[-1]):
                        next_coords = np.vstack([next_coords, next_coords[0]])
                    
                    # Write tool move from end of current contour to start of next contour
                    end_point = coords[-1]
                    start_point = next_coords[0]
                    
                    if len(end_point) >= 2 and len(start_point) >= 2:
                        writer.writerow([end_point[0], end_point[1], z_position, layer_index, 'Tool_Move'])
                        writer.writerow([start_point[0], start_point[1], z_position, layer_index, 'Tool_Move'])
                        total_points_written += 2
                        logger.info(f"  Tool move {contour_count}: {end_point} -> {start_point}")

                current_polygon = next_polygon
                if current_polygon.is_empty or not current_polygon.is_valid:
                    break
                
                contour_count += 1
            
            logger.info(f"Layer {layer_index}: wrote {contour_count + 1} contours with {total_points_written} total points in correct sequence")
                
        except Exception as e:
            logger.error(f"Error regenerating contour sequence for layer {layer_index}: {e}")
    
    def _write_fallback_sequence(self, writer, deposition, tool_moves, layer_index, z_position):
        """Fallback method: write deposition and tool moves separately (wrong order)."""
        deposition_written = 0
        tool_moves_written = 0
        
        # Write all deposition points
        if hasattr(deposition, '__len__') and len(deposition) > 0:
            for point in deposition:
                if hasattr(point, '__len__') and len(point) >= 2:
                    writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
                    deposition_written += 1
        
        # Write all tool moves
        if hasattr(tool_moves, '__len__') and len(tool_moves) > 0:
            for i in range(0, len(tool_moves), 2):
                if i + 1 < len(tool_moves):
                    start_point = tool_moves[i]
                    end_point = tool_moves[i + 1]
                    
                    if (hasattr(start_point, '__len__') and len(start_point) >= 2 and
                        hasattr(end_point, '__len__') and len(end_point) >= 2):
                        writer.writerow([start_point[0], start_point[1], z_position, layer_index, 'Tool_Move'])
                        writer.writerow([end_point[0], end_point[1], z_position, layer_index, 'Tool_Move'])
                        tool_moves_written += 2
        
        logger.info(f"Layer {layer_index}: wrote {deposition_written} deposition points, {tool_moves_written} tool move points (fallback sequence)")

    def _write_hybrid1_sequence_to_csv(self, writer, deposition, tool_moves, layer_index, z_position):
        """Write hybrid1 toolpath with proper sequence: contour -> tool_move -> zigzag_fill."""
        try:
            # For hybrid1, we need to regenerate the proper sequence
            # The deposition array contains: contour points + zigzag points (mixed together)
            # The tool_moves array contains: [contour_to_zigzag, zigzag_connections...]
            
            boundary_points = self._get_layer_boundary(layer_index)
            
            if boundary_points is not None:
                logger.info(f"Layer {layer_index}: Regenerating hybrid1 sequence from boundary points")
                self._write_regenerated_hybrid1_sequence(writer, boundary_points, layer_index, z_position)
            else:
                logger.warning(f"Layer {layer_index}: No boundary points available for hybrid1, using fallback")
                self._write_fallback_sequence(writer, deposition, tool_moves, layer_index, z_position)
                        
        except Exception as e:
            logger.error(f"Error writing hybrid1 sequence for layer {layer_index}: {e}")
            self._write_fallback_sequence(writer, deposition, tool_moves, layer_index, z_position)
    
    def _write_regenerated_hybrid1_sequence(self, writer, boundary_points, layer_index, z_position):
        """Regenerate and write the proper hybrid1 sequence: contour -> tool_move -> zigzag."""
        try:
            # Regenerate the hybrid1 toolpath to get proper sequencing
            boundary_points = np.asarray(boundary_points)
            
            # Ensure boundary is closed
            if not np.allclose(boundary_points[0], boundary_points[-1]):
                boundary_points = np.vstack([boundary_points, boundary_points[0]])

            slice_polygon = Polygon(boundary_points)
            if not slice_polygon.is_valid:
                slice_polygon = slice_polygon.buffer(0)
                if not slice_polygon.is_valid:
                    logger.warning(f"Invalid polygon for layer {layer_index}")
                    return

            # Step 1: Generate and write contour
            contour_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
            if contour_polygon.is_empty:
                return

            contour_points = np.array(contour_polygon.exterior.coords)
            contour_points_written = 0
            
            # Write contour as deposition
            for point in contour_points:
                if len(point) >= 2:
                    writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
                    contour_points_written += 1
            
            logger.info(f"  Contour: wrote {contour_points_written} deposition points")

            # Step 2: Generate zigzag fill
            fill_polygon = contour_polygon.buffer(-self.config.tool_width)
            if fill_polygon.is_empty:
                logger.info(f"  No fill area for layer {layer_index}")
                return

            # Generate zigzag lines (simplified version)
            from shapely.affinity import rotate
            rotated_fill_polygon = rotate(fill_polygon, self.config.raster_angle, origin='center')
            min_x, min_y, max_x, max_y = rotated_fill_polygon.bounds

            zigzag_lines = []
            y = min_y
            
            while y <= max_y:
                raster_line = LineString([(min_x, y), (max_x, y)])
                clipped = rotated_fill_polygon.intersection(raster_line)

                if isinstance(clipped, LineString):
                    zigzag_lines.append(np.array(clipped.coords))
                elif hasattr(clipped, 'geoms'):  # MultiLineString
                    for subline in clipped.geoms:
                        zigzag_lines.append(np.array(subline.coords))

                y += self.config.hatch_distance

            # Reverse alternate lines and rotate back
            for i in range(len(zigzag_lines)):
                if i % 2 == 1:
                    zigzag_lines[i] = zigzag_lines[i][::-1]
                # Rotate back to original orientation
                zigzag_lines[i] = self.rotate_points(zigzag_lines[i], -self.config.raster_angle, fill_polygon.centroid)

            if zigzag_lines:
                # Step 3: Write tool move from contour to zigzag
                contour_end = contour_points[-1]
                zigzag_start = zigzag_lines[0][0]
                
                writer.writerow([contour_end[0], contour_end[1], z_position, layer_index, 'Tool_Move'])
                writer.writerow([zigzag_start[0], zigzag_start[1], z_position, layer_index, 'Tool_Move'])
                
                logger.info(f"  Tool move: contour end {contour_end} -> zigzag start {zigzag_start}")

                # Step 4: Write zigzag lines with deposition connections between them
                zigzag_points_written = 0
                connection_points_written = 0
                tool_moves_written = 2  # Already wrote contour->zigzag tool move
                
                for i, zigzag_line in enumerate(zigzag_lines):
                    # Write zigzag line as deposition
                    for point in zigzag_line:
                        if len(point) >= 2:
                            writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
                            zigzag_points_written += 1
                    
                    # Write deposition connection to next zigzag line (not tool move!)
                    if i < len(zigzag_lines) - 1:
                        line_end = zigzag_line[-1]
                        next_line_start = zigzag_lines[i + 1][0]
                        
                        writer.writerow([line_end[0], line_end[1], z_position, layer_index, 'Deposition'])
                        writer.writerow([next_line_start[0], next_line_start[1], z_position, layer_index, 'Deposition'])
                        connection_points_written += 2
                
                logger.info(f"  Zigzag: wrote {zigzag_points_written} zigzag points, {connection_points_written} connection points, {tool_moves_written} tool move points")
            
            total_points = contour_points_written + zigzag_points_written + connection_points_written + tool_moves_written
            logger.info(f"Layer {layer_index}: hybrid1 sequence complete - {total_points} total points")
                
        except Exception as e:
            logger.error(f"Error regenerating hybrid1 sequence for layer {layer_index}: {e}")

    def _write_contour_integrated_sequence_to_csv(self, writer, deposition, tool_moves, layer_index, z_position):
        """Write contour toolpath with proper integration using stored move types."""
        try:
            if len(deposition) == 0:
                return
            
            # Check if we have stored move types for this layer
            if hasattr(self, 'layer_move_types') and layer_index in self.layer_move_types:
                move_types = self.layer_move_types[layer_index]
                logger.info(f"Layer {layer_index}: Found stored move types, total: {len(move_types)}")
                
                # Write points with their correct types
                points_written = 0
                deposition_written = 0
                tool_moves_written = 0
                
                for i, point in enumerate(deposition):
                    if hasattr(point, '__len__') and len(point) >= 2:
                        if i < len(move_types):
                            move_type = move_types[i]
                        else:
                            move_type = 'Deposition'  # Default fallback
                        
                        writer.writerow([point[0], point[1], z_position, layer_index, move_type])
                        points_written += 1
                        
                        if move_type == 'Deposition':
                            deposition_written += 1
                        elif move_type == 'Tool_Move':
                            tool_moves_written += 1
                
                logger.info(f"Layer {layer_index}: wrote {points_written} total points "
                          f"({deposition_written} deposition, {tool_moves_written} tool moves)")
            else:
                # Fallback: write all as deposition and add tool moves separately
                logger.warning(f"No move types stored for layer {layer_index}, using fallback method")
                logger.warning(f"Available layer_move_types keys: {list(self.layer_move_types.keys()) if hasattr(self, 'layer_move_types') else 'None'}")
                self._write_original_sequence_to_csv(writer, deposition, tool_moves, [], [], layer_index, z_position)
                        
        except Exception as e:
            logger.error(f"Error writing contour integrated sequence for layer {layer_index}: {e}")
            # Fallback to original method
            self._write_original_sequence_to_csv(writer, deposition, tool_moves, [], [], layer_index, z_position)

    def _write_contour_sequence_to_csv(self, writer, boundary_points, layer_index, z_position):
        """Write contour toolpath sequence with proper tool move identification."""
        try:
            # Regenerate contour toolpath with proper sequencing
            boundary_points = np.asarray(boundary_points)
            
            # Check if boundary_points is valid
            if len(boundary_points) < 3:
                logger.warning(f"Insufficient boundary points for layer {layer_index}")
                return
            
            # Ensure the boundary is closed
            if not np.allclose(boundary_points[0], boundary_points[-1]):
                boundary_points = np.vstack([boundary_points, boundary_points[0]])

            slice_polygon = Polygon(boundary_points)
            if not slice_polygon.is_valid:
                slice_polygon = slice_polygon.buffer(0)
                if not slice_polygon.is_valid:
                    logger.warning(f"Invalid polygon for layer {layer_index}")
                    return

            # Start with inner offset of half tool width for proper boundary clearance
            current_polygon = slice_polygon.buffer(-self.config.tool_width / 2)
            if current_polygon.is_empty or not current_polygon.is_valid:
                logger.warning(f"Polygon too small after tool width offset for layer {layer_index}")
                return

            contour_index = 0
            while current_polygon.area > 0:
                if isinstance(current_polygon, MultiPolygon):
                    current_polygon = max(current_polygon.geoms, key=lambda p: p.area)

                coords = np.array(current_polygon.exterior.coords)
                
                # Ensure each contour is closed
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])

                # Write all points of this contour as deposition
                for point in coords:
                    if len(point) >= 2:  # Ensure point has at least X, Y coordinates
                        writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
                
                # Calculate next contour
                next_polygon = current_polygon.buffer(-self.config.hatch_distance)
                if not next_polygon.is_empty and next_polygon.is_valid and next_polygon.area > 0:
                    if isinstance(next_polygon, MultiPolygon):
                        next_polygon = max(next_polygon.geoms, key=lambda p: p.area)
                    
                    next_coords = np.array(next_polygon.exterior.coords)
                    if not np.allclose(next_coords[0], next_coords[-1]):
                        next_coords = np.vstack([next_coords, next_coords[0]])
                    
                    # Write tool move from end of current contour to start of next contour
                    end_point = coords[-1]
                    start_point = next_coords[0]
                    
                    if len(end_point) >= 2 and len(start_point) >= 2:
                        writer.writerow([end_point[0], end_point[1], z_position, layer_index, 'Tool_Move'])
                        writer.writerow([start_point[0], start_point[1], z_position, layer_index, 'Tool_Move'])

                current_polygon = next_polygon
                if current_polygon.is_empty or not current_polygon.is_valid:
                    break
                
                contour_index += 1
                
        except Exception as e:
            logger.error(f"Error writing contour sequence for layer {layer_index}: {e}")
            # Fallback to original method if contour generation fails
            if hasattr(self, 'toolpaths') and layer_index < len(self.toolpaths):
                deposition, tool_moves, fly_lines, fillets = self.toolpaths[layer_index]
                self._write_original_sequence_to_csv(writer, deposition, tool_moves, fly_lines, fillets, layer_index, z_position)
    
    def _write_original_sequence_to_csv(self, writer, deposition, tool_moves, fly_lines, fillets, layer_index, z_position):
        """Write original sequence for non-contour strategies."""
        try:
            # Write deposition points
            if hasattr(deposition, '__len__') and len(deposition) > 0:
                for point in deposition:
                    if hasattr(point, '__len__') and len(point) >= 2:
                        writer.writerow([point[0], point[1], z_position, layer_index, 'Deposition'])
            
            # Write tool moves
            if hasattr(tool_moves, '__len__') and len(tool_moves) > 0:
                for tool_move in tool_moves:
                    try:
                        if hasattr(tool_move, 'shape') and len(tool_move.shape) == 2 and tool_move.shape[0] == 2:
                            # Tool move is a 2-point segment
                            if len(tool_move[0]) >= 2 and len(tool_move[1]) >= 2:
                                writer.writerow([tool_move[0][0], tool_move[0][1], z_position, layer_index, 'Tool_Move'])
                                writer.writerow([tool_move[1][0], tool_move[1][1], z_position, layer_index, 'Tool_Move'])
                        else:
                            # Tool move is a sequence of points
                            if hasattr(tool_move, '__len__'):
                                for point in tool_move:
                                    if hasattr(point, '__len__') and len(point) >= 2:
                                        writer.writerow([point[0], point[1], z_position, layer_index, 'Tool_Move'])
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not write tool move for layer {layer_index}: {e}")
            
            # Write fly lines if any
            if hasattr(fly_lines, '__len__') and len(fly_lines) > 0:
                for point in fly_lines:
                    if hasattr(point, '__len__') and len(point) >= 2:
                        writer.writerow([point[0], point[1], z_position, layer_index, 'Fly_Line'])
            
            # Write fillets if any
            if hasattr(fillets, '__len__') and len(fillets) > 0:
                for point in fillets:
                    if hasattr(point, '__len__') and len(point) >= 2:
                        writer.writerow([point[0], point[1], z_position, layer_index, 'Fillet'])
                        
        except Exception as e:
            logger.error(f"Error writing original sequence for layer {layer_index}: {e}")
    
    def _get_layer_boundary(self, layer_index):
        """Get boundary points for a specific layer."""
        if hasattr(self, 'slices') and layer_index < len(self.slices):
            slice_position, boundary_points = self.slices[layer_index]
            if boundary_points is not None and len(boundary_points) > 0:
                # Return the boundary points of the slice
                return boundary_points
        return None

    def save_complete_3d_toolpath_csv(self):
        """Save complete 3D toolpath to CSV - wrapper method."""
        try:
            # Use the new method with proper contour sequencing
            self.save_complete_3d_toolpath_csv_with_contours()
        except Exception as e:
            logger.error(f"Error with enhanced CSV generation: {e}")
            logger.info("Falling back to original CSV generation method")
            self.save_complete_3d_toolpath_csv_original()
    
    def save_complete_3d_toolpath_csv_original(self):
        """Original CSV generation method as fallback."""
        csv_file = Path(self.config.output_dir) / "complete_3d_toolpath.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z', 'Layer', 'Type'])
            
            current_z = 0
            for i, (deposition, tool_moves, fly_lines, fillets) in enumerate(self.toolpaths):
                # Write deposition points
                if hasattr(deposition, '__len__') and len(deposition) > 0:
                    for point in deposition:
                        if hasattr(point, '__len__') and len(point) >= 2:
                            writer.writerow([point[0], point[1], current_z, i, 'Deposition'])
                
                # Write tool moves
                if hasattr(tool_moves, '__len__') and len(tool_moves) > 0:
                    for tool_move in tool_moves:
                        try:
                            if hasattr(tool_move, 'shape') and len(tool_move.shape) == 2 and tool_move.shape[0] == 2:
                                writer.writerow([tool_move[0][0], tool_move[0][1], current_z, i, 'Tool_Move'])
                                writer.writerow([tool_move[1][0], tool_move[1][1], current_z, i, 'Tool_Move'])
                            else:
                                for point in tool_move:
                                    if hasattr(point, '__len__') and len(point) >= 2:
                                        writer.writerow([point[0], point[1], current_z, i, 'Tool_Move'])
                        except:
                            pass  # Skip problematic tool moves
                            
                current_z += self.config.layer_height
                
        logger.info(f"Complete 3D toolpath saved to {csv_file} (original method)")
                
    def run(self):
        """Execute the complete toolpath generation pipeline."""
        logger.info("Starting toolpath generation pipeline...")
        
        try:
            # Load and process meshes
            self.load_meshes()
            self.apply_transformations()
            
            # Slice meshes
            self.slice_meshes()
            
            # Generate toolpaths
            self.generate_toolpaths()
            
            # Evaluate performance
            self.evaluate_toolpaths()
            
            # Export results
            self.export_toolpaths()
            
            # Visualize results
            self.visualize_results()
            
            # Save specific layer toolpath if requested
            if self.config.save_layer_toolpath is not None:
                self.save_layer_toolpath_csv(self.config.save_layer_toolpath)
                
            # Save complete 3D toolpath if requested
            if self.config.save_3d_toolpath:
                self.save_complete_3d_toolpath_csv()
            
            logger.info("Toolpath generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in toolpath generation pipeline: {e}")
            raise
            
    # Helper methods for geometric operations
    def rotate_points(self, points, angle, origin):
        """Rotate an array of 2D points around a given origin by an angle in degrees."""
        if points.size == 0:
            return points
        angle_rad = np.radians(angle)
        origin = np.array([origin.x, origin.y])
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                     [np.sin(angle_rad), np.cos(angle_rad)]])
        return np.dot(points - origin, rotation_matrix.T) + origin
        
    def create_fillet_between_racks(self, prev_end, curr_start, radius):
        """Create fillet arcs between racks (placeholder implementation)."""
        # This would contain the actual fillet creation logic
        # For now, return simple connecting line
        connecting_line = np.array([prev_end, curr_start])
        arc1 = np.array([prev_end])
        arc2 = np.array([curr_start])
        return arc1, connecting_line, arc2
        
    # Turbine repair helper methods (complete implementations)
    def densify_boundary(self, points, step=1.0):
        """Densify boundary points by adding intermediate points."""
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
        
    def fit_largest_circles(self, polygon, boundary_points, num_circles=3, resolution=0.5, tolerance=0.1, debug=False):
        """Fit circles inside the polygon such that they touch exactly two well-separated points on the boundary."""
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

                    # Find candidate boundary points close to the expected radius
                    dists = np.linalg.norm(boundary_points - np.array([x, y]), axis=1)
                    indices_close = np.where(np.abs(dists - distance_to_boundary) < tolerance)[0]
                    candidate_points = boundary_points[indices_close]

                    found_pair = False
                    for i in range(len(candidate_points)):
                        for j in range(i + 1, len(candidate_points)):
                            p1 = candidate_points[i]
                            p2 = candidate_points[j]

                            # Arc distance filter
                            index_diff = abs(indices_close[i] - indices_close[j])
                            if index_diff < 10:
                                continue

                            # Check if points are nearly opposite relative to center
                            v1 = p1 - np.array([x, y])
                            v2 = p2 - np.array([x, y])
                            v1 /= np.linalg.norm(v1)
                            v2 /= np.linalg.norm(v2)
                            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

                            if angle < np.deg2rad(150):
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
                    if len(circles) == 1:
                        ax.plot(circle_x, circle_y, 'ko', label='Circle Centers')
                        ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx', label='Boundary Points')
                    else:
                        ax.plot(circle_x, circle_y, 'ko')
                        ax.plot(near_pts[:, 0], near_pts[:, 1], 'rx')

        if debug:
            ax.set_aspect('equal')
            ax.set_title("Fitted Circles", fontsize=20)
            ax.set_xlabel("X (mm)", fontsize=18, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=18, labelpad=10)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend(loc="upper left", fontsize=12)
            ax.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()

        return circles, circles_with_boundary_points
        
    def compute_curvature_influence(self, fitted_circles, num_samples=500):
        """Compute curvature influence from fitted circles."""
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
        
    def filter_centers_by_curvature(self, centers, curvature_values, z_thresh=2.0):
        """Filter centers based on curvature threshold."""
        curvature_array = np.abs(np.array(curvature_values))
        mean_curv = np.mean(curvature_array)
        std_curv = np.std(curvature_array)
        
        # Z-score threshold filtering
        z_scores = (curvature_array - mean_curv) / std_curv
        keep_mask = np.abs(z_scores) <= z_thresh
        
        filtered_centers = centers[keep_mask]
        filtered_curvatures = curvature_array[keep_mask]
        
        return filtered_centers, filtered_curvatures
        
    def refit_spline_from_filtered_centers(self, filtered_centers, num_samples=500):
        """Refit spline from filtered centers."""
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
        
    def fit_left_right_splines_from_circle_contacts(self, filtered_circles, smoothing=0):
        """Fit left and right splines from circle contact points."""
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
        
    def extend_spline(self, spline, left_extension=10, right_extension=10, step=0.1):
        """Extend a 2D spline represented as (x, y) arrays."""
        if spline is None:
            return np.array([]), np.array([])
            
        # Normalize input to (x, y)
        if isinstance(spline, (list, np.ndarray)) and len(spline) == 2:
            x, y = np.array(spline[0]), np.array(spline[1])
        else:
            x, y = np.array(spline[0]), np.array(spline[1])

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
        
    def compute_spline_intersection(self, x_left, y_left, x_right, y_right):
        """Compute the intersection point between two splines."""
        # Create LineString objects from left and right spline coordinates
        left_line = LineString(np.column_stack((x_left, y_left)))
        right_line = LineString(np.column_stack((x_right, y_right)))

        # Compute the intersection using shapely
        intersection = left_line.intersection(right_line)

        # Handle different types of intersection results
        if intersection.is_empty:
            logger.debug("No intersection found.")
            return None, None
        elif intersection.geom_type == 'Point':
            return intersection.x, intersection.y
        else:
            logger.debug(f"Intersection is of type {intersection.geom_type}, handling not implemented.")
            return None, None
        
    def rebuild_boundary_with_spline_top(self, boundary_points, point1, point2, 
                                       x_left_ext, y_left_ext, x_right_ext, y_right_ext, intersection):
        """Rebuild boundary with spline top section."""
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
        
    def plot_updated_boundary(self, boundary, point1, point2, intersection):
        """Plot the final updated boundary."""
        plt.figure(figsize=(8, 8))
        
        # Plot the final boundary polygon
        plt.plot(boundary[:, 0], boundary[:, 1], 'b--', label='Updated Boundary')
        
        # Close the polygon visually
        plt.plot([boundary[-1, 0], boundary[0, 0]], [boundary[-1, 1], boundary[0, 1]], 'b--')

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
        
    def sort_points_counterclockwise(self, points):
        """Sort points counterclockwise."""
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
        sort_order = np.argsort(angles)
        return points[sort_order]
        
    def ensure_closed_boundary(self, boundary_points):
        """Ensure boundary is closed by connecting last point to first point."""
        if len(boundary_points) < 3:
            logger.warning(f"Boundary has only {len(boundary_points)} points, cannot ensure closure")
            return boundary_points
            
        boundary_points = np.asarray(boundary_points)
        
        # Check if boundary is already closed (last point equals first point)
        distance_to_start = np.linalg.norm(boundary_points[-1] - boundary_points[0])
        
        if not np.allclose(boundary_points[0], boundary_points[-1], atol=1e-6):
            # Add the first point at the end to close the boundary
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
            logger.debug(f"Boundary closed by connecting last point to first (distance: {distance_to_start:.6f})")
        else:
            logger.debug("Boundary is already closed")
            
        return boundary_points


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="AS4RCSD4 - Additive Manufacturing Toolpath Generation for Repair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--mesh-file", help="Path to the main mesh file (STL)")
    parser.add_argument("--non-planar-surface", help="Path to non-planar surface mesh file (STL)")
    
    # Processing parameters
    parser.add_argument("--layer-height", type=float, default=1.0, help="Layer height for slicing (mm)")
    parser.add_argument("--add-layers", type=int, default=0, help="Number of extra layers to add")
    parser.add_argument("--slicing-axis", choices=["x", "y", "z"], default="z", help="Slicing axis")
    parser.add_argument("--rotate-180", action="store_true", help="Rotate model 180 degrees")
    
    # Toolpath parameters
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
                       help="Toolpath strategy (1:Raster, 2:ZigZag, 3:Contour, 4:Spiral, 5:Hybrid1, 6:Hybrid2)")
    parser.add_argument("--tool-width", type=float, default=2.0, help="Tool width/laser spot diameter (mm)")
    parser.add_argument("--hatch-distance", type=float, default=1.5, help="Hatch distance between lines (mm)")
    parser.add_argument("--fly-in-out", action="store_true", help="Enable fly-in and fly-out paths")
    parser.add_argument("--fly-distance", type=float, default=2.0, help="Fly-in/out distance (mm)")
    parser.add_argument("--raster-angle", type=float, default=0.0, help="Cross section angle (degrees)")
    parser.add_argument("--corner-radius", type=float, default=0.0, help="Corner radius for smoothing (mm)")
    
    # Geometry processing
    parser.add_argument("--enable-turbine-repair", action="store_true", help="Enable turbine blade repair processing")
    parser.add_argument("--extension-length", type=float, default=5.0, help="Extension length for turbine repair")
    parser.add_argument("--spline-smoothing", type=float, default=0.1, help="Spline smoothing factor")
    parser.add_argument("--num-circles", type=int, default=5, help="Number of circles for turbine repair")
    parser.add_argument("--circle-resolution", type=float, default=0.05, help="Circle fitting resolution")
    parser.add_argument("--circle-tolerance", type=float, default=0.05, help="Circle fitting tolerance")
    parser.add_argument("--curvature-threshold", type=float, default=2.0, help="Curvature filtering threshold")
    
    # Evaluation and output
    parser.add_argument("--disable-evaluation", action="store_true", help="Disable toolpath evaluation")
    parser.add_argument("--disable-visualization", action="store_true", help="Disable visualization")
    parser.add_argument("--no-export", action="store_true", help="Disable CSV export")
    parser.add_argument("--output-dir", default="output", help="Output directory for results")
    parser.add_argument("--debug-layer", type=int, help="Specific layer for debug analysis")
    
    # Visualization options
    parser.add_argument("--visualize-all", action="store_true", help="Visualize all layers automatically")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--show-3d-slices", action="store_true", help="Show 3D slices visualization")
    parser.add_argument("--show-boundary-extension", action="store_true", help="Show boundary extension process (turbine repair)")
    parser.add_argument("--show-hybrid-process", action="store_true", help="Show hybrid deposition method process")
    parser.add_argument("--show-layer-toolpath", action="store_true", help="Show detailed layer toolpath with color map")
    parser.add_argument("--show-3d-toolpath", action="store_true", help="Show final 3D toolpath visualization")
    parser.add_argument("--save-layer-toolpath", type=int, help="Save specific layer toolpath to CSV (layer number)")
    parser.add_argument("--save-3d-toolpath", action="store_true", help="Save complete 3D toolpath to CSV")
    
    # Configuration file
    parser.add_argument("--config", help="Load configuration from JSON file")
    parser.add_argument("--save-config", help="Save current configuration to JSON file")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress most output")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = ToolpathConfig.from_json(args.config)
            
            # Override config with command line arguments if provided
            if getattr(args, 'mesh_file', None):
                config.mesh_file = args.mesh_file
            if getattr(args, 'non_planar_surface', None):
                config.non_planar_surface_file = args.non_planar_surface
                
        else:
            # Require mesh_file if no config provided
            if not getattr(args, 'mesh_file', None):
                parser.error("--mesh-file is required when not using --config")
                
            # Create configuration from command line arguments
            config = ToolpathConfig(
                mesh_file=args.mesh_file,
                non_planar_surface_file=args.non_planar_surface,
                layer_height=args.layer_height,
                add_layers=args.add_layers,
                slicing_axis=args.slicing_axis,
                rotate_180=args.rotate_180,
                strategy=args.strategy,
                tool_width=args.tool_width,
                hatch_distance=args.hatch_distance,
                fly_in_out=args.fly_in_out,
                fly_distance=args.fly_distance,
                raster_angle=args.raster_angle,
                corner_radius=args.corner_radius,
                enable_turbine_repair=args.enable_turbine_repair,
                extension_length=args.extension_length,
                spline_smoothing=args.spline_smoothing,
                num_circles=args.num_circles,
                circle_resolution=args.circle_resolution,
                circle_tolerance=args.circle_tolerance,
                curvature_threshold=args.curvature_threshold,
                enable_evaluation=not args.disable_evaluation,
                enable_visualization=not args.disable_visualization,
                export_csv=not args.no_export,
                output_dir=args.output_dir,
                debug_layer=args.debug_layer,
                visualize_all_layers=args.visualize_all,
                save_plots=args.save_plots,
                show_3d_slices=args.show_3d_slices,
                show_boundary_extension=args.show_boundary_extension,
                show_hybrid_process=args.show_hybrid_process,
                show_layer_toolpath=args.show_layer_toolpath,
                show_3d_toolpath=args.show_3d_toolpath,
                save_layer_toolpath=args.save_layer_toolpath,
                save_3d_toolpath=args.save_3d_toolpath
            )
        
        # Save configuration if requested
        if args.save_config:
            config.save_json(args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
        
        # Create and run toolpath generator
        generator = ToolpathGenerator(config)
        generator.run()
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()