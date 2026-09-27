import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk

# Set control flags for different visualizations/analyses
visualize_models_flag = True    
visualize_slices_flag = True
analyze_specific_layer_flag = True  



def align_repair_volume(damaged_model_path, repair_volume_path):
    print('Aligning...')

    # Load the models
    damaged_model = pv.read(damaged_model_path)
    repair_volume = pv.read(repair_volume_path)

    # Access the center attribute (as a property, not a method)
    damaged_center = damaged_model.center
    repair_center = repair_volume.center

    damaged_bounds = damaged_model.bounds
    # Calculate Length, Width, and Height from the bounding box
    damaged_L = damaged_bounds[1] - damaged_bounds[0]  # X dimension (Length)
    damaged_W = damaged_bounds[3] - damaged_bounds[2]  # Y dimension (Width)
    damaged_H = damaged_bounds[5] - damaged_bounds[4]  # Z dimension (Height)
    trans_x = damaged_L / 2
    trans_y = damaged_W / 2

    fixed_translation = [trans_x, trans_y, 0.0]  # Translation in mm

    repair_volume.translate(fixed_translation, inplace=True)
    # print(f"Applied fixed translation of {fixed_translation}")

    return damaged_model, repair_volume


# Function to visualize damaged and repair volumes
def visualize_models(damaged_model, repair_volume):
    if visualize_models_flag:

        
        # Create a plotter
        plotter = pv.Plotter()
        plotter.add_mesh(damaged_model, color="red", opacity=0.7, label="Damaged Model")
        plotter.add_mesh(repair_volume, color="blue", opacity=0.5, label="Repair Volume")

        # Add legend and display
        plotter.add_legend()
        plotter.show()

def slice_model(repair_volume, layer_height):
    """
    Slices the repair volume based on the specified layer height.
    Skips empty slices and proceeds to the next height until a valid slice is found.
    """
    # Get the bounds of the model to determine the Z-range
    min_z, max_z = repair_volume.bounds[4], repair_volume.bounds[5]
    # print(f"Repair volume bounds: {repair_volume.bounds}")
    
    # Generate a list of slice heights from min_z to max_z with the given layer height
    slice_heights = np.arange(min_z, max_z, layer_height)
    # print(f"Slicing at heights: {slice_heights}")

    slices = []
    
    for height in slice_heights:
        # Create a slicing plane at the current height (perpendicular to Z-axis)
        plane_origin = (repair_volume.center[0], repair_volume.center[1], height)
        
        # Slice the repair volume at the current height using the z-axis
        slice = repair_volume.slice(normal='z', origin=plane_origin)
        
        # Check if the slice is empty
        if slice.n_points == 0:
            print(f"Warning: Slice at height {height} mm is empty. Skipping.")
            continue  # Skip empty slices and move to the next height
        
        # If slice has points, add it to the list
        slices.append(slice)
    
    return slices, slice_heights
    

def visualize_slices(slices, slice_heights):
    if visualize_slices_flag:
    

        plotter = pv.Plotter()

        # Visualize each slice and display the corresponding height
        for slice, height in zip(slices, slice_heights):
            plotter.add_mesh(slice, color='lightblue', opacity=0.7, line_width=3)
            plotter.add_text(f'Layer Height: {height:.2f} mm', position='upper_left', color='black', font_size=10)

        plotter.show()

def visualize_layer_top_view(slice, layer_height):
    """
    Visualizes a specific slice in 2D top view.
    The view will be from above (looking down along the Z-axis).
    """
    plotter = pv.Plotter()

    # Plot the slice in top view (viewing along the Z-axis)
    plotter.add_mesh(slice, color='lightblue', opacity=0.7, line_width=3)  # Increase the line_width for thicker lines

    # Set the camera to look from the top (Z-axis direction)
    plotter.view_isometric()  # Top-down view
    
    # Add the text to display the layer height or number
    plotter.add_text(f"Layer at {layer_height:.2f} mm", position='upper_left', color='black', font_size=12)

    plotter.show()

def generate_contour_toolpath(slice):
    """
    Generates a contour (outline) toolpath by tracing the outer boundary of the slice.
    """

    # Extract the boundary of the slice as a polyline
    contour = slice.extract_all_edges()
    
    # Generate the toolpath from the contour (polyline)
    toolpath = [(p[0], p[1]) for p in contour.points]
    
    print(f"Is slice vtkPolyData? {isinstance(slice, vtk.vtkPolyData)}")
    print(f"Number of Points in Slice: {slice.GetNumberOfPoints()}")
    print(f"Bounds of Slice: {slice.GetBounds()}")

    return toolpath


def generate_raster_toolpath(slice, tool_width):
    """
    Generates a raster (zigzag) toolpath for the interior of a slice.
    """
    min_x, max_x, min_y, max_y = slice.bounds[0], slice.bounds[1], slice.bounds[2], slice.bounds[3]
    step_size = tool_width  # Tool width defines the distance between passes

    toolpath = []
    y_pos = min_y
    direction = 1  # Direction: 1 for left-to-right, -1 for right-to-left

    while y_pos < max_y:
        x_start, x_end = min_x, max_x
        if direction == 1:
            toolpath.append([(x_start, y_pos), (x_end, y_pos)])
        else:
            toolpath.append([(x_end, y_pos), (x_start, y_pos)])

        # Move to the next line (increment y)
        y_pos += step_size
        direction *= -1  # Reverse direction (zigzag)

        
    print (toolpath)
    return toolpath

import numpy as np

def generate_spiral_toolpath(slice, tool_width, num_turns=10):
    """
    Generates a spiral toolpath that starts from the outer boundary and moves inward.
    """
    # Get the center and the bounds of the slice
    center_x, center_y = slice.center[0], slice.center[1]
    min_x, max_x, min_y, max_y = slice.bounds[0], slice.bounds[1], slice.bounds[2], slice.bounds[3]

    # Number of steps in the spiral
    steps = 100

    toolpath = []
    radius_step = min((max_x - min_x), (max_y - min_y)) / (2 * num_turns)
    
    for i in range(steps):
        angle = 2 * np.pi * i / steps
        radius = radius_step * i
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        toolpath.append((x, y))

    return toolpath

def rotate_toolpath(toolpath, angle):
    """
    Rotates the toolpath by the specified angle (in degrees).
    """
    angle_rad = np.radians(angle)
    rotated_toolpath = []

    for path in toolpath:
        rotated_path = []
        for (x, y) in path:
            # Apply rotation matrix
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            rotated_path.append((x_rot, y_rot))
        rotated_toolpath.append(rotated_path)

    return rotated_toolpath
    
def choose_toolpath_strategy(slice, tool_width):
    """
    Allows the user to choose the toolpath strategy.
    """
    print("Choose the toolpath strategy:")
    print("1. Contour Path")
    print("2. Raster Path")
    print("3. Spiral Path")
    print("4. Grid Path")

    choice = int(input("Enter the number corresponding to the strategy: "))

    if choice == 1:
        return generate_contour_toolpath(slice)
    elif choice == 2:
        return generate_raster_toolpath(slice, tool_width)
    elif choice == 3:
        return generate_spiral_toolpath(slice, tool_width)
    elif choice == 4:
        return generate_grid_toolpath(slice, tool_width)
    else:
        print("Invalid choice. Defaulting to Raster Path.")
        return generate_raster_toolpath(slice, tool_width)

def apply_orientation_change(toolpath, layer_number, rotation_angle=90):
    """
    Applies a rotation to the toolpath for every alternate layer.
    """
    if layer_number % 2 == 0:  # Rotate every other layer
        print(f"Rotating layer {layer_number} by {rotation_angle} degrees.")
        return rotate_toolpath(toolpath, rotation_angle)
    else:
        return toolpath


def generate_grid_toolpath(slice, tool_width):
    """
    Generates a grid-like toolpath by sweeping horizontally and vertically.
    """
    min_x, max_x, min_y, max_y = slice.bounds[0], slice.bounds[1], slice.bounds[2], slice.bounds[3]
    step_size = tool_width  # Tool width defines the spacing of the grid lines

    # Generate vertical grid lines
    vertical_lines = []
    for x_pos in np.arange(min_x, max_x, step_size):
        vertical_lines.append([(x_pos, min_y), (x_pos, max_y)])

    # Generate horizontal grid lines
    horizontal_lines = []
    for y_pos in np.arange(min_y, max_y, step_size):
        horizontal_lines.append([(min_x, y_pos), (max_x, y_pos)])

    # Combine the grid lines
    toolpath = vertical_lines + horizontal_lines

    return toolpath

def generate_toolpath_by_strategy(slice, tool_width, strategy):
    """
    Generates the toolpath for a slice based on the selected strategy.
    """
    if strategy == 1:
        return generate_contour_toolpath(slice)
    elif strategy == 2:
        return generate_raster_toolpath(slice, tool_width)
    elif strategy == 3:
        return generate_spiral_toolpath(slice, tool_width)
    elif strategy == 4:
        return generate_grid_toolpath(slice, tool_width)
    else:
        print("Invalid strategy. Defaulting to Raster Path.")
        return generate_raster_toolpath(slice, tool_width)

def visualize_toolpath_3d(toolpaths, slice_heights):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import matplotlib
    matplotlib.use('TkAgg')  # Ensure GUI-compatible backend

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, toolpath in enumerate(toolpaths):
        layer_height = slice_heights[idx]
        flattened_toolpath = [point for sublist in toolpath for point in sublist]
        x = [point[0] for point in flattened_toolpath]
        y = [point[1] for point in flattened_toolpath]
        z = [layer_height] * len(flattened_toolpath)

        ax.plot(x, y, z, label=f"Layer {idx + 1}")
        ax.scatter(x, y, z, s=5)

    ax.set_title("3D Toolpath Visualization")
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_zlabel("Z-axis (mm)")
    plt.legend()
    plt.show(block=True)  # Ensure the plot stays open


def visualize_toolpath_2d(layer_toolpaths, layer_height, layer_index):
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.use('TkAgg')  # Ensure GUI-compatible backend

    fig, ax = plt.subplots()

    for toolpath in layer_toolpaths:
        x = [point[0] for point in toolpath]
        y = [point[1] for point in toolpath]
        ax.plot(x, y, marker='o')

    ax.set_title(f"2D Toolpath for Layer {layer_index + 1} (Height: {layer_height:.2f} mm)")
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show(block=True)  # Ensure the plot stays open



def main():
    print("=== Additive Manufacturing Repair Toolpath Generation ===")

    # Input file paths
    damaged_model_path = "/home/miguel-cruz/Documents/AS4RCSD/Dataset/Part1.STL"
    repair_volume_path = "/home/miguel-cruz/Documents/AS4RCSD/Dataset/Part1 V.STL"

    # Align and repair volume
    damaged_model, repair_volume = align_repair_volume(damaged_model_path, repair_volume_path)

    # Visualize models
    print("Visualizing models...")
    visualize_models(damaged_model, repair_volume)

    # Ask the user for the desired layer height
    layer_height = float(input("Enter the layer height (in mm): "))

    # Slice the repair volume
    slices, slice_heights = slice_model(repair_volume, layer_height)

    # Check if any slices were generated
    if not slices:
        print("No valid slices were generated. Exiting...")
        return

    # Visualize the slices
    visualize_slices(slices, slice_heights)

    # Ask the user for the laser width spot diameter
    tool_width = float(input("Enter the laser width spot diameter (in mm): "))

    # Global toolpath strategy selection
    print("\nChoose a toolpath strategy to apply to all layers:")
    print("1. Contour Path")
    print("2. Raster Path")
    print("3. Spiral Path")
    print("4. Grid Path")
    global_strategy = int(input("Enter the number corresponding to the strategy: "))

    # Ask if rotation between layers is needed
    rotate_layers = input("Do you want to apply rotation between layers? (yes/no): ").strip().lower() == "yes"
    rotation_angle = 0
    if rotate_layers:
        rotation_angle = float(input("Enter the rotation angle (in degrees): "))

    # Apply the global toolpath strategy to all layers
    toolpaths = []
    for idx, slice in enumerate(slices):
        toolpath = generate_toolpath_by_strategy(slice, tool_width, global_strategy)

        # Apply rotation if required
        if rotate_layers:
            toolpath = apply_orientation_change(toolpath, idx + 1, rotation_angle)
        toolpaths.append(toolpath)

    # Ask if any specific layer needs a different toolpath strategy
    while True:
        change_layer = input("\nDo you need to change the toolpath strategy for any specific layer? (yes/no): ").strip().lower() == "yes"
        if not change_layer:
            break

        layer_idx = int(input(f"Enter the layer number (1 to {len(slices)}): ")) - 1
        if 0 <= layer_idx < len(slices):
            print("\nChoose a new toolpath strategy for this layer:")
            print("1. Contour Path")
            print("2. Raster Path")
            print("3. Spiral Path")
            print("4. Grid Path")
            specific_strategy = int(input("Enter the number corresponding to the strategy: "))
            toolpaths[layer_idx] = generate_toolpath_by_strategy(slices[layer_idx], tool_width, specific_strategy)
            print(f"Updated toolpath for layer {layer_idx + 1}.")
        else:
            print("Invalid layer number. Please try again.")

    # Ask if additional layers need to be added above the current layers
    add_extra_layers = input("\nDo you want to add extra layers above the current layers? (yes/no): ").strip().lower() == "yes"
    if add_extra_layers:
        num_extra_layers = int(input("Enter the number of extra layers to add: "))
        offset = float(input("Enter the height offset for each new layer (in mm): "))
        for i in range(num_extra_layers):
            new_height = slice_heights[-1] + (i + 1) * offset
            slices.append(slices[-1])  # Copy the last slice
            slice_heights = np.append(slice_heights, new_height)  # Append new height
            toolpaths.append(toolpaths[-1])  # Use the same toolpath as the last layer
            print(f"Added extra layer at height {new_height:.2f} mm.")

    print("\nAll layers processed.")

    # Visualize the toolpath in 3D
    visualize_toolpath_3d(toolpaths, slice_heights)

    # Allow the user to view a specific layer in 2D
    while True:
        view_layer = input("\nDo you want to view the toolpath for a specific layer in 2D? (yes/no): ").strip().lower() == "yes"
        if not view_layer:
            break

        layer_idx = int(input(f"Enter the layer number (1 to {len(slices)}): ")) - 1
        if 0 <= layer_idx < len(toolpaths):
            visualize_toolpath_2d(toolpaths[layer_idx], slice_heights[layer_idx], layer_idx)
        else:
            print("Invalid layer number. Please try again.")

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()
