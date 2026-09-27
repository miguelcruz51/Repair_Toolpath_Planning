import numpy as np
import open3d as o3d

# Load the damaged part
damaged_mesh = o3d.io.read_triangle_mesh('/home/miguel-cruz/Documents/AS4RCSD_K_versionn/Data/A2.STL')
vertices = np.asarray(damaged_mesh.vertices)

# Calculate bounding box
min_pt = np.min(vertices, axis=0)
max_pt = np.max(vertices, axis=0)
center = (min_pt + max_pt) / 2
size = max_pt - min_pt

print('DAMAGED PART (A2.STL) BOUNDING BOX:')
print('=' * 50)
print(f'Min corner: X={min_pt[0]:.3f}, Y={min_pt[1]:.3f}, Z={min_pt[2]:.3f}')
print(f'Max corner: X={max_pt[0]:.3f}, Y={max_pt[1]:.3f}, Z={max_pt[2]:.3f}')
print(f'Center:     X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}')
print(f'Size:       X={size[0]:.3f}, Y={size[1]:.3f}, Z={size[2]:.3f}')
print()
print('REPAIR VOLUME (A_vol_to_repair_ref.STL) BOUNDING BOX:')
print('=' * 50)

# Load repair volume
repair_mesh = o3d.io.read_triangle_mesh('Data/A_vol_to_repair_ref.STL')
repair_vertices = np.asarray(repair_mesh.vertices)

repair_min = np.min(repair_vertices, axis=0)
repair_max = np.max(repair_vertices, axis=0)
repair_center = (repair_min + repair_max) / 2
repair_size = repair_max - repair_min

print(f'Min corner: X={repair_min[0]:.3f}, Y={repair_min[1]:.3f}, Z={repair_min[2]:.3f}')
print(f'Max corner: X={repair_max[0]:.3f}, Y={repair_max[1]:.3f}, Z={repair_max[2]:.3f}')
print(f'Center:     X={repair_center[0]:.3f}, Y={repair_center[1]:.3f}, Z={repair_center[2]:.3f}')
print(f'Size:       X={repair_size[0]:.3f}, Y={repair_size[1]:.3f}, Z={repair_size[2]:.3f}')