#!/usr/bin/env python3
"""
Simple STL coordinate analysis without visualization.
Run this to understand the coordinate systems.
"""

import open3d as o3d
import numpy as np

def analyze_stl(filepath, name):
    """Analyze an STL file."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
        
        if not mesh.has_vertices():
            print("ERROR: No vertices!")
            return None
        
        vertices = np.asarray(mesh.vertices)
        min_pt = np.min(vertices, axis=0)
        max_pt = np.max(vertices, axis=0)
        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt
        
        print(f"\nBounding Box:")
        print(f"  Min:    X={min_pt[0]:8.3f}, Y={min_pt[1]:8.3f}, Z={min_pt[2]:8.3f}")
        print(f"  Max:    X={max_pt[0]:8.3f}, Y={max_pt[1]:8.3f}, Z={max_pt[2]:8.3f}")
        print(f"  Center: X={center[0]:8.3f}, Y={center[1]:8.3f}, Z={center[2]:8.3f}")
        print(f"  Size:   X={size[0]:8.3f}, Y={size[1]:8.3f}, Z={size[2]:8.3f}")
        print(f"\nVertices: {len(vertices)}, Triangles: {len(mesh.triangles)}")
        
        return {
            'min': min_pt,
            'max': max_pt,
            'center': center,
            'size': size
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def check_relationship(info1, info2, name1, name2):
    """Check relationship between two meshes."""
    print(f"\n{'='*70}")
    print(f"RELATIONSHIP: {name1} vs {name2}")
    print(f"{'='*70}")
    
    # Check overlap
    overlap_x = not (info1['max'][0] < info2['min'][0] or info2['max'][0] < info1['min'][0])
    overlap_y = not (info1['max'][1] < info2['min'][1] or info2['max'][1] < info1['min'][1])
    overlap_z = not (info1['max'][2] < info2['min'][2] or info2['max'][2] < info1['min'][2])
    overlap = overlap_x and overlap_y and overlap_z
    
    print(f"\nBounding Box Overlap:")
    print(f"  X: {'YES' if overlap_x else 'NO'}")
    print(f"  Y: {'YES' if overlap_y else 'NO'}")
    print(f"  Z: {'YES' if overlap_z else 'NO'}")
    print(f"  Overall: {'✓ OVERLAPPING' if overlap else '✗ SEPARATE'}")
    
    # Distance between centers
    distance = np.linalg.norm(info1['center'] - info2['center'])
    print(f"\nDistance between centers: {distance:.3f} mm")
    
    # Offset between origins
    offset = info2['min'] - info1['min']
    print(f"\nOffset ({name1} origin → {name2} origin):")
    print(f"  X: {offset[0]:+8.3f}")
    print(f"  Y: {offset[1]:+8.3f}")
    print(f"  Z: {offset[2]:+8.3f}")
    
    # Check if repair is inside damaged
    if overlap:
        inside_x = info2['min'][0] >= info1['min'][0] and info2['max'][0] <= info1['max'][0]
        inside_y = info2['min'][1] >= info1['min'][1] and info2['max'][1] <= info1['max'][1]
        inside_z = info2['min'][2] >= info1['min'][2] and info2['max'][2] <= info1['max'][2]
        
        if inside_x and inside_y and inside_z:
            print(f"\n✓ {name2} is COMPLETELY INSIDE {name1}")
        else:
            print(f"\n⚠ {name2} PARTIALLY overlaps {name1}")
            if not inside_x: print(f"   - Extends beyond in X")
            if not inside_y: print(f"   - Extends beyond in Y")
            if not inside_z: print(f"   - Extends beyond in Z")
    
    return overlap

if __name__ == "__main__":
    print("="*70)
    print("STL COORDINATE ANALYSIS")
    print("="*70)
    
    damaged_path = "Data/A2.STL"
    repair_path = "Data/A_vol_to_repair_ref.STL"
    
    damaged_info = analyze_stl(damaged_path, "DAMAGED PART")
    repair_info = analyze_stl(repair_path, "REPAIR VOLUME")
    
    if damaged_info and repair_info:
        overlap = check_relationship(damaged_info, repair_info, "Damaged Part", "Repair Volume")
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        if overlap:
            print("\n✓ STL files are correctly positioned!")
            print("  The repair volume overlaps with the damaged part.")
            print("\n  This means the issue is likely with:")
            print("  - Toolpath generation in AS4RCSD4")
            print("  - Rotation/transformation during export")
            print("  - G-code conversion")
        else:
            print("\n✗ STL files are NOT aligned!")
            print("  The repair volume is separate from the damaged part.")
            print("\n  To fix, apply this offset to repair volume:")
            offset = repair_info['min'] - damaged_info['min']
            print(f"    X: {-offset[0]:+.3f}")
            print(f"    Y: {-offset[1]:+.3f}")
            print(f"    Z: {-offset[2]:+.3f}")
    
    print("\n" + "="*70)
