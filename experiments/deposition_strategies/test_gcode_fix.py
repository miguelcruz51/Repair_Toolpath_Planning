#!/usr/bin/env python3
"""
Test script to verify the G-code plotting fix for mixed G01/G00 commands
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcode_dev import GCodeConverter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_mixed_gcode():
    """Test the plotting with mixed G01/G00 commands"""
    
    # Test case 1: Mixed G01/G00 (your problematic case)
    mixed_gcode = [
        "; Layer 0",
        "G00 X347.823 Y64.103 Z0.000 F3000",
        "G01 X353.127 Y64.103 Z0.000 F1500",
        "G00 X347.342 Y64.903 Z0.000 F3000",
        "G01 X354.630 Y64.903 Z0.000 F1500",
        "G00 X347.270 Y65.703 Z0.000 F3000",
        "G01 X355.633 Y65.703 Z0.000 F1500",
        "G00 X347.413 Y66.503 Z0.000 F3000",
        "G01 X356.474 Y66.503 Z0.000 F1500",
        "G00 X357.774 Y72.903 Z0.000 F3000",
        "; Layer 1",
        "G00 X345.495 Y63.328 Z0.800 F3000",
        "G01 X345.495 Y63.328 Z0.800 F1500",
        "G01 X353.109 Y63.328 Z0.800 F1500"
    ]
    
    # Test case 2: All G01 (should work)
    all_g01_gcode = [
        "; Layer 0",
        "G01 X347.823 Y64.103 Z0.000 F1500",
        "G01 X353.127 Y64.103 Z0.000 F1500",
        "G01 X347.342 Y64.903 Z0.000 F1500",
        "G01 X354.630 Y64.903 Z0.000 F1500"
    ]
    
    # Test case 3: All G00 (should work)
    all_g00_gcode = [
        "; Layer 0",
        "G00 X347.823 Y64.103 Z0.000 F3000",
        "G00 X353.127 Y64.103 Z0.000 F3000",
        "G00 X347.342 Y64.903 Z0.000 F3000",
        "G00 X354.630 Y64.903 Z0.000 F3000"
    ]
    
    test_cases = [
        ("Mixed G01/G00", mixed_gcode),
        ("All G01", all_g01_gcode),
        ("All G00", all_g00_gcode)
    ]
    
    converter = GCodeConverter()
    
    for name, gcode_lines in test_cases:
        print(f"\nTesting: {name}")
        print(f"G-code lines: {len(gcode_lines)}")
        
        # Set the gcode lines
        converter.gcode_lines = gcode_lines
        
        # Parse and get coordinates
        converter.parse_gcode()
        x_coords, y_coords, z_coords, commands = converter.get_gcode_coordinates()
        
        print(f"Parsed coordinates: {len(x_coords)} points")
        print(f"Commands: {commands}")
        
        # Check if we have valid coordinates
        if x_coords:
            print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
            print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
            print(f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
            print("✓ Coordinates extracted successfully")
        else:
            print("✗ No coordinates extracted - this indicates a problem")
    
    print("\n" + "="*50)
    print("Test completed. If all test cases show coordinates,")
    print("the fix is working correctly!")

if __name__ == "__main__":
    test_mixed_gcode()