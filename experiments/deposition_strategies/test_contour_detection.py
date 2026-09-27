#!/usr/bin/env python3
"""
Test script to demonstrate contour strategy detection and optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcode_dev import GCodeConverter

def test_contour_detection():
    """Test the contour detection functionality"""
    
    # Sample G-code representing a contour strategy (simplified version of your example)
    contour_gcode = [
        "; Layer 0",
        # First contour (outer)
        "G01 X357.552 Y73.104 Z0.000 F1500",
        "G01 X357.761 Y72.921 Z0.000 F1500", 
        "G01 X357.844 Y72.809 Z0.000 F1500",
        "G01 X358.459 Y71.866 Z0.000 F1500",
        "G01 X358.552 Y71.662 Z0.000 F1500",
        "G01 X358.594 Y71.498 Z0.000 F1500",
        "G01 X358.636 Y71.262 Z0.000 F1500",
        "G01 X358.637 Y70.464 Z0.000 F1500",
        "G01 X357.552 Y73.104 Z0.000 F1500",  # Close contour
        
        # Link to second contour (this should be detected as linking move)
        "G01 X348.152 Y64.834 Z0.000 F1500",  # Large jump - linking move
        
        # Second contour (middle)
        "G01 X348.062 Y65.832 Z0.000 F1500",
        "G01 X349.144 Y69.900 Z0.000 F1500",
        "G01 X349.480 Y70.456 Z0.000 F1500",
        "G01 X349.973 Y70.879 Z0.000 F1500",
        "G01 X352.361 Y72.422 Z0.000 F1500",
        "G01 X352.651 Y72.473 Z0.000 F1500",
        "G01 X348.152 Y64.834 Z0.000 F1500",  # Close contour
        
        # Link to third contour (another linking move)
        "G01 X348.909 Y65.341 Z0.000 F1500",  # Another jump - linking move
        
        # Third contour (inner)
        "G01 X348.871 Y65.763 Z0.000 F1500",
        "G01 X349.888 Y69.584 Z0.000 F1500",
        "G01 X350.099 Y69.933 Z0.000 F1500",
        "G01 X350.453 Y70.237 Z0.000 F1500",
        "G01 X352.660 Y71.662 Z0.000 F1500",
        "G01 X348.909 Y65.341 Z0.000 F1500",  # Close contour
        
        "; Layer 1",
        "G01 X100.0 Y100.0 Z0.800 F1500",
        "G01 X110.0 Y100.0 Z0.800 F1500",
    ]
    
    converter = GCodeConverter()
    converter.gcode_lines = contour_gcode
    
    print("Contour Strategy Detection Test")
    print("=" * 50)
    
    # Test contour detection
    layer_num = 0
    link_distance_threshold = 5.0  # 5mm threshold for detecting linking moves
    
    print(f"Analyzing Layer {layer_num} for contour strategy...")
    print(f"Link distance threshold: {link_distance_threshold} mm")
    print("-" * 50)
    
    contour_analysis = converter.detect_contour_strategy(layer_num, link_distance_threshold)
    
    if "error" in contour_analysis:
        print(f"Error: {contour_analysis['error']}")
        return
    
    print(f"Strategy Detected: {contour_analysis['strategy_detected']}")
    print(f"Total Moves: {contour_analysis['total_moves']}")
    print(f"Contours Found: {contour_analysis['contours_detected']}")
    print(f"Linking Moves: {contour_analysis.get('total_link_moves', 0)}")
    
    if contour_analysis.get('contours'):
        print(f"Average Contour Size: {contour_analysis.get('avg_contour_size', 0):.1f} moves")
        print(f"Contour Size Range: {contour_analysis.get('min_contour_size', 0)} - {contour_analysis.get('max_contour_size', 0)} moves")
    
    # Show detailed contour breakdown
    print("\nContour Breakdown:")
    print("-" * 30)
    contours = contour_analysis.get('contours', [])
    link_moves = contour_analysis.get('link_moves', [])
    
    for i, contour in enumerate(contours):
        print(f"Contour {i+1}: {len(contour)} moves (indices {contour[0]}-{contour[-1]})")
    
    if link_moves:
        print(f"\nLinking Move Indices: {link_moves}")
        moves = contour_analysis.get('moves', [])
        print("Linking Move Details:")
        for idx in link_moves:
            if idx < len(moves):
                move = moves[idx]
                print(f"  Index {idx}: {move['command']} X{move['x']:.3f} Y{move['y']:.3f}")
    
    # Test optimization
    if contour_analysis['strategy_detected'] == 'contour_offset':
        print("\n" + "=" * 50)
        print("Testing Contour Optimization")
        print("=" * 50)
        
        print("G-code BEFORE optimization:")
        layer_lines = converter.get_gcode_lines_for_layer(layer_num)
        for i, line in enumerate(layer_lines):
            if not line.strip().startswith(';'):
                parsed = converter.parse_gcode_line(line)
                if parsed['command']:
                    marker = " ← LINK" if i-1 in link_moves else ""  # Adjust for comment line
                    print(f"  {i}: {parsed['command']} - {line.strip()}{marker}")
        
        # Apply optimization
        optimization_result = converter.optimize_contour_links(layer_num, contour_analysis)
        
        if "error" not in optimization_result:
            print(f"\n✓ Optimization completed!")
            print(f"Converted {optimization_result['converted_moves']} of {optimization_result['total_link_moves']} linking moves")
            
            print("\nG-code AFTER optimization:")
            layer_lines_after = converter.get_gcode_lines_for_layer(layer_num)
            for i, line in enumerate(layer_lines_after):
                if not line.strip().startswith(';'):
                    parsed = converter.parse_gcode_line(line)
                    if parsed['command']:
                        marker = " ← OPTIMIZED" if i-1 in link_moves and parsed['command'] == 'G00' else ""
                        print(f"  {i}: {parsed['command']} - {line.strip()}{marker}")
        else:
            print(f"✗ Optimization failed: {optimization_result['error']}")
    
    print("\n" + "=" * 50)
    print("Contour detection test completed!")
    print("\nKey Features Demonstrated:")
    print("• Automatic contour strategy detection")
    print("• Linking move identification based on distance threshold")
    print("• Contour statistics and analysis")
    print("• Automatic optimization of linking moves (G01 → G00)")
    print("• Feedrate optimization for travel moves")

if __name__ == "__main__":
    test_contour_detection()