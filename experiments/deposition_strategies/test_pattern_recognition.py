#!/usr/bin/env python3
"""
Test script to demonstrate the new pattern recognition feature
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gcode_dev import GCodeConverter

def test_pattern_recognition():
    """Test the refined pattern recognition functionality"""
    
    # Sample G-code with structured patterns
    sample_gcode = [
        "; Layer 0",
        "G00 X347.823 Y64.103 Z0.000 F3000",  # Initial 1
        "G00 X347.823 Y64.103 Z0.000 F3000",  # Initial 2
        "G01 X353.127 Y64.103 Z0.000 F1500",  # Middle alternating start
        "G00 X347.342 Y64.903 Z0.000 F3000",
        "G01 X354.630 Y64.903 Z0.000 F1500",
        "G00 X347.270 Y65.703 Z0.000 F3000",
        "G01 X355.633 Y65.703 Z0.000 F1500",  # Middle alternating end
        "G00 X357.774 Y72.903 Z0.000 F3000",  # Final 1
        "G00 X357.774 Y72.903 Z0.000 F3000",  # Final 2
        "; Layer 1",
        "G01 X345.495 Y63.328 Z0.800 F1500",
        "G01 X353.109 Y63.328 Z0.800 F1500",
        "G01 X344.688 Y64.128 Z0.800 F1500",
        "G01 X354.712 Y64.128 Z0.800 F1500",
        "G01 X344.269 Y64.928 Z0.800 F1500",
        "G01 X355.807 Y64.928 Z0.800 F1500",
        "G01 X343.926 Y65.728 Z0.800 F1500",
        "; Layer 2",
        "G00 X100.0 Y100.0 Z1.600 F3000",
        "G00 X110.0 Y100.0 Z1.600 F3000",
        "G00 X120.0 Y100.0 Z1.600 F3000",
        "G00 X130.0 Y100.0 Z1.600 F3000",
    ]
    
    converter = GCodeConverter()
    converter.gcode_lines = sample_gcode
    
    print("Refined Pattern Recognition Test")
    print("=" * 50)
    
    # Test pattern analysis with initial/final preservation
    initial_keep = 2
    final_keep = 2
    
    print(f"Testing with Initial Keep: {initial_keep}, Final Keep: {final_keep}")
    print("=" * 50)
    
    # Test pattern analysis for each layer
    layers = [0, 1, 2]
    
    for layer_num in layers:
        print(f"\nAnalyzing Layer {layer_num}:")
        print("-" * 30)
        
        pattern_info = converter.analyze_layer_pattern(layer_num, initial_keep, final_keep)
        
        if "error" in pattern_info:
            print(f"Error: {pattern_info['error']}")
            continue
        
        print(f"Total Commands: {pattern_info['total_commands']}")
        print(f"Initial {initial_keep} Commands: {' → '.join(pattern_info['initial_commands'])}")
        print(f"Final {final_keep} Commands: {' → '.join(pattern_info['final_commands'])}")
        print(f"Middle Commands ({len(pattern_info['middle_commands'])}): {' → '.join(pattern_info['middle_commands'])}")
        print(f"Middle Pattern Type: {pattern_info['middle_pattern_type']}")
        
        if pattern_info.get('middle_alternating'):
            middle_sequence = pattern_info.get('middle_sequence', [])
            print(f"Middle Alternating Pattern: {' → '.join(middle_sequence)}")
        
        # Show complete structure
        commands = pattern_info.get('commands', [])
        print(f"Complete Structure:")
        for i, cmd in enumerate(commands):
            section = ""
            if i < initial_keep:
                section = " [INITIAL]"
            elif i >= len(commands) - final_keep:
                section = " [FINAL]"
            else:
                section = " [MIDDLE]"
            print(f"  {i+1}. {cmd}{section}")
    
    print("\n" + "=" * 50)
    print("Testing Refined Pattern Application")
    print("=" * 50)
    
    # Test applying Layer 0's pattern to Layer 1
    layer_0_pattern = converter.analyze_layer_pattern(0, initial_keep, final_keep)
    if "error" not in layer_0_pattern:
        print(f"\nApplying Layer 0's structured pattern to Layer 1...")
        print(f"Reference Pattern Structure:")
        print(f"  Initial: {' → '.join(layer_0_pattern['initial_commands'])}")
        print(f"  Middle: {layer_0_pattern['middle_pattern_type']} ({' → '.join(layer_0_pattern['middle_sequence'])})")
        print(f"  Final: {' → '.join(layer_0_pattern['final_commands'])}")
        
        # Show Layer 1 before modification
        layer_1_before = converter.get_gcode_lines_for_layer(1)
        print(f"\nLayer 1 BEFORE pattern application:")
        gcode_commands_before = []
        for line in layer_1_before:
            if not line.strip().startswith(';'):
                parsed = converter.parse_gcode_line(line)
                if parsed['command']:
                    gcode_commands_before.append(parsed['command'])
        
        for i, cmd in enumerate(gcode_commands_before):
            print(f"  {i+1}. {cmd}")
        
        # Apply pattern
        success = converter.apply_pattern_to_layer(1, layer_0_pattern)
        
        if success:
            print(f"\n✓ Structured pattern applied successfully!")
            
            # Show Layer 1 after modification
            layer_1_after = converter.get_gcode_lines_for_layer(1)
            print(f"\nLayer 1 AFTER pattern application:")
            gcode_commands_after = []
            for line in layer_1_after:
                if not line.strip().startswith(';'):
                    parsed = converter.parse_gcode_line(line)
                    if parsed['command']:
                        gcode_commands_after.append(parsed['command'])
            
            for i, cmd in enumerate(gcode_commands_after):
                section = ""
                if i < initial_keep:
                    section = " [INITIAL - preserved]"
                elif i >= len(gcode_commands_after) - final_keep:
                    section = " [FINAL - preserved]"
                else:
                    section = " [MIDDLE - patterned]"
                print(f"  {i+1}. {cmd}{section}")
        else:
            print(f"\n✗ Failed to apply pattern")
    
    print("\n" + "=" * 50)
    print("Refined pattern recognition test completed!")
    print("Key improvements:")
    print("• Preserves initial and final commands exactly")
    print("• Applies pattern only to middle section")
    print("• Handles layers with different command counts")
    print("• Provides detailed structure analysis")

if __name__ == "__main__":
    test_pattern_recognition()