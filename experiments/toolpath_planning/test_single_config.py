#!/usr/bin/env python3
"""
Test a single configuration to debug batch evaluation issues.
"""

import json
import subprocess
import sys
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "mesh_file": "/home/miguelcruz/Documents/AS4RCSD/Data/Model_A_R.STL",
    "non_planar_surface_file": "/home/miguelcruz/Documents/AS4RCSD/Data/Model_A_S.STL",
    "layer_height": 1,
    "strategy": 1,
    "tool_width": 2.0,
    "hatch_distance": 1.0,
    "grid_size": 50,
    "enable_evaluation": True,
    "evaluate": "all",
    "enable_visualization": False,
    "save_evaluation_plots": False,
    "use_parallel_evaluation": True,
    "num_workers": -1,
    "export_csv": True,
    "output_dir": "test_output",
    "rotate_180": True,
    "enable_final_transform": True,
    "final_rotate_axis": "x",
    "final_rotate_angle": -180.0,
}

def main():
    print("Testing single configuration...")
    print(f"Strategy: {TEST_CONFIG['strategy']}")
    print(f"Tool Width: {TEST_CONFIG['tool_width']}")
    print(f"Hatch Distance: {TEST_CONFIG['hatch_distance']}")
    print()
    
    # Save config
    config_file = "test_config_debug.json"
    with open(config_file, 'w') as f:
        json.dump(TEST_CONFIG, f, indent=2)
    
    print(f"Config saved to: {config_file}")
    print(f"Running: python AS4RCSD5.py --config {config_file}")
    print()
    
    # Run
    try:
        result = subprocess.run(
            ['python', 'AS4RCSD5.py', '--config', config_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Success: {result.returncode == 0}")
        print()
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
            print()
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print()
        
        # Check if output was created
        output_dir = Path(TEST_CONFIG['output_dir'])
        if output_dir.exists():
            print(f"Output directory created: {output_dir}")
            files = list(output_dir.glob('*'))
            print(f"Files created: {len(files)}")
            for f in files:
                print(f"  - {f.name}")
        else:
            print(f"Output directory NOT created: {output_dir}")
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT after 120 seconds")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Clean up
    try:
        Path(config_file).unlink()
    except:
        pass

if __name__ == '__main__':
    main()
