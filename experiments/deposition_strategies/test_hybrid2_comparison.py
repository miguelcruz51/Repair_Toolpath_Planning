#!/usr/bin/env python3
"""
Quick test script to compare original vs optimized Hybrid2
"""

import json
import subprocess
import time
from pathlib import Path

def run_evaluation(config_file, output_suffix):
    """Run evaluation with given config."""
    print(f"\n{'='*60}")
    print(f"Running evaluation: {output_suffix}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'AS4RCSD5.py', '--config', config_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"\nCompleted in {elapsed:.2f} seconds")
        
        # Count warnings
        warnings = result.stdout.count("WARNING")
        errors = result.stdout.count("ERROR")
        
        print(f"Warnings: {warnings}")
        print(f"Errors: {errors}")
        
        return {
            'success': result.returncode == 0,
            'time': elapsed,
            'warnings': warnings,
            'errors': errors,
            'output': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 5 minutes")
        return {
            'success': False,
            'time': 300,
            'warnings': 0,
            'errors': 1,
            'output': 'Timeout'
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'success': False,
            'time': 0,
            'warnings': 0,
            'errors': 1,
            'output': str(e)
        }

def main():
    """Compare original vs optimized Hybrid2."""
    
    # Load base config
    with open('example_config.json', 'r') as f:
        config = json.load(f)
    
    # Ensure we're testing Hybrid2
    config['strategy'] = 6
    config['enable_evaluation'] = True
    config['enable_visualization'] = False  # Disable for speed
    
    # Test 1: Original Hybrid2
    config['use_optimized_hybrid2'] = False
    config['output_dir'] = str(Path(config['output_dir']).parent / 'hybrid2_original')
    
    with open('test_config_original.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    results_original = run_evaluation('test_config_original.json', 'Original Hybrid2')
    
    # Test 2: Optimized Hybrid2
    config['use_optimized_hybrid2'] = True
    config['output_dir'] = str(Path(config['output_dir']).parent / 'hybrid2_optimized')
    
    with open('test_config_optimized.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    results_optimized = run_evaluation('test_config_optimized.json', 'Optimized Hybrid2')
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15}")
    print(f"{'-'*60}")
    print(f"{'Success':<30} {results_original['success']!s:<15} {results_optimized['success']!s:<15}")
    print(f"{'Time (seconds)':<30} {results_original['time']:<15.2f} {results_optimized['time']:<15.2f}")
    print(f"{'Warnings':<30} {results_original['warnings']:<15} {results_optimized['warnings']:<15}")
    print(f"{'Errors':<30} {results_original['errors']:<15} {results_optimized['errors']:<15}")
    
    # Calculate improvement
    if results_original['time'] > 0:
        speedup = results_original['time'] / results_optimized['time']
        print(f"\nSpeedup: {speedup:.2f}x")
    
    warning_reduction = results_original['warnings'] - results_optimized['warnings']
    print(f"Warning Reduction: {warning_reduction}")
    
    print(f"\nCheck output directories for detailed results:")
    print(f"  Original: {Path(config['output_dir']).parent / 'hybrid2_original'}")
    print(f"  Optimized: {Path(config['output_dir']).parent / 'hybrid2_optimized'}")

if __name__ == '__main__':
    main()
