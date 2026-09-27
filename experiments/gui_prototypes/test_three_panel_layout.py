#!/usr/bin/env python3
"""
Test the new three-panel layout
"""

import tkinter as tk
from tkinter import ttk

def test_layout():
    """Test the three-panel layout"""
    
    root = tk.Tk()
    root.title("Three Panel Layout Test")
    root.geometry("1600x1000")
    
    # Create main frame
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left panel for basic controls
    left_panel = ttk.Frame(main_frame, width=280)
    left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
    left_panel.pack_propagate(False)
    
    # Middle panel for visualization
    middle_panel = ttk.Frame(main_frame)
    middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    
    # Right panel for advanced features
    right_panel = ttk.Frame(main_frame, width=280)
    right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    right_panel.pack_propagate(False)
    
    # Add content to left panel
    left_label = ttk.Label(left_panel, text="LEFT PANEL", font=("Arial", 12, "bold"))
    left_label.pack(pady=10)
    
    for i in range(5):
        frame = ttk.LabelFrame(left_panel, text=f"Basic Feature {i+1}")
        frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(frame, text="Button", width=15).pack(pady=2)
    
    # Add content to middle panel
    middle_label = ttk.Label(middle_panel, text="MIDDLE PANEL - VISUALIZATION", font=("Arial", 12, "bold"))
    middle_label.pack(pady=10)
    
    # Simulate notebook with tabs
    notebook = ttk.Notebook(middle_panel)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    viz_frame = ttk.Frame(notebook)
    notebook.add(viz_frame, text="Visualization")
    ttk.Label(viz_frame, text="3D Plot would go here", font=("Arial", 16)).pack(expand=True)
    
    gcode_frame = ttk.Frame(notebook)
    notebook.add(gcode_frame, text="G-Code")
    ttk.Label(gcode_frame, text="G-Code editor would go here", font=("Arial", 16)).pack(expand=True)
    
    # Add content to right panel
    right_label = ttk.Label(right_panel, text="RIGHT PANEL", font=("Arial", 12, "bold"))
    right_label.pack(pady=10)
    
    # Pattern Recognition section
    pattern_frame = ttk.LabelFrame(right_panel, text="Pattern Recognition")
    pattern_frame.pack(fill=tk.X, pady=2, padx=5)
    
    ttk.Label(pattern_frame, text="Reference Layer:").pack(anchor=tk.W)
    ttk.Entry(pattern_frame, width=20).pack(fill=tk.X, pady=2)
    
    preserve_frame = ttk.Frame(pattern_frame)
    preserve_frame.pack(fill=tk.X, pady=2)
    ttk.Label(preserve_frame, text="Initial:").pack(side=tk.LEFT)
    ttk.Entry(preserve_frame, width=5).pack(side=tk.LEFT, padx=2)
    ttk.Label(preserve_frame, text="Final:").pack(side=tk.LEFT)
    ttk.Entry(preserve_frame, width=5).pack(side=tk.LEFT, padx=2)
    
    button_frame = ttk.Frame(pattern_frame)
    button_frame.pack(fill=tk.X, pady=2)
    ttk.Button(button_frame, text="Analyze", width=10).pack(side=tk.LEFT, padx=2)
    ttk.Button(button_frame, text="Apply All", width=10).pack(side=tk.LEFT, padx=2)
    
    # Add more sections to right panel
    for i in range(3):
        frame = ttk.LabelFrame(right_panel, text=f"Advanced Feature {i+1}")
        frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(frame, text="Button", width=15).pack(pady=2)
    
    print("Three-panel layout created!")
    print("Left: Basic controls")
    print("Middle: Visualization and G-code")
    print("Right: Advanced features including Pattern Recognition")
    
    root.mainloop()

if __name__ == "__main__":
    test_layout()