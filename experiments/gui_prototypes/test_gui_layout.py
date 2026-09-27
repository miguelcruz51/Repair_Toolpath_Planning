#!/usr/bin/env python3
"""
Simple test to check if the pattern recognition section appears in the GUI
"""

import tkinter as tk
from tkinter import ttk

def test_pattern_section():
    """Test just the pattern recognition section"""
    
    root = tk.Tk()
    root.title("Pattern Recognition Test")
    root.geometry("400x600")
    
    # Create scrollable frame
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Add mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _bind_to_mousewheel(event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _unbind_from_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")
    
    canvas.bind('<Enter>', _bind_to_mousewheel)
    canvas.bind('<Leave>', _unbind_from_mousewheel)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Add some test sections
    for i in range(5):
        test_frame = ttk.LabelFrame(scrollable_frame, text=f"Test Section {i+1}")
        test_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
        
        for j in range(3):
            ttk.Button(test_frame, text=f"Button {j+1}", width=10).pack(pady=2)
    
    # Add Pattern Recognition section
    pattern_frame = ttk.LabelFrame(scrollable_frame, text="Pattern Recognition")
    pattern_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
    
    # Pattern analysis
    ttk.Label(pattern_frame, text="Reference Layer:").pack(anchor=tk.W)
    pattern_layer_var = tk.StringVar(value="0")
    ttk.Entry(pattern_frame, textvariable=pattern_layer_var).pack(fill=tk.X, pady=2)
    
    # Initial and final commands settings
    preserve_frame = ttk.Frame(pattern_frame)
    preserve_frame.pack(fill=tk.X, pady=2)
    
    # Initial commands
    ttk.Label(preserve_frame, text="Keep Initial:").pack(side=tk.LEFT)
    initial_commands_var = tk.StringVar(value="2")
    initial_entry = ttk.Entry(preserve_frame, textvariable=initial_commands_var, width=5)
    initial_entry.pack(side=tk.LEFT, padx=(2, 10))
    
    # Final commands  
    ttk.Label(preserve_frame, text="Keep Final:").pack(side=tk.LEFT)
    final_commands_var = tk.StringVar(value="2")
    final_entry = ttk.Entry(preserve_frame, textvariable=final_commands_var, width=5)
    final_entry.pack(side=tk.LEFT, padx=2)
    
    # Pattern buttons
    pattern_buttons_frame1 = ttk.Frame(pattern_frame)
    pattern_buttons_frame1.pack(fill=tk.X, pady=1)
    
    def test_analyze():
        print("Analyze button clicked!")
    
    def test_apply():
        print("Apply button clicked!")
    
    ttk.Button(pattern_buttons_frame1, text="Analyze", command=test_analyze, width=8).pack(side=tk.LEFT, padx=(0, 2))
    ttk.Button(pattern_buttons_frame1, text="Apply", command=test_apply, width=8).pack(side=tk.LEFT, padx=(2, 0))
    
    # Pattern info display
    pattern_info_var = tk.StringVar(value="No pattern analyzed")
    ttk.Label(pattern_frame, textvariable=pattern_info_var, font=("Arial", 8), wraplength=280).pack(anchor=tk.W, pady=2)
    
    # Add more test sections after
    for i in range(3):
        test_frame = ttk.LabelFrame(scrollable_frame, text=f"After Pattern Section {i+1}")
        test_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
        
        for j in range(2):
            ttk.Button(test_frame, text=f"Button {j+1}", width=10).pack(pady=2)
    
    print("Pattern Recognition section should be visible!")
    print("Try scrolling with mouse wheel if you don't see it.")
    
    root.mainloop()

if __name__ == "__main__":
    test_pattern_section()