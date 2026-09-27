# Functionality Inventory

This document records every Python source file in the original repository (root and
`Dataset/Boolean Comparison/`) with its purpose, inputs, outputs, dependencies, overlap
relationships, and a single classification label from the set
`{reusable, experiment, script, obsolete}`. Files whose classification could not be
determined confidently from inspection carry a `needs-manual-review` note.

Classification legend:
- **reusable**: importable, general-purpose computational functionality (destined for `src/`).
- **experiment**: standalone research investigation / comparison / figure generation (destined for `experiments/`).
- **script**: command-line or GUI entry point that invokes reusable functionality (destined for `scripts/` or `examples/`).
- **obsolete**: superseded and not retained.

---

## Root-level Python files

### AS4RCSD.py
- **Purpose**: Earliest toolpath prototype. Aligns a repair volume to a damaged model, slices the repair volume by layer height, and visualizes models/slices with PyVista/Matplotlib.
- **Inputs**: STL paths for damaged model and repair volume (function arguments); visualization control flags.
- **Outputs**: PyVista/Matplotlib windows; sliced layer data in memory.
- **Dependencies**: pyvista, numpy, matplotlib, vtk.
- **Classification**: experiment
- **Overlap**: First in the `AS4RCSD.py .. AS4RCSD5.py` family; superseded by later versions but retained as version history.

### AS4RCSD2.py
- **Purpose**: Second toolpath prototype. Adds point-cloud rotation, NURBS/B-spline surface fitting, RANSAC/DBSCAN/PCA processing, and shapely-based path geometry. Has a `main()`.
- **Inputs**: Point clouds / meshes via functions; hard-coded/example paths inside `main`.
- **Outputs**: Plots and intermediate geometry.
- **Dependencies**: open3d, numpy, matplotlib, geomdl, scipy, shapely, sklearn, skimage.
- **Classification**: experiment
- **Overlap**: Member of the `AS4RCSD` family; superseded by AS4RCSD5.

### AS4RCSD3.py
- **Purpose**: Third toolpath prototype. Introduces the `slice_mesh` routine, lasso selection, spline fitting, Hausdorff comparison, and interactive layer selection (`input(...)` prompts). Has a `main()`.
- **Inputs**: Meshes/point clouds; interactive console input for layer selection.
- **Outputs**: Plots, sliced geometry, toolpaths.
- **Dependencies**: open3d, numpy, matplotlib, scipy, shapely, trimesh, pyvista, sklearn, geomdl, sympy.
- **Classification**: experiment
- **Overlap**: Member of the `AS4RCSD` family; the direct predecessor of AS4RCSD4/5.

### AS4RCSD4.py
- **Purpose**: Fourth toolpath version. First "optimized" refactor introducing `ToolpathConfig` dataclass, argparse/`--config`, and a structured `ToolpathGenerator`-style pipeline. Has `main()`.
- **Inputs**: JSON config via `--config`; STL mesh + surface files referenced in config.
- **Outputs**: Toolpaths, evaluation results, plots.
- **Dependencies**: open3d, numpy, matplotlib, scipy, shapely, trimesh, pyvista, sklearn, geomdl, sympy.
- **Classification**: experiment
- **Overlap**: Immediate predecessor of AS4RCSD5.py (nearly identical structure); superseded by AS4RCSD5.

### AS4RCSD5.py
- **Purpose**: Canonical, most complete toolpath generation pipeline. Contains the current `ToolpathConfig` dataclass and `ToolpathGenerator` (load, slice, strategy generation, evaluation). Imports metric functions via `from evaluation_metrics import (...)`. Has `main()`.
- **Inputs**: JSON config via `--config` (mesh file, surface file, layer height, strategy, tool width, hatch distance, grid size, transform/turbine parameters, evaluation flags, output dir).
- **Outputs**: Toolpath CSVs, evaluation metric files, plots, config JSON.
- **Dependencies**: open3d, numpy, matplotlib, scipy, shapely, trimesh, pyvista, sklearn, geomdl, sympy; local `evaluation_metrics`.
- **Classification**: reusable
- **Secondary role**: also provides a command-line entry point (`main()` / `--config`). The core is destined for `src/toolpath/generator.py`; the entry point becomes `scripts/toolpath/run_pipeline.py`.
- **Overlap**: Newest of the `AS4RCSD` family; canonical replacement for AS4RCSD.py .. AS4RCSD4.py.

### evaluation_metrics.py
- **Purpose**: Deposition/toolpath evaluation metrics: total length, deposition-to-travel ratio, turn-angle analysis, centroid computation, radial sampling, CANPD per-layer distance, deposition color map.
- **Inputs**: Toolpath arrays / paths (numpy arrays, layer tuples).
- **Outputs**: Numerical metric values, color maps.
- **Dependencies**: numpy, shapely, scipy, matplotlib.
- **Classification**: reusable
- **Overlap**: Imported by AS4RCSD5.py. Destined for `src/evaluation/metrics.py`.

### Simulation.py
- **Purpose**: Deposition trajectory simulation and visualization (`simulate_deposition`, `simulate_material_deposition`, trajectory CSV import). Reusable simulation functions plus an interactive `__main__` that calls `input(...)` for a CSV filename.
- **Inputs**: Trajectory arrays; CSV trajectory file (interactive prompt in `__main__`).
- **Outputs**: Matplotlib/Plotly animations/plots.
- **Dependencies**: matplotlib, numpy, open3d, plotly.
- **Classification**: reusable
- **Secondary role**: the interactive `__main__` becomes an example entry point. Reusable functions destined for `src/deposition/simulation.py`. The `input(...)` prompt must not carry over into the reusable module.
- **Overlap**: None.

### gcode_dev.py
- **Purpose**: G-code development tool. `GCodeConverter` class parses CSV toolpaths and G-code lines, converts between them, plus a large Tkinter GUI for editing/visualization.
- **Inputs**: CSV toolpath files, G-code files (via GUI file dialogs).
- **Outputs**: G-code files, plots, GUI.
- **Dependencies**: pandas, numpy, tkinter, matplotlib.
- **Classification**: reusable
- **Secondary role**: the Tkinter GUI portion becomes a script entry point (`scripts/gcode/gcode_editor.py`). Non-UI parsing (`parse_gcode_line`, `parse_gcode`, `load_csv`, coordinate extraction) destined for `src/gcode/converter.py`.
- **Overlap**: Near-duplicate of gcode_dev_v2.py.

### gcode_dev_v2.py
- **Purpose**: Revised version of gcode_dev.py with the same `GCodeConverter` interface and Tkinter GUI.
- **Inputs**: CSV toolpath files, G-code files (via GUI).
- **Outputs**: G-code files, plots, GUI.
- **Dependencies**: pandas, numpy, tkinter, matplotlib.
- **Classification**: reusable
- **Secondary role**: the Tkinter GUI portion becomes a script entry point. Parsing logic consolidated into `src/gcode/converter.py`.
- **Overlap**: Direct successor of gcode_dev.py (same public parsing API). The two overlap heavily; v2 is the more recent. Both retained.

### repair_editor.py
- **Purpose**: Unified Tkinter "power tool" combining G-code editing, 3D visualization (open3d), and alignment analysis.
- **Inputs**: G-code, damaged STL, repair STL (via GUI file paths).
- **Outputs**: Edited G-code, 3D views, GUI.
- **Dependencies**: open3d, numpy, tkinter.
- **Classification**: script (GUI)
- **Overlap**: Shares G-code parsing concepts with gcode_dev*/visualizer_advanced. Destined for `scripts/gcode/repair_editor.py`.

### visualizer_advanced.py
- **Purpose**: Advanced Tkinter 3D visualizer for the repair process (load damaged/repair STL, load toolpath, toggle visibility/transparency, screenshots).
- **Inputs**: Damaged STL, repair STL, G-code/CSV toolpath (via GUI).
- **Outputs**: 3D visualization, screenshots, GUI.
- **Dependencies**: open3d, numpy, tkinter.
- **Classification**: script (GUI)
- **Overlap**: Visualization overlaps with repair_editor.py. Destined for `scripts/visualization/repair_visualizer.py`.

### ran3d.py
- **Purpose**: RAN-3D voxel-based point-cloud processing / damage detection (`VoxelProcessing` class, ICP, KD-tree, segmentation).
- **Inputs**: Reads parameters from an absolute machine-specific path (`/home/miguel-cruz/Documents/RAN-3D/...`); point clouds.
- **Outputs**: Processed point clouds, visualizations.
- **Dependencies**: open3d, numpy, matplotlib, more_itertools, pandas, scipy.
- **Classification**: experiment
- **Overlap**: Related to the separate damage-detection repository (Repository 1). Destined for `experiments/damage_detection/ran3d.py`.
- **Note**: needs-manual-review — contains a hard-coded machine-specific parameters path that must be parameterized if ever promoted beyond an experiment.

### analyze_bb.py
- **Purpose**: One-off bounding-box analysis of a damaged part and repair volume; prints min/max/center/size.
- **Inputs**: Hard-coded STL paths (one absolute machine-specific path).
- **Outputs**: Console prints.
- **Dependencies**: numpy, open3d.
- **Classification**: experiment
- **Overlap**: Overlaps conceptually with analyze_stl_coordinates.py. Destined for `experiments/geometry/`.
- **Note**: needs-manual-review — contains an absolute machine-specific STL path.

### analyze_stl_coordinates.py
- **Purpose**: STL coordinate-system analysis without visualization; prints bounding boxes and overlap for multiple STLs.
- **Inputs**: STL file paths (function arguments; `__main__` uses example paths).
- **Outputs**: Console prints, returned dicts.
- **Dependencies**: open3d, numpy.
- **Classification**: experiment
- **Overlap**: Overlaps with analyze_bb.py (both do bounding-box/coordinate analysis). Destined for `experiments/geometry/`.

### batch_evaluation.py
- **Purpose**: Batch runner that executes many toolpath configurations by invoking AS4RCSD5.py via subprocess and aggregating results into CSV/summary files.
- **Inputs**: In-file `BASE_CONFIG` (contains absolute machine-specific mesh/surface paths); generated per-test configs.
- **Outputs**: Per-config result directories, aggregated CSV and summary files under `batch_results/`.
- **Dependencies**: standard library (json, subprocess, csv, datetime, pathlib); invokes AS4RCSD5.py.
- **Classification**: script
- **Overlap**: Overlaps with batch_evaluation_template.py and batch_evaluation.sh. Destined for `scripts/evaluation/batch_evaluate.py`.
- **Note**: needs-manual-review — contains absolute machine-specific paths in `BASE_CONFIG` that must become arguments.

### batch_evaluation_template.py
- **Purpose**: Simplified batch-evaluation template with an editable `TEST_CONFIGS` list.
- **Inputs**: In-file `MESH_FILE`/`SURFACE_FILE` (absolute machine-specific paths); `TEST_CONFIGS`.
- **Outputs**: Result directories/CSV under `batch_results/`.
- **Dependencies**: standard library (json, subprocess, csv, datetime, pathlib); invokes AS4RCSD5.py.
- **Classification**: script
- **Overlap**: Template variant of batch_evaluation.py.
- **Note**: needs-manual-review — contains absolute machine-specific paths.

### test_contour_detection.py
- **Purpose**: Demonstration/test of contour-strategy detection using `GCodeConverter` on sample G-code.
- **Inputs**: In-file sample G-code.
- **Outputs**: Console output / plots.
- **Dependencies**: gcode_dev (local).
- **Classification**: experiment
- **Overlap**: Depends on gcode_dev.py; part of the deposition-strategy studies. Destined for `experiments/deposition_strategies/`.

### test_pattern_recognition.py
- **Purpose**: Demonstration/test of G-code pattern-recognition feature using `GCodeConverter` on structured sample G-code.
- **Inputs**: In-file sample G-code.
- **Outputs**: Console output.
- **Dependencies**: gcode_dev (local).
- **Classification**: experiment
- **Overlap**: Depends on gcode_dev.py; related to test_contour_detection.py. Destined for `experiments/deposition_strategies/`.

### test_hybrid2_comparison.py
- **Purpose**: Compares original vs optimized Hybrid2 strategy by running AS4RCSD5.py via subprocess on two configs.
- **Inputs**: Config file paths.
- **Outputs**: Console comparison output, evaluation artifacts.
- **Dependencies**: standard library (json, subprocess, pathlib); invokes AS4RCSD5.py.
- **Classification**: experiment
- **Overlap**: Related to the batch-evaluation scripts. Destined for `experiments/deposition_strategies/`.

### test_gcode_fix.py
- **Purpose**: Ad-hoc developer test verifying G-code plotting for mixed G01/G00 commands.
- **Inputs**: In-file sample G-code.
- **Outputs**: Plots / console output.
- **Dependencies**: gcode_dev (local), matplotlib.
- **Classification**: experiment
- **Overlap**: Depends on gcode_dev.py.
- **Note**: needs-manual-review — pure developer debug helper; the design flags such files as candidates to drop, but it is retained as an experiment pending confirmation.

### test_gui_layout.py
- **Purpose**: Ad-hoc Tkinter test checking that a pattern-recognition GUI section renders.
- **Inputs**: None (builds a Tk window).
- **Outputs**: Tk window.
- **Dependencies**: tkinter.
- **Classification**: experiment
- **Note**: needs-manual-review — GUI layout debug helper; retained as an experiment pending confirmation to drop.

### test_three_panel_layout.py
- **Purpose**: Ad-hoc Tkinter test of a three-panel GUI layout.
- **Inputs**: None (builds a Tk window).
- **Outputs**: Tk window.
- **Dependencies**: tkinter.
- **Classification**: experiment
- **Note**: needs-manual-review — GUI layout debug helper; retained as an experiment pending confirmation to drop.

### test_single_config.py
- **Purpose**: Debug helper that runs a single toolpath configuration through AS4RCSD5.py via subprocess.
- **Inputs**: In-file `TEST_CONFIG` (contains absolute machine-specific paths).
- **Outputs**: Console output, evaluation artifacts.
- **Dependencies**: standard library (json, subprocess, pathlib); invokes AS4RCSD5.py.
- **Classification**: experiment
- **Note**: needs-manual-review — contains absolute machine-specific paths; debug helper related to batch_evaluation.py.

---

## `Dataset/Boolean Comparison/` Python files

### Dataset/Boolean Comparison/BC.py
- **Purpose**: Boolean-comparison damage extraction: registers a nominal vs damaged STL pair (PCA/feature/ICP), extracts missing surface and volume, visualizes, and exports `.pcd`.
- **Inputs**: Nominal and damaged STL paths; alignment/visualization/export flags; voxel/DBSCAN parameters.
- **Outputs**: `.pcd` files (surface/volume), visualizations.
- **Dependencies**: open3d, numpy, scipy, sklearn.
- **Classification**: experiment
- **Overlap**: Near-duplicate of BC2.py and closely related to DamageRec.py. Destined for `experiments/geometry/`.

### Dataset/Boolean Comparison/BC2.py
- **Purpose**: Revised Boolean-comparison damage extraction with adjusted default alignment methods and thresholds and an added `system` flag.
- **Inputs**: Nominal/damaged STL paths; flags and parameters.
- **Outputs**: `.pcd` files, visualizations.
- **Dependencies**: open3d, numpy, scipy, sklearn, matplotlib.
- **Classification**: experiment
- **Overlap**: Direct successor of BC.py (same `process_pair` API, tuned defaults). Both retained as version history.

### Dataset/Boolean Comparison/DamageRec.py
- **Purpose**: Damage-region reconstruction: aligns nominal/damaged meshes and extracts the missing region using PCA/cross-section/feature-matching/ICP options.
- **Inputs**: Nominal/damaged STL paths; alignment option flags; voxel/threshold parameters.
- **Outputs**: Extracted missing-region point clouds, visualizations.
- **Dependencies**: open3d, numpy, scipy, sklearn.
- **Classification**: experiment
- **Overlap**: Shares the boolean/registration approach with BC.py and BC2.py. Destined for `experiments/geometry/`.

---

## Overlap summary

- **AS4RCSD family**: `AS4RCSD.py`, `AS4RCSD2.py`, `AS4RCSD3.py`, `AS4RCSD4.py`, `AS4RCSD5.py` are successive versions of the same toolpath-generation effort. `AS4RCSD5.py` is canonical (reusable); the rest are retained as experiments (version history). No file deleted.
- **G-code tools**: `gcode_dev.py` vs `gcode_dev_v2.py` share the same `GCodeConverter` public parsing API; v2 is the newer revision. Both retained; parsing logic consolidated into `src/gcode/converter.py`.
- **Batch evaluation**: `batch_evaluation.py`, `batch_evaluation_template.py`, and `test_single_config.py` overlap as configuration-driven subprocess runners of AS4RCSD5.py.
- **Geometry/coordinate analysis**: `analyze_bb.py` vs `analyze_stl_coordinates.py` both compute bounding boxes / coordinate systems.
- **Boolean comparison**: `BC.py`, `BC2.py`, and `DamageRec.py` overlap in nominal-vs-damaged registration and missing-region extraction.
- **Visualization/editing GUIs**: `repair_editor.py` and `visualizer_advanced.py` overlap in 3D repair visualization.

## Files flagged `needs-manual-review`

- `ran3d.py` — absolute machine-specific parameters path.
- `analyze_bb.py` — absolute machine-specific STL path.
- `batch_evaluation.py`, `batch_evaluation_template.py`, `test_single_config.py` — absolute machine-specific paths in in-file config blocks.
- `test_gcode_fix.py`, `test_gui_layout.py`, `test_three_panel_layout.py` — pure developer/GUI debug helpers; candidates to drop pending confirmation, retained as experiments for now.
