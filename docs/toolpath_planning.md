# Toolpath Planning

## Purpose

Orchestrate the full repair process-planning pipeline: load the repair volume
and the substrate surface, apply reference-frame transforms, slice into layers,
fill each layer with the selected deposition strategy, project the in-plane path
onto the non-planar substrate, add fly-in/fly-out moves, evaluate the result, and
export the 3D trajectory. This is the flagship component, relocated from
`AS4RCSD5.py` with its public signatures and numerical behaviour preserved.

## Inputs

Everything arrives through `ToolpathConfig` (a dataclass, JSON-serializable) or
through command-line overrides. The fields that matter most:

| Field | Default | Meaning |
|---|---|---|
| `mesh_file` | repo-relative `Data/Toolpath_Models/Model_C_R.STL` | Repair volume mesh (STL) |
| `non_planar_surface_file` | repo-relative `Model_C_S.STL` | Substrate surface mesh (STL) |
| `layer_height` | `1.0` | Slice spacing in mm |
| `add_layers` | `0` | Extra layers above the mesh top |
| `strategy` | `1` | Deposition strategy 1–6, see `docs/deposition_strategies.md` |
| `tool_width` | `2.0` | Bead width in mm |
| `hatch_distance` | `1.5` | Spacing between deposition lines in mm |
| `raster_angle` | `0.0` | In-plane orientation of raster/zig-zag lines, degrees |
| `corner_radius` | `0.0` | Fillet radius at zig-zag corners, mm |
| `fly_in_out`, `fly_distance` | `True`, `2.0` | Approach/retract moves and their length |
| `enable_turbine_repair`, `extension_length` | `False`, `5.0` | Turbine-blade boundary extension |
| `enable_initial_transform`, `enable_final_transform` | `False` | Reference-frame handling, see `docs/coordinate_transformations.md` |
| `enable_evaluation`, `evaluate`, `grid_size` | `True`, `"all"`, `500` | Metric computation, see `docs/evaluation.md` |
| `export_csv`, `output_dir` | `True`, `"output"` | Trajectory export |

Library entry points:

- `src.toolpath.ToolpathConfig` — configuration dataclass, with
  `from_json` / `save_json` / `to_dict`
- `src.toolpath.default_config(**overrides)` — configuration using
  repository-relative default data locations
- `src.toolpath.ToolpathGenerator(config)` — the pipeline;
  `run()` executes every stage, and the stages are also callable individually
  (`load_meshes`, `slice_meshes`, `generate_toolpaths`, `evaluate_toolpaths`)

## Processing

1. **Load meshes** — read the repair volume and, when given, the substrate
   surface.
2. **Apply the initial transform** — optional offsets and axis remaps that bring
   the input mesh into the planning frame.
3. **Slice** — `slice_meshes` produces `(z, boundary_points)` per layer.
4. **Generate per-layer paths** — dispatch to the strategy selected by
   `strategy`, producing deposition, tool-move, fly-line, and fillet arrays.
5. **Project onto the substrate** — the first layers follow the reconstructed
   non-planar contour instead of a flat plane.
6. **Add fly-in/fly-out moves** — approach and retract segments of length
   `fly_distance` when `fly_in_out` is set.
7. **Evaluate** — compute the Deposition_Metric set per layer and overall.
8. **Apply the final transform** — optional offsets, axis remaps, and a rotation
   applied to the exported coordinates for the machine frame.
9. **Export** — write the trajectory CSVs, plots, and the effective
   configuration.

## Outputs

- `toolpath_trajectory.csv` and `complete_3d_toolpath.csv` — the trajectory with
  `X, Y, Z, Layer, Type` columns, where `Type` distinguishes deposition from
  travel.
- `layer_<n>_toolpath.csv` — per-layer paths.
- `toolpath_config.json` — the effective configuration actually used.
- `coordinate_transformations.json` — the transforms applied, for traceability.
- `evaluation_overall_strategy.json`, `evaluation_layer_<n>.json/.txt` — metrics.
- Plots: layer toolpaths, 3D toolpath, deposition colour maps.

Per-layer toolpath data uses the tuple
`(deposition_array, tool_moves_array, fly_lines_array, fillets_array)`.

## Example

```bash
cd clear_repo

# From a configuration file
python scripts/toolpath/run_pipeline.py --config example_config.json

# Overriding the inputs on the command line
python scripts/toolpath/run_pipeline.py \
    --mesh-file ../Data/Toolpath_Models/Model_C_R.STL \
    --non-planar-surface ../Data/Toolpath_Models/Model_C_S.STL \
    --output-dir results/toolpath/run1

# Resolve and save the effective configuration without running
python scripts/toolpath/run_pipeline.py --config example_config.json \
    --save-config results/effective_config.json --dry-run
```

```python
from src.toolpath import ToolpathGenerator, default_config

config = default_config(strategy=2, layer_height=0.5, output_dir="results/zigzag")
ToolpathGenerator(config).run()
```

## Dependencies

open3d, numpy, scipy, shapely, trimesh, pyvista, matplotlib, scikit-learn,
geomdl, sympy, pandas. Metrics come from `src.evaluation.metrics`. Matplotlib
selects a non-interactive backend automatically when no display is available, so
the pipeline is safe to run headless.
