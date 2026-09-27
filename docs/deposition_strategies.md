# Deposition Strategies

## Purpose

Fill a single layer boundary with a deposition pattern. The strategy determines
how much of the layer is covered, how much non-productive travel is required, how
smooth the motion is, and how well the deposited material engages the existing
wall. Comparing the six strategies under identical process parameters is the core
experiment of the project.

## Inputs

| Input | Type | Meaning |
|---|---|---|
| `boundary_points` | array of 2D points | The layer boundary to fill, from slicing |
| `non_planar_boundary_points` | array of 2D points | Substrate contour (Hybrid 2 only) |
| `layer_index` | int | Index of the layer, used for bookkeeping and plots |
| `tool_width` | float, mm | Bead width; also the rasterization buffer width |
| `hatch_distance` | float, mm | Spacing between adjacent deposition lines |
| `raster_angle` | float, degrees | In-plane orientation of raster/zig-zag lines |
| `corner_radius` | float, mm | Fillet radius at zig-zag corners (0 disables) |
| `enable_extension` | bool | Hybrid 2 contour extension |
| `use_optimized_hybrid2` | bool | Select the simplified, more robust Hybrid 2 |

The strategy is selected by the `strategy` field of `ToolpathConfig`:

| `strategy` | Name | Generator method | Character |
|---|---|---|---|
| 1 | Raster | `generate_raster_toolpath` | Parallel deposition lines with travel moves between them |
| 2 | Zig-zag | `generate_zigzag_toolpath` | Continuous alternating lines, optional corner fillets |
| 3 | Contour | `generate_contour_toolpath` | Inward contour-parallel offsets of the boundary |
| 4 | Spiral | `generate_spiral_toolpath` | One continuous inward spiral, no travel moves |
| 5 | Hybrid 1 | `generate_hybrid1_toolpath` | Contour perimeter followed by zig-zag infill |
| 6 | Hybrid 2 | `generate_hybrid2_toolpath`, `generate_hybrid2_optimized_toolpath` | Contour-following offsets with boundary extension and inner offset |

All generators are methods of `src.toolpath.generator.ToolpathGenerator` and keep
the signatures they had in `AS4RCSD5.py`.

## Processing

1. Convert `boundary_points` into a Shapely polygon.
2. For line-based strategies (1, 2, 5), rotate the polygon by `raster_angle`,
   intersect a family of lines spaced `hatch_distance` apart with the polygon,
   then rotate the resulting segments back.
3. For offset-based strategies (3, 4, 6), repeatedly buffer the polygon inward by
   `hatch_distance` and collect the resulting rings; the spiral variant joins the
   rings into one continuous path.
4. Classify each produced segment as deposition, tool move (laser off, in-plane),
   fly line (approach/retract), or fillet.
5. Insert `NaN` separators between disconnected deposition runs so that length
   and turn-angle metrics do not count the gaps as deposited material.
6. Hybrid 2 additionally extends the contour beyond the boundary by
   `extension_length` and adds an inner offset so the bead engages the wall.

## Outputs

Each generator returns the per-layer tuple

```
(deposition_array, tool_moves_array, fly_lines_array, fillets_array)
```

- `deposition_array` — the ordered deposition polyline, `NaN`-separated between
  disconnected runs.
- `tool_moves_array` — in-plane repositioning moves with the source off.
- `fly_lines_array` — approach and retract moves.
- `fillets_array` — corner-smoothing arcs, empty when `corner_radius` is 0.

Move classification is also recorded per layer so the exported CSV can mark each
point as deposition or travel.

## Example

```bash
cd clear_repo

# Compare two strategies with otherwise identical parameters
python scripts/toolpath/run_pipeline.py --config configs/zigzag.json
python scripts/toolpath/run_pipeline.py --config configs/hybrid2.json

# Sweep several strategies and collect the metrics into one CSV
python scripts/evaluation/batch_evaluate.py --configs configs/ --output-dir results/batch
```

```python
from src.toolpath import ToolpathGenerator, default_config

# strategy 6 = Hybrid 2, with the simplified variant
config = default_config(strategy=6, use_optimized_hybrid2=True,
                        tool_width=2.0, hatch_distance=1.5)
ToolpathGenerator(config).run()
```

Research comparisons live in `experiments/deposition_strategies/`
(`test_hybrid2_comparison.py`, `test_contour_detection.py`,
`test_pattern_recognition.py`); earlier generations of the strategy code are kept
as version history in `experiments/toolpath_planning/`.

## Dependencies

shapely (polygon offsetting, line intersection, buffering), numpy, scipy
(splines and nearest-neighbour queries), geomdl (curve approximation for
extension and smoothing), matplotlib (strategy plots).
