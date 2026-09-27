# Evaluation (Engineering Metrics)

## Purpose

Turn a planned toolpath into comparable numbers, so deposition strategies and
process parameters can be ranked on the same footing: how much of the layer is
actually covered, how much material is laid twice, how much of the deposit rests
on supporting material, how closely the deposit follows the target boundary, and
how efficient and smooth the motion is.

The numerical behaviour of every metric is preserved exactly from the original
`evaluation_metrics.py`; the module was relocated to
`src/evaluation/metrics.py` without changing the mathematics.

## Inputs

| Input | Meaning |
|---|---|
| `toolpaths` | List of per-layer `(deposition, tool_moves, fly_lines, fillets)` tuples |
| `slices` | List of per-layer `(layer_height, boundary_points)` pairs |
| `tool_width` | Bead width in mm; also the buffer width used to turn a path into an area |
| `grid_size` | Grid resolution for the area-based metrics (default 500; 50–100 for quick runs) |
| `num_samples` | Radial samples per boundary for CANPD (default 360) |

Library entry points (`src.evaluation`):

- `calculate_total_length(path)`
- `deposition_to_travel_ratio(deposition_length, travel_length)`
- `turn_angle_analysis(toolpaths)`
- `calculate_centroid(points)`, `radial_sample(points, centroid, num_samples=360)`
- `centroid_based_distance_per_layer(toolpaths, slices, tool_width, visualize=False, num_samples=360)`
- `calculate_deposition_color_map(...)` — retained for compatibility; superseded by
  the grid analysis on `ToolpathGenerator` and returning placeholder values

Area-based metrics are computed by
`ToolpathGenerator.calculate_layer_grid_metrics` and aggregated by
`evaluate_overall_strategy`.

## Processing

1. **Length metrics** — sum the Euclidean distances between consecutive path
   points. `NaN` separators mark breaks between disconnected runs and are skipped,
   so travel between raster lines is never counted as deposition.
2. **Turn angles** — for every interior triple of deposition points, compute the
   angle between the incoming and outgoing segment vectors, ignoring turns below
   0.1° and zero-length segments, then average over all valid turns of all layers.
3. **CANPD** — buffer the deposition polyline by `tool_width / 2` to get the
   deposited area, take its exterior as the deposited boundary, compute the
   centroid of the target boundary and of the deposited boundary, sample both
   radially at `num_samples` angles about their centroids, take the nearest
   deposited point for each sampled target point, discard distances beyond two
   standard deviations, and average the rest.
4. **Grid metrics** — rasterize the layer at `grid_size` × `grid_size` over the
   boundary bounding box. Each deposition segment is buffered by
   `tool_width / 2`, intersected with the boundary polygon, and accumulated, so a
   cell's value counts how many times it was deposited. The substrate contour is
   rasterized into a separate grid. Cells are then classified against a mask of
   cells inside the boundary.
5. **Wall engagement** — record, for each contour cell, the index of the first
   deposition segment that reaches it, and normalize the mean of those indices
   against the number of segments.

## Outputs

Per-layer records and a strategy-level average, written to
`evaluation_layer_<n>.json` / `.txt` and `evaluation_overall_strategy.json`, and
collected across cases by `scripts/evaluation/batch_evaluate.py` into a CSV plus a
text summary.

### Retained Deposition_Metrics

| Metric | Input | Output | Engineering interpretation |
|---|---|---|---|
| **Deposition length** | The deposition polyline of a layer (`NaN` runs skipped) | Length in mm | Productive travel — the distance over which material is laid. Scales with deposited volume at fixed bead geometry. |
| **Travel length** | Tool moves and fly lines of a layer | Length in mm | Non-productive travel with the source off. Costs cycle time and creates thermal interruptions. |
| **Deposition/travel ratio** | Both lengths above | Dimensionless ratio (`inf` when travel is zero) | Motion efficiency. Higher means more of the motion is productive; continuous strategies such as spiral and zig-zag score higher than raster. |
| **Average turn angle** | Deposition polyline, turns below 0.1° ignored | Degrees | Motion smoothness. Sharp turns force decelerations, which cause local over-deposition and heat accumulation, so lower is better. |
| **CANPD** (centroid-based average nearest-point deviation) | Layer boundary and deposition polyline, `tool_width`, `num_samples` | Distance in mm (`inf` when a layer has too few points or an empty deposited area) | Geometric fidelity of the layer outline: how far the deposited boundary sits from the target slice boundary, averaged over 360 radial directions with 2σ outlier rejection. Lower means a closer boundary match and less post-machining. |
| **Coverage %** | Deposition grid and the in-boundary cell mask | Percentage of in-boundary cells | Share of the layer area that receives material at least once. Shortfall indicates unfilled gaps between beads. |
| **Over-deposition %** | Combined deposition-and-contour grid | Percentage of in-boundary cells with an accumulated count above the supported level | Share of area receiving excess material from overlapping passes. Excess drives geometric error, residual stress, and wasted feedstock. |
| **Supported deposition %** | Deposition grid together with the contour grid | Percentage of in-boundary cells deposited exactly onto supporting material | Share of the deposit that lands on existing material rather than into free space. Low values predict sagging or collapse of unsupported beads. |
| **Supporting contour %** | Contour grid and deposition grid | Percentage of contour cells that later carry deposition | How much of the reconstructed substrate contour actually acts as a support for subsequent beads. |
| **Wall engagement index (WEI)** | Per-cell index of the first deposition segment reaching each contour cell | Value in 0–1 | How early in the layer the path engages the existing wall. Higher means engagement happens sooner, which improves fusion with the parent material. |

The batch runner also reports `total_layers`, `total_deposition_length`,
`total_travel_length`, and the standard deviations of coverage and
over-deposition across layers.

## Example

```bash
cd clear_repo

# Evaluate as part of a pipeline run (enable_evaluation defaults to true)
python scripts/toolpath/run_pipeline.py --config example_config.json

# Evaluate several configurations and collect their metrics
python scripts/evaluation/batch_evaluate.py --configs configs/ --output-dir results/batch
```

```python
from src.evaluation import (
    calculate_total_length, deposition_to_travel_ratio,
    turn_angle_analysis, centroid_based_distance_per_layer,
)

deposition_length = calculate_total_length(deposition)
travel_length = calculate_total_length(tool_moves)
ratio = deposition_to_travel_ratio(deposition_length, travel_length)
avg_turn = turn_angle_analysis(toolpaths)
per_layer, avg_canpd = centroid_based_distance_per_layer(toolpaths, slices, tool_width=2.0)
```

For faster iteration, lower `grid_size` (for example to 100): the area-based
metrics scale with `grid_size²`.

## Dependencies

numpy (array maths), shapely (buffering the path into an area, polygon
containment), scipy (`cKDTree` for nearest-point queries), matplotlib (CANPD and
colour-map visualizations). The grid metrics live on
`src.toolpath.generator.ToolpathGenerator` and therefore pull in the pipeline
dependency set.
