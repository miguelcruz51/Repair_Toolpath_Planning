# Coordinate Transformations (Reference Frames)

## Purpose

Move geometry and trajectories between the frames involved in a repair job: the
CAD frame the meshes arrive in, the planning frame the slicer and strategies work
in, and the machine frame the G-code must be expressed in. Handling this
explicitly is what allows the same planned toolpath to be executed on a machine
whose axes are named or oriented differently.

## Inputs

| Function | Inputs |
|---|---|
| `rotate_mesh(mesh, axis, angle, center=None)` | Open3D `TriangleMesh`, axis `"x"`/`"y"`/`"z"`, angle in degrees, optional rotation centre (mesh centre when omitted) |
| `rotate_points(points, angle, origin)` | 2D points, angle in degrees, an origin exposing `.x`/`.y` (Shapely `Point`) |
| `transform_coordinates(coords, offset_x, offset_y, offset_z, remap_x, remap_y, remap_z)` | `(N, 3)` array, three offsets in mm, three axis names `"X"`/`"Y"`/`"Z"` |
| `rotate_coordinates(coords, axis, angle_degrees)` | `(N, 3)` array, axis `"x"`/`"y"`/`"z"`, angle in degrees |

Library entry point: `src.reference_frames.transformations`. The functions
delegate to the preserved `ToolpathGenerator` implementations, so behaviour is
identical to `AS4RCSD5.py`.

Through the configuration, two transform stages are available:

| Stage | Fields | Applied to |
|---|---|---|
| Initial | `enable_initial_transform`, `initial_offset_x/y/z`, `initial_remap_x/y/z` | The input mesh, before slicing |
| Final | `enable_final_transform`, `final_offset_x/y/z`, `final_remap_x/y/z`, `final_rotate_axis`, `final_rotate_angle` | The exported trajectory coordinates |

`rotate_180` is a convenience flag that flips the input mesh before processing.

## Processing

1. **Axis remap** — each output axis takes its value from the named input axis, so
   `remap_x="Y", remap_y="X", remap_z="Z"` swaps X and Y. A remap is a pure
   relabelling; no scaling is involved.
2. **Offset** — a translation in millimetres is added per axis, used to place the
   part origin where the machine expects it.
3. **Rotation** — a rigid rotation about a single principal axis by an angle in
   degrees, applied about the mesh centre (`rotate_mesh`) or about the coordinate
   origin (`rotate_coordinates`).

The initial transform runs before slicing, so the layer stack is built in the
planning frame. The final transform runs after trajectory generation, so the
planned geometry is untouched and only the exported numbers change. The transforms
actually applied are recorded to `coordinate_transformations.json` for
traceability.

## Outputs

- `rotate_mesh` returns the rotated `TriangleMesh`.
- `rotate_points` returns the rotated 2D point array.
- `transform_coordinates` and `rotate_coordinates` return transformed `(N, 3)`
  arrays.
- Within the pipeline: a transformed input mesh, transformed export coordinates,
  and `coordinate_transformations.json` plus a
  `coordinate_system_transformation.png` diagram in the output directory.

## Example

```bash
cd clear_repo
# Transform fields are set in the configuration file, then:
python scripts/toolpath/run_pipeline.py --config configs/machine_frame.json
```

```python
import numpy as np
from src.reference_frames.transformations import (
    transform_coordinates, rotate_coordinates,
)

coords = np.array([[1.0, 2.0, 3.0]])

# Swap X and Y, then shift Z by 10 mm
remapped = transform_coordinates(coords, 0.0, 0.0, 10.0, "Y", "X", "Z")

# Rotate the trajectory 90 degrees about Z for the machine frame
machine = rotate_coordinates(remapped, "z", 90.0)
```

Configuration excerpt:

```json
{
  "enable_final_transform": true,
  "final_offset_x": 100.0,
  "final_offset_y": 50.0,
  "final_offset_z": 0.0,
  "final_remap_x": "X",
  "final_remap_y": "Y",
  "final_remap_z": "Z",
  "final_rotate_axis": "z",
  "final_rotate_angle": 90.0
}
```

## Dependencies

numpy (rotation matrices and array arithmetic), open3d (mesh rotation), shapely
(2D rotation origin). Importing `src.reference_frames.transformations` pulls in
`src.toolpath.generator` and therefore the full pipeline dependency set.
