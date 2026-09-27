# Slicing

## Purpose

Cut the repair volume mesh into a stack of planar layer boundaries along the
Z-axis, at a configurable layer height. The resulting boundaries are the input
domain for every deposition strategy: each one defines the region a single layer
must fill.

## Inputs

| Input | Type | Meaning |
|---|---|---|
| `mesh` | Open3D `TriangleMesh` | The repair volume to slice |
| `layer_height` | float, mm | Vertical spacing between slice planes (default `0.5` in `slice_mesh`, `1.0` in `ToolpathConfig`) |
| `add_layers` | int | Extra slice planes above the mesh top, for over-build |
| `plane` | length-4 array | Plane equation `[nx, ny, nz, d]` for a single cross-section (`extract_slice`) |

Library entry points:

- `src.slicing.slice_mesh(mesh, layer_height=0.5, add_layers=0)`
- `src.slicing.extract_slice(mesh, plane)`

Both delegate to the preserved implementations on
`src.toolpath.generator.ToolpathGenerator`, so the numerical behaviour is
identical to the original `AS4RCSD5.py` code.

## Processing

1. Read the mesh bounding box and derive the Z range to be sliced.
2. Step from the bottom of the mesh to the top in `layer_height` increments,
   plus `add_layers` additional planes above the top.
3. For each plane, intersect the triangles with the plane to obtain segment
   endpoints (`extract_slice`).
4. Order the intersection points into a usable boundary contour and drop planes
   that do not intersect the mesh.

The Z-axis is the slicing axis (`slicing_axis = "z"`); any other orientation is
achieved by first rotating the mesh, see `docs/coordinate_transformations.md`.

## Outputs

- `slice_mesh` returns a list of `(layer_height, boundary_points)` pairs, ordered
  bottom to top. `layer_height` is the absolute Z of the plane in millimetres,
  `boundary_points` is the 2D boundary of the cross-section at that plane.
- `extract_slice` returns the 2D boundary points of a single plane, or `None`
  when the plane does not intersect the mesh.

## Example

```bash
cd clear_repo
python examples/slicing/slice_stl.py --layer-height 0.5 --max-layers 5
python examples/slicing/slice_stl.py --mesh ../Data/Toolpath_Models/Model_A_R.STL
```

```python
import open3d as o3d
from src.slicing import slice_mesh

mesh = o3d.io.read_triangle_mesh("../Data/Toolpath_Models/Model_C_R.STL")
slices = slice_mesh(mesh, layer_height=0.5)
for z, boundary in slices[:3]:
    print(z, len(boundary))
```

## Dependencies

open3d (mesh I/O and geometry), numpy (array maths), shapely (boundary
polygons). Slicing pulls in `src.toolpath.generator`, which imports the full
pipeline dependency set listed in `requirements.txt`.
