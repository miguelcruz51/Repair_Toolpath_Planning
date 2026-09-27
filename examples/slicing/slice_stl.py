#!/usr/bin/env python3
"""Minimal example: slice an STL mesh into layer boundaries.

Demonstrates :func:`src.slicing.slice_mesh`, which delegates to the preserved
``ToolpathGenerator`` slicing implementation.

All inputs are command-line arguments; the default mesh is expressed relative to
the repository root (Requirements 5.2, 5.3).

Usage::

    python examples/slicing/slice_stl.py
    python examples/slicing/slice_stl.py --mesh Data/Toolpath_Models/Model_A_R.STL
    python examples/slicing/slice_stl.py --layer-height 0.5 --max-layers 5
"""

import argparse
import sys
from pathlib import Path

# clear_repo/ is two levels up from examples/slicing/.
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CLEAR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAR_REPO_ROOT))

from src.slicing import slice_mesh  # noqa: E402
from src.toolpath import DEFAULT_MESH_FILE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for this example."""
    parser = argparse.ArgumentParser(
        description="Slice an STL mesh along Z and report the layer boundaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh",
        default=str(DEFAULT_MESH_FILE),
        help="Path to the STL mesh to slice (repository-relative default).",
    )
    parser.add_argument(
        "--layer-height", type=float, default=1.0, help="Layer height in millimetres."
    )
    parser.add_argument(
        "--add-layers", type=int, default=0, help="Extra layers above the mesh top."
    )
    parser.add_argument(
        "--max-layers", type=int, default=5, help="Number of layers to print."
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    mesh_path = Path(args.mesh)
    if not mesh_path.is_file():
        print(f"Mesh file not found: {mesh_path}", file=sys.stderr)
        return 1

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        print(f"Mesh is empty: {mesh_path}", file=sys.stderr)
        return 1

    slices = slice_mesh(mesh, layer_height=args.layer_height, add_layers=args.add_layers)

    print(f"Mesh:        {mesh_path}")
    print(f"Vertices:    {len(mesh.vertices)}")
    print(f"Layer height: {args.layer_height} mm")
    print(f"Slices:      {len(slices)}")
    for height, boundary in slices[: args.max_layers]:
        print(f"  z = {height:8.3f} mm -> {len(boundary)} boundary points")
    if len(slices) > args.max_layers:
        print(f"  ... {len(slices) - args.max_layers} more layers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
