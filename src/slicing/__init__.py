"""Slicing: mesh slicing along Z.

The slicing implementation lives on :class:`src.toolpath.generator.ToolpathGenerator`
(relocated verbatim from ``AS4RCSD5.py``). This package exposes it as plain
functions by delegating to the preserved methods, so slicing can be used without
running the full toolpath pipeline. The numerical behaviour is unchanged.

Inputs
    An Open3D ``TriangleMesh`` and a layer height in millimetres.

Outputs
    ``slice_mesh`` returns a list of ``(layer_height, boundary_points)`` pairs.
    ``extract_slice`` returns the 2D boundary points of a single plane, or
    ``None`` when the plane does not intersect the mesh.
"""

from ..toolpath.generator import ToolpathGenerator

__all__ = ["slice_mesh", "extract_slice"]


def _delegate() -> ToolpathGenerator:
    """Return a ToolpathGenerator whose slicing methods can be used standalone.

    ``slice_mesh`` and ``extract_slice`` read no configuration state, so an
    uninitialised instance is sufficient and avoids creating an output
    directory as a side effect of importing this module.
    """
    return ToolpathGenerator.__new__(ToolpathGenerator)


def slice_mesh(mesh, layer_height=0.5, add_layers=0):
    """Slice ``mesh`` along the Z-axis into ``(layer_height, boundary)`` pairs.

    Delegates to :meth:`src.toolpath.generator.ToolpathGenerator.slice_mesh`.
    """
    return _delegate().slice_mesh(mesh, layer_height, add_layers)


def extract_slice(mesh, plane):
    """Extract the 2D cross-section of ``mesh`` at the given ``plane``.

    ``plane`` is a length-4 array holding the plane equation ``[nx, ny, nz, d]``.
    Delegates to :meth:`src.toolpath.generator.ToolpathGenerator.extract_slice`.
    """
    return _delegate().extract_slice(mesh, plane)
