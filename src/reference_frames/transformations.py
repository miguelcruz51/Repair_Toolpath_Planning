"""Coordinate transforms, rotations, and axis remaps.

The transform implementations live on
:class:`src.toolpath.generator.ToolpathGenerator` (relocated verbatim from
``AS4RCSD5.py``). This module exposes them as plain functions by delegating to
the preserved methods, so reference-frame work can be done without running the
full toolpath pipeline. The numerical behaviour is unchanged.

Inputs
    An Open3D ``TriangleMesh`` (``rotate_mesh``) or an ``(N, 3)`` coordinate
    array (``transform_coordinates``, ``rotate_coordinates``), plus offsets,
    axis names, and angles in degrees.

Outputs
    The rotated mesh, or the transformed coordinate array.
"""

from ..toolpath.generator import ToolpathGenerator

__all__ = [
    "rotate_mesh",
    "rotate_points",
    "transform_coordinates",
    "rotate_coordinates",
]


def _delegate() -> ToolpathGenerator:
    """Return a ToolpathGenerator whose transform methods can be used standalone.

    These methods read no configuration state, so an uninitialised instance is
    sufficient and avoids creating an output directory on import.
    """
    return ToolpathGenerator.__new__(ToolpathGenerator)


def rotate_mesh(mesh, axis, angle, center=None):
    """Rotate ``mesh`` about ``axis`` ("x", "y", or "z") by ``angle`` degrees.

    Delegates to :meth:`src.toolpath.generator.ToolpathGenerator.rotate_mesh`.
    """
    return _delegate().rotate_mesh(mesh, axis, angle, center)


def rotate_points(points, angle, origin):
    """Rotate 2D ``points`` around ``origin`` by ``angle`` degrees.

    ``origin`` is a Shapely ``Point``-like object exposing ``.x`` and ``.y``.
    Delegates to :meth:`src.toolpath.generator.ToolpathGenerator.rotate_points`.
    """
    return _delegate().rotate_points(points, angle, origin)


def transform_coordinates(coords, offset_x, offset_y, offset_z,
                          remap_x, remap_y, remap_z):
    """Apply offsets and an axis remap to an ``(N, 3)`` coordinate array.

    Delegates to
    :meth:`src.toolpath.generator.ToolpathGenerator._transform_coordinates`.
    """
    return _delegate()._transform_coordinates(
        coords, offset_x, offset_y, offset_z, remap_x, remap_y, remap_z
    )


def rotate_coordinates(coords, axis, angle_degrees):
    """Rotate an ``(N, 3)`` coordinate array about ``axis`` by ``angle_degrees``.

    Delegates to
    :meth:`src.toolpath.generator.ToolpathGenerator._rotate_coordinates`.
    """
    return _delegate()._rotate_coordinates(coords, axis, angle_degrees)
