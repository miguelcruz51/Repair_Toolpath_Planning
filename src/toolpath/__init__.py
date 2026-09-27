"""Toolpath: ToolpathConfig and ToolpathGenerator orchestration.

Relocated from the root-level ``AS4RCSD5.py`` script. Public signatures and
numerical behaviour are preserved; see :mod:`src.toolpath.generator`.
"""

from .generator import (
    ToolpathConfig,
    ToolpathGenerator,
    default_config,
    DEFAULT_DATA_DIR,
    DEFAULT_MESH_FILE,
    DEFAULT_NON_PLANAR_SURFACE_FILE,
    DEFAULT_OUTPUT_DIR,
)

__all__ = [
    "ToolpathConfig",
    "ToolpathGenerator",
    "default_config",
    "DEFAULT_DATA_DIR",
    "DEFAULT_MESH_FILE",
    "DEFAULT_NON_PLANAR_SURFACE_FILE",
    "DEFAULT_OUTPUT_DIR",
]
