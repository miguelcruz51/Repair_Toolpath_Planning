"""Deposition: strategy generation, boundary reconstruction, and simulation."""

from .simulation import (
    simulate_deposition,
    simulate_material_deposition,
    import_trajectory_from_csv,
)

__all__ = [
    "simulate_deposition",
    "simulate_material_deposition",
    "import_trajectory_from_csv",
]
