"""Property-based tests for the clear_repo skeleton (spec: repo-reorganization).

These tests validate that each functional-area package directory under
``clear_repo/src/`` exists and is a well-formed Python package (contains an
``__init__.py`` file).
"""

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "clear_repo" / "src"

# The fixed set of functional areas defined by the design layout.
FUNCTIONAL_AREAS = [
    "geometry",
    "reference_frames",
    "slicing",
    "deposition",
    "toolpath",
    "evaluation",
    "gcode",
    "visualization",
]


# **Feature: repo-reorganization, Property 3: Functional-area directories are well-formed**
# **Validates: Requirements 3.1, 3.2**
@settings(max_examples=200)
@given(st.sampled_from(FUNCTIONAL_AREAS))
def test_functional_area_directories_are_well_formed(area):
    """For any functional area in the fixed set, the directory src/<area>/
    exists and contains an __init__.py file."""
    area_dir = SRC_ROOT / area
    assert area_dir.is_dir(), f"Expected functional-area directory {area_dir} to exist."
    init_file = area_dir / "__init__.py"
    assert init_file.is_file(), (
        f"Expected {init_file} to exist so that src/{area}/ is a well-formed package."
    )
