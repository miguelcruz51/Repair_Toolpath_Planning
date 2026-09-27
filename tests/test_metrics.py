"""Property-based test for relocated evaluation metrics (spec: repo-reorganization).

Validates that the metric functions relocated into
``clear_repo/src/evaluation/metrics.py`` produce numerically identical results
to the original root-level ``evaluation_metrics.py`` implementation on the same
input.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The relocated package module and the original root-level module.
relocated = _load(
    "relocated_evaluation_metrics",
    REPO_ROOT / "clear_repo" / "src" / "evaluation" / "metrics.py",
)
original = _load("original_evaluation_metrics", REPO_ROOT / "evaluation_metrics.py")


_finite_floats = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False
)


@st.composite
def _paths(draw):
    """A 2D path. NaN rows are allowed: they represent tool moves, which the
    length metric is specified to skip."""
    n = draw(st.integers(min_value=0, max_value=25))
    points = []
    for _ in range(n):
        if draw(st.integers(min_value=0, max_value=9)) == 0:
            points.append([float("nan"), float("nan")])
        else:
            points.append([draw(_finite_floats), draw(_finite_floats)])
    return np.array(points, dtype=float).reshape(-1, 2)


def _equal(a, b):
    """Numerical equality treating NaN as equal to NaN and matching infinities."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
    return np.allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


# **Feature: repo-reorganization, Property 7: Metric numerical behavior preserved**
# **Validates: Requirements 4.2**
@settings(max_examples=200)
@given(
    path=_paths(),
    deposition_length=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    travel_length=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    num_samples=st.integers(min_value=1, max_value=36),
)
def test_relocated_metrics_preserve_numerical_behavior(
    path, deposition_length, travel_length, num_samples
):
    """For any valid input, each relocated Deposition_Metric function returns the
    same value as the original implementation on the same input."""
    assert _equal(
        relocated.calculate_total_length(path),
        original.calculate_total_length(path),
    )

    assert _equal(
        relocated.deposition_to_travel_ratio(deposition_length, travel_length),
        original.deposition_to_travel_ratio(deposition_length, travel_length),
    )

    toolpaths = [(path, None, None, None)]
    assert _equal(
        relocated.turn_angle_analysis(toolpaths),
        original.turn_angle_analysis(toolpaths),
    )

    if len(path) > 0:
        assert _equal(
            relocated.calculate_centroid(path),
            original.calculate_centroid(path),
        )

        centroid = relocated.calculate_centroid(path)
        assert _equal(
            relocated.radial_sample(path, centroid, num_samples),
            original.radial_sample(path, centroid, num_samples),
        )
