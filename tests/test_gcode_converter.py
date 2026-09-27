"""Property-based test for the relocated G-code converter (spec: repo-reorganization).

Validates the parser round-trip companion of Property 7: coordinate lines
emitted in the converter's own output format parse back to the same
coordinates, commands, and feedrates.
"""

import importlib.util
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter_module = _load(
    "relocated_gcode_converter",
    REPO_ROOT / "clear_repo" / "src" / "gcode" / "converter.py",
)
GCodeConverter = converter_module.GCodeConverter


# Coordinates are emitted with three decimals, so the printable domain is
# multiples of 0.001 within the machine's working envelope.
_coordinate = st.integers(min_value=-1_000_000, max_value=1_000_000).map(
    lambda thousandths: thousandths / 1000.0
)
_feedrate = st.integers(min_value=1, max_value=10_000)
_command = st.sampled_from(["G00", "G01"])

_move = st.tuples(_command, _coordinate, _coordinate, _coordinate, _feedrate)


# **Feature: repo-reorganization, Property 7 (parser round-trip companion): generated coordinate lines parse back to the same coordinates**
# **Validates: Requirements 4.1**
@settings(max_examples=200)
@given(st.lists(_move, min_size=1, max_size=20), st.integers(min_value=0, max_value=99))
def test_generated_coordinate_lines_parse_back_to_same_coordinates(moves, layer):
    """For any sequence of moves rendered in the converter's output format,
    parsing those lines recovers the original command, coordinates, and
    feedrate, and coordinate extraction reproduces the input sequence."""
    converter = GCodeConverter()
    converter.gcode_lines = [f"; Layer {layer}"] + [
        f"{command} X{x:.3f} Y{y:.3f} Z{z:.3f} F{f}"
        for command, x, y, z, f in moves
    ]

    parsed = converter.parse_gcode()

    # The layer comment must not be read as a motion command.
    assert parsed[0]["type"] == "comment"
    assert parsed[0]["x"] is None

    for (command, x, y, z, f), entry in zip(moves, parsed[1:]):
        assert entry["type"] == "gcode"
        assert entry["command"] == command
        assert entry["x"] == x
        assert entry["y"] == y
        assert entry["z"] == z
        assert entry["f"] == float(f)

    x_coords, y_coords, z_coords, commands = converter.get_gcode_coordinates()
    assert x_coords == [x for _, x, _, _, _ in moves]
    assert y_coords == [y for _, _, y, _, _ in moves]
    assert z_coords == [z for _, _, _, z, _ in moves]
    assert commands == [command for command, _, _, _, _ in moves]
