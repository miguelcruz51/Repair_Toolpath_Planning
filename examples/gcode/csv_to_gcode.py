#!/usr/bin/env python3
"""Minimal example: convert a toolpath CSV into G-code and parse it back.

Demonstrates :class:`src.gcode.GCodeConverter` (``load_csv``,
``convert_to_gcode``, ``parse_gcode``, ``get_available_layers``,
``get_gcode_coordinates``) without the Tkinter UI.

The input CSV must have ``X, Y, Z, Layer, Type`` columns. The default sample is
expressed relative to the repository root and the output path is a command-line
argument (Requirements 5.2, 5.3).

Usage::

    python examples/gcode/csv_to_gcode.py
    python examples/gcode/csv_to_gcode.py --csv Results/test/layer_2_toolpath.csv
    python examples/gcode/csv_to_gcode.py --output /tmp/toolpath.gcode --layer 0
"""

import argparse
import sys
from pathlib import Path

# clear_repo/ is two levels up from examples/gcode/; the repository root is one
# level above that.
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CLEAR_REPO_ROOT.parent
if str(CLEAR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAR_REPO_ROOT))

from src.gcode import GCodeConverter  # noqa: E402

DEFAULT_TOOLPATH_CSV = REPO_ROOT / "Results" / "test" / "complete_3d_toolpath.csv"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for this example."""
    parser = argparse.ArgumentParser(
        description="Convert a toolpath CSV (X, Y, Z, Layer, Type) into G-code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_TOOLPATH_CSV),
        help="Toolpath CSV to convert (repository-relative default).",
    )
    parser.add_argument(
        "--output",
        help="Write the generated G-code here. Omit to print a preview only.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Emit only this layer instead of the whole toolpath.",
    )
    parser.add_argument(
        "--feedrate-deposition",
        type=int,
        default=1500,
        help="Feedrate for G01 deposition moves (mm/min).",
    )
    parser.add_argument(
        "--feedrate-travel",
        type=int,
        default=3000,
        help="Feedrate for G00 travel moves (mm/min).",
    )
    parser.add_argument(
        "--preview-lines", type=int, default=10, help="G-code lines to print."
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Toolpath CSV not found: {csv_path}", file=sys.stderr)
        return 1

    converter = GCodeConverter()
    if not converter.load_csv(str(csv_path)):
        print(
            f"Could not load {csv_path}; expected X, Y, Z, Layer, Type columns.",
            file=sys.stderr,
        )
        return 1

    # convert_to_gcode() populates converter.gcode_lines for the whole toolpath;
    # get_layer_gcode() returns a single layer without touching that state.
    full_gcode = converter.convert_to_gcode(
        feedrate_g01=args.feedrate_deposition,
        feedrate_g00=args.feedrate_travel,
    )
    gcode = converter.get_layer_gcode(args.layer) if args.layer is not None else full_gcode
    if not gcode:
        print(f"No G-code produced for layer {args.layer}.", file=sys.stderr)
        return 1

    # Parsing the generated G-code back recovers the motion coordinates.
    converter.parse_gcode()
    x_coords, _, _, _ = converter.get_gcode_coordinates()

    print(f"CSV:            {csv_path}")
    print(f"CSV rows:       {len(converter.df)}")
    print(f"Layers:         {converter.get_available_layers()}")
    print(f"G-code lines:   {len(gcode)}")
    print(f"Parsed moves:   {len(x_coords)}")
    print("Preview:")
    for line in gcode[: args.preview_lines]:
        print(f"  {line}")
    if len(gcode) > args.preview_lines:
        print(f"  ... {len(gcode) - args.preview_lines} more lines")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(gcode) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
