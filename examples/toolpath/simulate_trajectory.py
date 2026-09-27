#!/usr/bin/env python3
"""Minimal example: animate a deposition trajectory read from a CSV file.

Demonstrates :mod:`src.deposition.simulation`
(``import_trajectory_from_csv``, ``simulate_deposition``,
``simulate_material_deposition``).

The trajectory CSV path is a command-line argument and defaults to a sample file
expressed relative to the repository root (Requirements 5.2, 5.3). The CSV has a
header row followed by X, Y, Z columns.

Usage::

    python examples/toolpath/simulate_trajectory.py --summary-only
    python examples/toolpath/simulate_trajectory.py
    python examples/toolpath/simulate_trajectory.py --mode material --interval 0.02
    python examples/toolpath/simulate_trajectory.py --csv results/toolpath/toolpath_trajectory.csv
"""

import argparse
import sys
from pathlib import Path

# clear_repo/ is two levels up from examples/toolpath/; the repository root is
# one level above that.
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CLEAR_REPO_ROOT.parent
if str(CLEAR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAR_REPO_ROOT))

from src.deposition import (  # noqa: E402
    import_trajectory_from_csv,
    simulate_deposition,
    simulate_material_deposition,
)

DEFAULT_TRAJECTORY_CSV = REPO_ROOT / "Part1_ZigZag_Toolpath.csv"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for this example."""
    parser = argparse.ArgumentParser(
        description="Animate a deposition trajectory stored as an X,Y,Z CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_TRAJECTORY_CSV),
        help="Trajectory CSV with a header row and X, Y, Z columns.",
    )
    parser.add_argument(
        "--mode",
        choices=["path", "material"],
        default="path",
        help="'path' draws line segments; 'material' draws deposited spheres.",
    )
    parser.add_argument(
        "--interval", type=float, default=0.01, help="Seconds between frames."
    )
    parser.add_argument(
        "--material-radius",
        type=float,
        default=0.5,
        help="Deposited bead radius used by the 'material' mode.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200,
        help="Limit on trajectory points, keeping the animation short.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the trajectory summary and exit (useful without a display).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Trajectory CSV not found: {csv_path}", file=sys.stderr)
        return 1

    trajectory = import_trajectory_from_csv(str(csv_path))
    if len(trajectory) == 0:
        print(f"Trajectory CSV has no points: {csv_path}", file=sys.stderr)
        return 1

    if args.max_points > 0:
        trajectory = trajectory[: args.max_points]

    print(f"Trajectory:  {csv_path}")
    print(f"Points:      {len(trajectory)}")
    print(f"X range:     {trajectory[:, 0].min():.3f} .. {trajectory[:, 0].max():.3f}")
    print(f"Y range:     {trajectory[:, 1].min():.3f} .. {trajectory[:, 1].max():.3f}")
    print(f"Z range:     {trajectory[:, 2].min():.3f} .. {trajectory[:, 2].max():.3f}")

    if args.summary_only:
        return 0

    if args.mode == "material":
        simulate_material_deposition(
            trajectory,
            material_radius=args.material_radius,
            interval=args.interval,
        )
    else:
        simulate_deposition(trajectory, interval=args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
