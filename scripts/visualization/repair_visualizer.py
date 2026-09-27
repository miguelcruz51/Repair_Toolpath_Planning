#!/usr/bin/env python3
"""Command-line entry point for the advanced repair process visualizer GUI.

Launches the existing ``AdvancedRepairVisualizer`` application, optionally
pre-loading the damaged part mesh, the repair volume mesh, and a toolpath
(G-code or CSV) from paths supplied as command-line arguments
(Requirements 2.3, 5.2, 5.3). No input path is hard-coded.

Usage::

    python scripts/visualization/repair_visualizer.py
    python scripts/visualization/repair_visualizer.py \
        --damaged Data/Toolpath_Models/Model_C_S.STL \
        --repair Data/Toolpath_Models/Model_C_R.STL \
        --gcode Results/test/toolpath_trajectory.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_support import (  # noqa: E402
    add_repo_root_to_path,
    load_repo_module,
    load_with_path,
    require_file,
    run_gui,
)

add_repo_root_to_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the advanced repair process visualizer GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--damaged", help="Damaged part STL to load on start-up.")
    parser.add_argument("--repair", help="Repair volume STL to load on start-up.")
    parser.add_argument("--gcode", help="G-code or CSV toolpath to load on start-up.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    damaged_path = require_file(parser, args.damaged, "Damaged part STL")
    repair_path = require_file(parser, args.repair, "Repair volume STL")
    gcode_path = require_file(parser, args.gcode, "G-code/CSV file")

    def build_and_run():
        visualizer_app = load_repo_module("visualizer_advanced")

        app = visualizer_app.AdvancedRepairVisualizer()
        load_with_path(visualizer_app, app.load_damaged, damaged_path)
        load_with_path(visualizer_app, app.load_repair, repair_path)
        load_with_path(visualizer_app, app.load_gcode, gcode_path)
        app.run()
        return 0

    return run_gui(build_and_run)


if __name__ == "__main__":
    sys.exit(main())
