#!/usr/bin/env python3
"""Command-line entry point for the repair strategy editor GUI.

Launches the existing ``RepairEditor`` application, optionally pre-loading the
G-code/CSV toolpath, the damaged part mesh, and the repair volume mesh from
paths supplied as command-line arguments (Requirements 2.3, 5.2, 5.3). No input
path is hard-coded.

Usage::

    python scripts/gcode/repair_editor.py
    python scripts/gcode/repair_editor.py --gcode Results/test/toolpath_trajectory.csv \
        --damaged Data/Toolpath_Models/Model_C_S.STL \
        --repair Data/Toolpath_Models/Model_C_R.STL
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
        description="Launch the repair strategy editor GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gcode", help="G-code or CSV toolpath to load on start-up.")
    parser.add_argument("--damaged", help="Damaged part STL to load on start-up.")
    parser.add_argument("--repair", help="Repair volume STL to load on start-up.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gcode_path = require_file(parser, args.gcode, "G-code/CSV file")
    damaged_path = require_file(parser, args.damaged, "Damaged part STL")
    repair_path = require_file(parser, args.repair, "Repair volume STL")

    def build_and_run():
        editor_app = load_repo_module("repair_editor")

        app = editor_app.RepairEditor()
        load_with_path(editor_app, app.load_gcode, gcode_path)
        load_with_path(editor_app, app.load_damaged, damaged_path)
        load_with_path(editor_app, app.load_repair, repair_path)
        app.run()
        return 0

    return run_gui(build_and_run)


if __name__ == "__main__":
    sys.exit(main())
