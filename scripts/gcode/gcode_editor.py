#!/usr/bin/env python3
"""Command-line entry point for the G-code converter and editor GUI.

Launches the existing ``GCodeGUI`` application, optionally pre-loading a
trajectory CSV or a G-code file whose path is supplied as a command-line
argument (Requirements 2.3, 5.2, 5.3). No input path is hard-coded.

Usage::

    python scripts/gcode/gcode_editor.py
    python scripts/gcode/gcode_editor.py --csv Results/test/toolpath_trajectory.csv
    python scripts/gcode/gcode_editor.py --gcode path/to/part.gcode
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
        description="Launch the G-code converter and editor GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", help="Trajectory CSV file to load on start-up.")
    parser.add_argument("--gcode", help="G-code file to load on start-up.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_path = require_file(parser, args.csv, "CSV file")
    gcode_path = require_file(parser, args.gcode, "G-code file")

    def build_and_run():
        import tkinter as tk

        gcode_app = load_repo_module("gcode_dev_v2")

        root = tk.Tk()
        app = gcode_app.GCodeGUI(root)
        load_with_path(gcode_app, app.load_csv, csv_path)
        load_with_path(gcode_app, app.load_gcode, gcode_path)
        root.mainloop()
        return 0

    return run_gui(build_and_run)


if __name__ == "__main__":
    sys.exit(main())
