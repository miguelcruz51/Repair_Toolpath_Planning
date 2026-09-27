#!/usr/bin/env python3
"""Command-line entry point for the toolpath generation pipeline.

Wraps :class:`src.toolpath.ToolpathGenerator`. All inputs arrive through a JSON
configuration file and/or command-line arguments; no input path is hard-coded to
a specific machine (Requirements 2.3, 5.1, 5.2, 5.3).

Usage::

    python scripts/toolpath/run_pipeline.py --config example_config.json
    python scripts/toolpath/run_pipeline.py --config cfg.json --output-dir results/run1
    python scripts/toolpath/run_pipeline.py --mesh-file Data/Toolpath_Models/Model_C_R.STL

When neither ``--config`` nor ``--mesh-file`` is given, the repository-relative
defaults from :func:`src.toolpath.default_config` are used.
"""

import argparse
import logging
import sys
from pathlib import Path

# The package root is clear_repo/, two levels up from scripts/toolpath/.
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CLEAR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAR_REPO_ROOT))

from src.toolpath import ToolpathConfig, ToolpathGenerator, default_config  # noqa: E402

logger = logging.getLogger("run_pipeline")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="Run the AS4RCSD toolpath generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON configuration file (see ToolpathConfig fields).",
    )
    parser.add_argument(
        "--mesh-file",
        help="Path to the repair volume mesh (STL). Overrides the config value.",
    )
    parser.add_argument(
        "--non-planar-surface",
        help="Path to the non-planar substrate surface mesh (STL). "
        "Overrides the config value.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated results. Overrides the config value.",
    )
    parser.add_argument(
        "--save-config",
        help="Write the effective configuration to this JSON file before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the configuration and exit without running the pipeline.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Errors only.")
    return parser


def resolve_config(args: argparse.Namespace) -> ToolpathConfig:
    """Build the effective :class:`ToolpathConfig` from CLI arguments.

    Precedence: explicit command-line paths override the configuration file,
    which overrides the repository-relative defaults.
    """
    if args.config:
        config = ToolpathConfig.from_json(args.config)
    elif args.mesh_file:
        config = ToolpathConfig(mesh_file=args.mesh_file)
    else:
        config = default_config()

    if args.mesh_file:
        config.mesh_file = args.mesh_file
    if args.non_planar_surface:
        config.non_planar_surface_file = args.non_planar_surface
    if args.output_dir:
        config.output_dir = args.output_dir

    return config


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = resolve_config(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"Could not build configuration: {error}")

    if not Path(config.mesh_file).is_file():
        parser.error(
            f"Mesh file not found: {config.mesh_file}. "
            f"Pass --mesh-file or set 'mesh_file' in the configuration file."
        )

    if args.save_config:
        config.save_json(args.save_config)
        logger.info("Configuration written to %s", args.save_config)

    if args.dry_run:
        logger.info("Dry run: mesh=%s output_dir=%s", config.mesh_file, config.output_dir)
        return 0

    try:
        ToolpathGenerator(config).run()
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130
    except Exception as error:  # pipeline failures must yield a non-zero status
        logger.error("Pipeline failed: %s", error)
        if args.verbose:
            raise
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
