#!/usr/bin/env python3
"""Command-line entry point for batch toolpath evaluation.

Wraps the batch evaluation logic of the original root-level
``batch_evaluation.py``: run the toolpath pipeline once per configuration,
collect the resulting Deposition_Metric values, and write a combined CSV plus a
text summary.

Unlike the original script, the test matrix is not embedded in the source. The
configurations are read from a directory of JSON files (or from explicit file
arguments), so no input path is hard-coded to a specific machine
(Requirements 2.3, 5.1, 5.2, 5.3).

Usage::

    python scripts/evaluation/batch_evaluate.py --configs configs/
    python scripts/evaluation/batch_evaluate.py --config a.json --config b.json \
        --output-dir results/batch --timeout 900
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# The package root is clear_repo/, two levels up from scripts/evaluation/.
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CLEAR_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAR_REPO_ROOT))

RUN_PIPELINE = CLEAR_REPO_ROOT / "scripts" / "toolpath" / "run_pipeline.py"
RESULTS_JSON_NAME = "evaluation_overall_strategy.json"

# Metric fields lifted from the evaluation JSON, with their fallback values.
METRIC_FIELDS = {
    "total_layers": 0,
    "total_deposition_length": 0,
    "total_travel_length": 0,
    "dep_to_travel_ratio": 0,
    "avg_turn_angle": 0,
    "avg_canpd": float("inf"),
    "avg_coverage_pct": 0,
    "avg_over_deposition_pct": 0,
    "avg_supported_deposition_pct": 0,
    "avg_supporting_contour_pct": 0,
    "avg_wall_engagement_index": 0,
    "std_coverage_pct": 0,
    "std_over_deposition_pct": 0,
}

logger = logging.getLogger("batch_evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the batch evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Run the toolpath pipeline for several configurations and "
        "collect their evaluation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        help="Directory containing one JSON configuration file per test case.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        dest="config_files",
        metavar="FILE",
        help="An individual JSON configuration file. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/batch",
        help="Directory for per-case outputs, the results CSV, and the summary.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-case timeout in seconds.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def collect_config_files(configs_dir, config_files) -> list:
    """Return the sorted list of configuration files to evaluate."""
    paths = []
    if configs_dir:
        directory = Path(configs_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Configs directory not found: {directory}")
        paths.extend(sorted(directory.glob("*.json")))
    paths.extend(Path(item) for item in config_files)

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Configuration file(s) not found: {missing}")
    return paths


def prepare_case_config(config_file: Path, output_root: Path) -> tuple:
    """Write a copy of ``config_file`` whose output_dir is case-specific.

    Returns ``(case_name, prepared_config_path, case_output_dir)``.
    """
    with open(config_file, "r") as handle:
        config = json.load(handle)

    case_name = config.get("name") or config_file.stem
    case_output_dir = output_root / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # 'name' is a batch-level label, not a ToolpathConfig field.
    config.pop("name", None)
    config["output_dir"] = str(case_output_dir)

    prepared = case_output_dir / "config.json"
    with open(prepared, "w") as handle:
        json.dump(config, handle, indent=2)

    return case_name, prepared, case_output_dir


def run_case(config_path: Path, case_name: str, timeout: float) -> dict:
    """Run the pipeline for one configuration and report the outcome."""
    logger.info("Running %s", case_name)
    start = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, str(RUN_PIPELINE), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        logger.error("%s timed out after %.2fs", case_name, elapsed)
        return {"success": False, "time": elapsed, "warnings": 0, "errors": 1}

    elapsed = time.time() - start
    output = (completed.stdout or "") + (completed.stderr or "")
    result = {
        "success": completed.returncode == 0,
        "time": elapsed,
        "warnings": output.count("WARNING"),
        "errors": output.count("ERROR"),
    }
    if not result["success"]:
        logger.error(
            "%s failed (exit %s): %s",
            case_name,
            completed.returncode,
            output.strip()[-500:],
        )
    return result


def load_metrics(case_output_dir: Path) -> dict:
    """Read the evaluation metrics for a completed case, if present."""
    json_file = Path(case_output_dir) / RESULTS_JSON_NAME
    if not json_file.is_file():
        return {}
    try:
        with open(json_file, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read %s: %s", json_file, error)
        return {}


def build_row(case_name: str, config_path: Path, run_result: dict, metrics: dict) -> dict:
    """Assemble one CSV row from a case's configuration, run result, and metrics."""
    with open(config_path, "r") as handle:
        config = json.load(handle)

    row = {
        "test_name": case_name,
        "strategy": config.get("strategy"),
        "tool_width": config.get("tool_width"),
        "hatch_distance": config.get("hatch_distance"),
        "grid_size": config.get("grid_size"),
        "use_optimized_hybrid2": config.get("use_optimized_hybrid2", False),
        "success": run_result["success"],
        "time_seconds": run_result["time"],
        "warnings": run_result["warnings"],
        "errors": run_result["errors"],
    }
    for field, fallback in METRIC_FIELDS.items():
        row[field] = metrics.get(field, fallback)
    return row


def write_results(rows: list, results_file: Path, summary_file: Path) -> None:
    """Write the combined CSV and the human-readable summary."""
    if rows:
        with open(results_file, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    successful = sum(1 for row in rows if row["success"])
    with open(summary_file, "w") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("BATCH EVALUATION SUMMARY\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Total tests: {len(rows)}\n")
        handle.write(f"Successful: {successful}\n")
        handle.write(f"Failed: {len(rows) - successful}\n")
        handle.write(
            f"Total time: {sum(row['time_seconds'] for row in rows):.2f}s\n\n"
        )
        handle.write("=" * 80 + "\n")
        handle.write("INDIVIDUAL RESULTS\n")
        handle.write("=" * 80 + "\n\n")
        for row in rows:
            handle.write(f"Test: {row['test_name']}\n")
            handle.write(f"  Strategy: {row['strategy']}\n")
            handle.write(f"  Tool Width: {row['tool_width']} mm\n")
            handle.write(f"  Hatch Distance: {row['hatch_distance']} mm\n")
            handle.write(f"  Success: {row['success']}\n")
            handle.write(f"  Time: {row['time_seconds']:.2f}s\n")
            handle.write(f"  Warnings: {row['warnings']}\n")
            handle.write(f"  Errors: {row['errors']}\n")
            if row["success"]:
                handle.write(f"  Layers: {row['total_layers']}\n")
                handle.write(f"  CANPD: {row['avg_canpd']}\n")
                handle.write(f"  Coverage: {row['avg_coverage_pct']}%\n")
                handle.write(f"  WEI: {row['avg_wall_engagement_index']}\n")
            handle.write("\n")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config_files = collect_config_files(args.configs, args.config_files)
    except FileNotFoundError as error:
        parser.error(str(error))

    if not config_files:
        parser.error("No configurations given. Use --configs <dir> or --config <file>.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_root / f"batch_results_{timestamp}.csv"
    summary_file = output_root / f"batch_summary_{timestamp}.txt"

    rows = []
    for index, config_file in enumerate(config_files, 1):
        logger.info("[%d/%d] %s", index, len(config_files), config_file)
        case_name, prepared, case_output_dir = prepare_case_config(
            config_file, output_root
        )
        run_result = run_case(prepared, case_name, args.timeout)
        metrics = load_metrics(case_output_dir)
        rows.append(build_row(case_name, prepared, run_result, metrics))

    write_results(rows, results_file, summary_file)
    successful = sum(1 for row in rows if row["success"])
    logger.info(
        "Completed %d/%d cases. Results: %s | Summary: %s",
        successful,
        len(rows),
        results_file,
        summary_file,
    )
    return 0 if successful == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
