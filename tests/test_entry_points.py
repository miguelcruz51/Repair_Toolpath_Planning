"""Unit tests for the command-line entry-point scripts (spec: repo-reorganization).

Covers Requirements 2.3 (entry points live under ``scripts/`` and invoke the
reusable modules), 5.2 (input paths arrive as command-line arguments), and
5.3 (no machine-specific input paths).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAR_REPO_ROOT = REPO_ROOT / "clear_repo"
SCRIPTS_ROOT = CLEAR_REPO_ROOT / "scripts"

# Same pattern used by the Property 5 test for reusable modules.
MACHINE_PATH_RE = re.compile(
    r"(/home/[^/\s\"']+/|/Users/[^/\s\"']+/|[A-Za-z]:\\+Users\\+)"
)

ENTRY_POINTS = [
    "scripts/toolpath/run_pipeline.py",
    "scripts/evaluation/batch_evaluate.py",
    "scripts/gcode/gcode_editor.py",
    "scripts/gcode/repair_editor.py",
    "scripts/visualization/repair_visualizer.py",
]


def _load(rel_path: str):
    """Import an entry-point script by file path under a unique module name."""
    path = CLEAR_REPO_ROOT / rel_path
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    name = "entrypoint_" + rel_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rel_path", ENTRY_POINTS)
def test_entry_point_exists(rel_path):
    assert (CLEAR_REPO_ROOT / rel_path).is_file(), f"Missing entry point {rel_path}"


@pytest.mark.parametrize("rel_path", ENTRY_POINTS)
def test_entry_point_has_no_machine_specific_path(rel_path):
    """Requirement 5.3: entry points must not embed machine-specific paths."""
    offenders = [
        (lineno, line.strip())
        for lineno, line in enumerate(
            (CLEAR_REPO_ROOT / rel_path).read_text(encoding="utf-8").splitlines(), 1
        )
        if MACHINE_PATH_RE.search(line)
    ]
    assert not offenders, f"{rel_path} contains machine-specific paths: {offenders}"


@pytest.mark.parametrize("rel_path", ENTRY_POINTS)
def test_entry_point_accepts_input_paths_as_arguments(rel_path):
    """Requirement 5.2: every entry point exposes its inputs as CLI options."""
    parser = _load(rel_path).build_parser()
    options = {
        option for action in parser._actions for option in action.option_strings
    }
    assert options - {"-h", "--help"}, f"{rel_path} exposes no input arguments"


def test_run_pipeline_uses_config_file(tmp_path):
    """Requirement 2.3/5.2: --config supplies the pipeline inputs."""
    module = _load("scripts/toolpath/run_pipeline.py")
    mesh = tmp_path / "part.STL"
    mesh.write_text("solid part\nendsolid part\n")
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps({"mesh_file": str(mesh), "output_dir": str(tmp_path / "out")})
    )

    args = module.build_parser().parse_args(["--config", str(config_file)])
    config = module.resolve_config(args)

    assert config.mesh_file == str(mesh)
    assert config.output_dir == str(tmp_path / "out")


def test_run_pipeline_cli_overrides_config(tmp_path):
    """Explicit command-line paths take precedence over the configuration file."""
    module = _load("scripts/toolpath/run_pipeline.py")
    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({"mesh_file": "from_config.STL"}))

    args = module.build_parser().parse_args(
        [
            "--config",
            str(config_file),
            "--mesh-file",
            "from_cli.STL",
            "--output-dir",
            "cli_out",
        ]
    )
    config = module.resolve_config(args)

    assert config.mesh_file == "from_cli.STL"
    assert config.output_dir == "cli_out"


def test_run_pipeline_defaults_are_repo_relative():
    """Requirement 5.3: default data locations resolve inside the repository."""
    module = _load("scripts/toolpath/run_pipeline.py")
    args = module.build_parser().parse_args([])
    config = module.resolve_config(args)

    assert Path(config.mesh_file).is_absolute()
    assert Path(config.mesh_file).is_relative_to(REPO_ROOT)


def test_run_pipeline_reports_missing_mesh(tmp_path, capsys):
    """A missing input file produces a clean argument error, not a traceback."""
    module = _load("scripts/toolpath/run_pipeline.py")
    with pytest.raises(SystemExit):
        module.main(["--mesh-file", str(tmp_path / "absent.STL")])
    assert "not found" in capsys.readouterr().err


def test_batch_evaluate_collects_configs_from_directory(tmp_path):
    """Requirement 2.3/5.2: the batch runner takes a configs directory."""
    module = _load("scripts/evaluation/batch_evaluate.py")
    (tmp_path / "b.json").write_text(json.dumps({"strategy": 2}))
    (tmp_path / "a.json").write_text(json.dumps({"strategy": 1}))
    (tmp_path / "notes.txt").write_text("ignored")

    found = module.collect_config_files(str(tmp_path), [])

    assert [path.name for path in found] == ["a.json", "b.json"]


def test_batch_evaluate_prepares_case_output_dir(tmp_path):
    """Each case gets its own output directory and a normalized config copy."""
    module = _load("scripts/evaluation/batch_evaluate.py")
    config_file = tmp_path / "case.json"
    config_file.write_text(json.dumps({"name": "Raster_HD06", "strategy": 1}))
    output_root = tmp_path / "batch"

    name, prepared, case_dir = module.prepare_case_config(config_file, output_root)
    written = json.loads(prepared.read_text())

    assert name == "Raster_HD06"
    assert case_dir == output_root / "Raster_HD06"
    assert written["output_dir"] == str(case_dir)
    assert "name" not in written


def test_batch_evaluate_rejects_missing_configs_dir():
    module = _load("scripts/evaluation/batch_evaluate.py")
    with pytest.raises(FileNotFoundError):
        module.collect_config_files("no/such/directory", [])
