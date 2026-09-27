"""Property-based test for experiment retention (spec: repo-reorganization).

Covers Property 8: every source file classified ``experiment`` in the
Functionality Inventory has a corresponding file under ``clear_repo/experiments/``.
"""

import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "clear_repo" / "docs" / "functionality_inventory.md"
EXPERIMENTS_ROOT = REPO_ROOT / "clear_repo" / "experiments"

VALID_CLASSIFICATIONS = {"reusable", "experiment", "script", "obsolete"}


def _experiment_entries():
    """Return inventory paths whose single classification label is ``experiment``."""
    text = INVENTORY_PATH.read_text(encoding="utf-8")
    entries = []

    sections = re.split(r"^### (.+)$", text, flags=re.MULTILINE)
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        if not heading.endswith(".py"):
            continue

        match = re.search(
            r"^\s*-\s*\*\*Classification\*\*:\s*(.+)$", body, flags=re.MULTILINE
        )
        if not match:
            continue
        line = match.group(1).lower()
        labels = {
            label
            for label in VALID_CLASSIFICATIONS
            if re.search(rf"\b{re.escape(label)}\b", line)
        }
        if labels == {"experiment"}:
            entries.append(heading)

    return sorted(entries)


def _retained_experiment_files():
    """Map basename -> relative paths of every Python file under experiments/."""
    retained = {}
    if EXPERIMENTS_ROOT.is_dir():
        for path in EXPERIMENTS_ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            retained.setdefault(path.name, []).append(
                path.relative_to(REPO_ROOT).as_posix()
            )
    return retained


EXPERIMENT_ENTRIES = _experiment_entries()
RETAINED_BY_NAME = _retained_experiment_files()


# **Feature: repo-reorganization, Property 8: Experiment scripts are retained**
# **Validates: Requirements 4.4**
@settings(max_examples=200)
@given(st.sampled_from(EXPERIMENT_ENTRIES))
def test_experiment_scripts_are_retained(entry_path):
    """For any file classified 'experiment' in the Functionality Inventory, a
    corresponding file exists under the experiments/ directory."""
    name = Path(entry_path).name
    assert name in RETAINED_BY_NAME, (
        f"Experiment {entry_path!r} has no corresponding file under "
        f"clear_repo/experiments/. Retained files: "
        f"{sorted(RETAINED_BY_NAME)}"
    )
