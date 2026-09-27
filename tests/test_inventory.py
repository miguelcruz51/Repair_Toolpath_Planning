"""Property-based tests for the Functionality Inventory (spec: repo-reorganization).

These tests validate the generated inventory document at
``clear_repo/docs/functionality_inventory.md`` against the original repository's
Python source files.
"""

import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "clear_repo" / "docs" / "functionality_inventory.md"

VALID_CLASSIFICATIONS = {"reusable", "experiment", "script", "obsolete"}

# Directories that hold generated / dependency code and are NOT original sources.
EXCLUDED_DIR_PARTS = {"clear_repo", "venv", "__pycache__", ".git"}


def _original_python_files():
    """Return the set of original Python source file paths (relative to repo root).

    Scope per the task: every ``.py`` file in the repository root and in
    ``Dataset/Boolean Comparison/``.
    """
    files = set()
    for path in REPO_ROOT.glob("*.py"):
        files.add(path.relative_to(REPO_ROOT).as_posix())
    bc_dir = REPO_ROOT / "Dataset" / "Boolean Comparison"
    if bc_dir.is_dir():
        for path in bc_dir.glob("*.py"):
            files.add(path.relative_to(REPO_ROOT).as_posix())
    return files


def _parse_inventory_entries():
    """Parse the inventory into a mapping of {file_path: classification_labels}.

    Each entry is a ``### <path>`` heading followed (within its section) by a
    ``**Classification**: <labels>`` line. The classification text may name more
    than one destination role (e.g. "reusable (with a script entry point)"), so
    the parser records every valid label token that appears on the line.
    """
    text = INVENTORY_PATH.read_text(encoding="utf-8")
    entries = {}

    # Split on level-3 headings.
    sections = re.split(r"^### (.+)$", text, flags=re.MULTILINE)
    # sections[0] is the preamble; then alternating (heading, body).
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""

        # Only treat headings that look like file paths (end in .py).
        if not heading.endswith(".py"):
            continue

        class_match = re.search(r"^\s*-\s*\*\*Classification\*\*:\s*(.+)$", body,
                                flags=re.MULTILINE)
        labels = set()
        if class_match:
            class_line = class_match.group(1).lower()
            for label in VALID_CLASSIFICATIONS:
                if re.search(rf"\b{re.escape(label)}\b", class_line):
                    labels.add(label)
        entries[heading] = labels

    return entries


ORIGINAL_FILES = sorted(_original_python_files())
INVENTORY_ENTRIES = _parse_inventory_entries()
INVENTORY_PATHS = set(INVENTORY_ENTRIES.keys())


# **Feature: repo-reorganization, Property 1: Inventory completeness**
# **Validates: Requirements 1.1**
@settings(max_examples=200)
@given(st.sampled_from(ORIGINAL_FILES))
def test_inventory_completeness(source_file):
    """For any Python source file in the original repository, the inventory
    contains an entry whose path matches that file."""
    assert source_file in INVENTORY_PATHS, (
        f"Source file {source_file!r} is missing from the inventory. "
        f"Inventoried paths: {sorted(INVENTORY_PATHS)}"
    )


# **Feature: repo-reorganization, Property 2: Single valid classification per entry**
# **Validates: Requirements 1.2**
@settings(max_examples=200)
@given(st.sampled_from(sorted(INVENTORY_PATHS)))
def test_single_valid_classification_per_entry(entry_path):
    """For any entry in the inventory, its classification label is exactly one
    member of {reusable, experiment, script, obsolete}."""
    labels = INVENTORY_ENTRIES[entry_path]
    assert len(labels) == 1, (
        f"Entry {entry_path!r} must have exactly one classification from "
        f"{sorted(VALID_CLASSIFICATIONS)}, but found {sorted(labels)}."
    )
