"""Property-based test for per-area documentation completeness (spec: repo-reorganization).

Property 11 - for any functional-area documentation file under ``clear_repo/docs/``,
the file contains a section for each of: purpose, inputs, processing, outputs,
example, and dependencies.

``docs/functionality_inventory.md`` is excluded: it is the audit artifact from
Requirement 1, not a functional-area document.
"""

import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

CLEAR_REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = CLEAR_REPO_ROOT / "docs"

# Documentation files that are not per-area documents.
NON_AREA_DOCS = {"functionality_inventory.md"}

# Required section -> accepted heading phrases for that section.
REQUIRED_SECTIONS = {
    "purpose": ("purpose",),
    "inputs": ("inputs", "input"),
    "processing": ("processing",),
    "outputs": ("outputs", "output"),
    "example": ("example", "examples", "example invocation"),
    "dependencies": ("dependencies",),
}


def _area_docs():
    """Every functional-area documentation file under docs/."""
    return sorted(
        path.name
        for path in DOCS_DIR.glob("*.md")
        if path.name not in NON_AREA_DOCS
    )


def _headings(path: Path):
    """Normalized text of every markdown heading in ``path``, skipping code fences."""
    headings = []
    in_code_fence = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if match:
            headings.append(match.group(1).strip().lower())
    return headings


AREA_DOCS = _area_docs()


def test_area_docs_exist():
    """The six functional-area documents required by the design are present."""
    expected = {
        "architecture.md",
        "slicing.md",
        "toolpath_planning.md",
        "deposition_strategies.md",
        "coordinate_transformations.md",
        "evaluation.md",
    }
    missing = sorted(expected - set(AREA_DOCS))
    assert not missing, f"Missing functional-area documentation files: {missing}"


# **Feature: repo-reorganization, Property 11: Per-area documentation completeness**
# **Validates: Requirements 9.1**
@settings(max_examples=200)
@given(
    st.sampled_from(_area_docs()),
    st.sampled_from(sorted(REQUIRED_SECTIONS)),
)
def test_per_area_documentation_completeness(doc_name, section):
    """For any functional-area documentation file, the file contains a section
    for each of purpose, inputs, processing, outputs, example, and dependencies."""
    path = DOCS_DIR / doc_name
    headings = _headings(path)
    accepted = REQUIRED_SECTIONS[section]
    assert any(heading in accepted for heading in headings), (
        f"docs/{doc_name} has no '{section}' section. "
        f"Accepted headings: {list(accepted)}; found: {headings}"
    )
