"""Property-based test for README section completeness (spec: repo-reorganization).

Property 10 - for any required section name, ``clear_repo/README.md`` contains a
markdown heading matching that name. Headings are collected from the lines
starting with ``#`` and matched case-insensitively against a small set of
accepted aliases per required section (for example "support and over-deposition"
is accepted as "Support and over-deposition analysis").
"""

import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

CLEAR_REPO_ROOT = Path(__file__).resolve().parents[1]
README = CLEAR_REPO_ROOT / "README.md"

# Required section -> the heading phrases accepted as that section.
# Requirement 7.2 fixes the component sections; 7.3 the repository structure
# section; 7.4 the research context section.
REQUIRED_SECTIONS = {
    "geometry processing": ("geometry processing",),
    "reference frames": ("reference frames",),
    "boundary reconstruction": ("boundary reconstruction",),
    "slicing": ("slicing",),
    "deposition strategies": ("deposition strategies",),
    "toolpath planning": ("toolpath planning",),
    "deposition prediction": ("deposition prediction",),
    "support and over-deposition": (
        "support and over-deposition",
        "support and over-deposition analysis",
    ),
    "engineering metrics": ("engineering metrics",),
    "g-code tools": ("g-code tools",),
    "repository structure": ("repository structure",),
    "research context": ("research context",),
}


def _headings():
    """Normalized text of every markdown heading in the README."""
    headings = []
    in_code_fence = False
    for raw in README.read_text(encoding="utf-8").splitlines():
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


HEADINGS = _headings()


# **Feature: repo-reorganization, Property 10: README section completeness**
# **Validates: Requirements 7.2, 7.3, 7.4**
@settings(max_examples=200)
@given(st.sampled_from(sorted(REQUIRED_SECTIONS)))
def test_readme_section_completeness(section):
    """For any required section name, the README contains a heading matching
    that name."""
    accepted = REQUIRED_SECTIONS[section]
    assert any(heading in accepted for heading in HEADINGS), (
        f"README.md has no heading for the required section '{section}'. "
        f"Accepted headings: {list(accepted)}; found: {HEADINGS}"
    )
