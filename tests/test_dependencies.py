"""Property-based test for the generated dependency list (spec: repo-reorganization).

Property 9 - every third-party package imported by code under ``src/``,
``scripts/``, or ``examples/`` appears in ``clear_repo/requirements.txt``.

Imports are collected with :mod:`ast` (no module is executed). First-party
packages (``src``, the local ``gui_support`` helper), standard-library modules,
and import-provider aliases (for example ``sklearn`` -> ``scikit-learn``) are
resolved before comparison.
"""

import ast
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

CLEAR_REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = CLEAR_REPO_ROOT / "requirements.txt"
SCANNED_DIRS = ("src", "scripts", "examples")

# Modules that ship inside this repository rather than being installed.
FIRST_PARTY = {"src", "gui_support"}

# Import name -> distribution name on PyPI, where the two differ.
IMPORT_TO_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "mpl_toolkits": "matplotlib",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
}


def _imported_top_level_modules():
    """Top-level module names imported by the scanned source trees."""
    modules = set()
    for directory in SCANNED_DIRS:
        for path in (CLEAR_REPO_ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        modules.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    # Relative imports (level > 0) are always first-party.
                    if node.level == 0 and node.module:
                        modules.add(node.module.split(".")[0])
    return modules


def _third_party_distributions():
    """Distribution names the scanned code depends on."""
    distributions = set()
    for module in _imported_top_level_modules():
        if module in FIRST_PARTY:
            continue
        if module in sys.stdlib_module_names:
            continue
        distributions.add(IMPORT_TO_DISTRIBUTION.get(module, module))
    return sorted(distributions)


def _declared_distributions():
    """Normalized distribution names declared in requirements.txt."""
    declared = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line
        for separator in ("==", ">=", "<=", "~=", ">", "<", "[", ";"):
            name = name.split(separator)[0]
        declared.add(name.strip().lower().replace("_", "-"))
    return declared


REQUIRED_DISTRIBUTIONS = _third_party_distributions()
DECLARED_DISTRIBUTIONS = _declared_distributions()


# **Feature: repo-reorganization, Property 9: Dependency completeness**
# **Validates: Requirements 8.1**
@settings(max_examples=200)
@given(st.sampled_from(REQUIRED_DISTRIBUTIONS))
def test_dependency_completeness(distribution):
    """For any third-party package imported by code under src/, scripts/, or
    examples/, that package appears in the dependency list."""
    normalized = distribution.lower().replace("_", "-")
    assert normalized in DECLARED_DISTRIBUTIONS, (
        f"{distribution} is imported by code under {SCANNED_DIRS} but is "
        f"missing from {REQUIREMENTS.name}; declared: "
        f"{sorted(DECLARED_DISTRIBUTIONS)}"
    )
