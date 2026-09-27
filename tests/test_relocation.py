"""Property-based tests for relocated reusable code (spec: repo-reorganization).

Covers three properties over the modules under ``clear_repo/src/``:

* Property 6 - public signatures of relocated symbols are unchanged.
* Property 4 - reusable modules contain no interactive ``input(`` call.
* Property 5 - reusable modules contain no machine-specific absolute paths.

Signatures are compared by parsing the source with :mod:`ast`, so the original
research scripts do not have to be imported (they pull in GUI backends).
"""

import ast
import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Repository root: two levels up from this file (clear_repo/tests/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "clear_repo" / "src"

# Every Python module under src/ is a Reusable_Module.
SRC_MODULES = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for p in SRC_ROOT.rglob("*.py")
    if "__pycache__" not in p.parts
)

# Relocated module -> original source file it was relocated from.
RELOCATIONS = {
    "clear_repo/src/toolpath/generator.py": "AS4RCSD5.py",
    "clear_repo/src/evaluation/metrics.py": "evaluation_metrics.py",
    "clear_repo/src/deposition/simulation.py": "Simulation.py",
}

# Delegating wrappers: (relocated module, function name) -> (original file,
# original class, original method). The wrapper drops the bound ``self``.
DELEGATIONS = {
    ("clear_repo/src/slicing/__init__.py", "slice_mesh"):
        ("AS4RCSD5.py", "ToolpathGenerator", "slice_mesh"),
    ("clear_repo/src/slicing/__init__.py", "extract_slice"):
        ("AS4RCSD5.py", "ToolpathGenerator", "extract_slice"),
    ("clear_repo/src/reference_frames/transformations.py", "rotate_mesh"):
        ("AS4RCSD5.py", "ToolpathGenerator", "rotate_mesh"),
    ("clear_repo/src/reference_frames/transformations.py", "rotate_points"):
        ("AS4RCSD5.py", "ToolpathGenerator", "rotate_points"),
    ("clear_repo/src/reference_frames/transformations.py", "transform_coordinates"):
        ("AS4RCSD5.py", "ToolpathGenerator", "_transform_coordinates"),
    ("clear_repo/src/reference_frames/transformations.py", "rotate_coordinates"):
        ("AS4RCSD5.py", "ToolpathGenerator", "_rotate_coordinates"),
}


def _signature(node: ast.AST) -> str:
    """Render a function's parameter signature as a comparable string.

    Parameter names, ordering, and default expressions are included; return
    annotations and decorators are not part of the call interface.
    """
    args = node.args
    parts = []

    positional = args.posonlyargs + args.args
    defaults = list(args.defaults)
    first_default = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if i == len(args.posonlyargs) and args.posonlyargs:
            parts.append("/")
        text = arg.arg
        if i >= first_default:
            text += "=" + ast.unparse(defaults[i - first_default])
        parts.append(text)

    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += "=" + ast.unparse(default)
        parts.append(text)

    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)

    return "(" + ", ".join(parts) + ")"


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _symbols(rel_path: str) -> dict:
    """Map symbol name -> signature for a module.

    Module-level functions are keyed by name; classes are keyed by name (mapped
    to their ``__init__`` signature when present) and their methods by
    ``Class.method``.
    """
    table = {}
    tree = _parse(rel_path)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            table[node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            table.setdefault(node.name, "(class)")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    table[f"{node.name}.{item.name}"] = _signature(item)
                    if item.name == "__init__":
                        table[node.name] = _signature(item)
    return table


def _public(name: str) -> bool:
    """A symbol is public when no path component starts with an underscore."""
    return all(not part.startswith("_") for part in name.split("."))


def _relocated_symbol_pairs():
    """Build the list of (label, relocated_signature, original_signature) triples."""
    pairs = []

    for relocated, original in RELOCATIONS.items():
        relocated_symbols = _symbols(relocated)
        original_symbols = _symbols(original)
        for name, signature in relocated_symbols.items():
            if not _public(name) or name not in original_symbols:
                continue
            pairs.append(
                (f"{relocated}::{name}", signature, original_symbols[name])
            )

    for (module, func), (orig_file, orig_class, orig_method) in DELEGATIONS.items():
        wrapper = _symbols(module)[func]
        method = _symbols(orig_file)[f"{orig_class}.{orig_method}"]
        # Drop the bound ``self`` from the original method signature.
        method_without_self = "(" + method[len("(self"):].lstrip(", ")
        pairs.append((f"{module}::{func}", wrapper, method_without_self))

    return pairs


RELOCATED_SYMBOL_PAIRS = _relocated_symbol_pairs()
SYMBOL_LABELS = sorted(label for label, _, _ in RELOCATED_SYMBOL_PAIRS)
SIGNATURES_BY_LABEL = {
    label: (new, old) for label, new, old in RELOCATED_SYMBOL_PAIRS
}

# Machine-specific absolute paths: Unix home directories and Windows user profiles.
MACHINE_PATH_RE = re.compile(
    r"(/home/[^/\s\"']+/|/Users/[^/\s\"']+/|[A-Za-z]:\\+Users\\+)"
)


# **Feature: repo-reorganization, Property 6: Public signatures preserved for relocated symbols**
# **Validates: Requirements 4.1**
@settings(max_examples=200)
@given(st.sampled_from(SYMBOL_LABELS))
def test_public_signatures_preserved_for_relocated_symbols(label):
    """For any public function or class relocated from an original file into
    src/, its parameter signature in the new location equals its signature in
    the original file."""
    relocated_signature, original_signature = SIGNATURES_BY_LABEL[label]
    assert relocated_signature == original_signature, (
        f"Signature drift for {label}: relocated {relocated_signature} "
        f"!= original {original_signature}"
    )


# **Feature: repo-reorganization, Property 4: Reusable modules are non-interactive**
# **Validates: Requirements 2.1**
# deadline=None: parsing the large generator.py exceeds the default per-example
# deadline; parse time is not part of what this property asserts.
@settings(max_examples=200, deadline=None)
@given(st.sampled_from(SRC_MODULES))
def test_reusable_modules_are_non_interactive(module_path):
    """For any Python module under src/, the module source contains no
    interactive input() call."""
    tree = _parse(module_path)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    ]
    assert not calls, (
        f"{module_path} makes interactive input() calls on lines "
        f"{[node.lineno for node in calls]}; reusable modules must be "
        f"non-interactive."
    )


# **Feature: repo-reorganization, Property 5: No machine-specific paths in reusable code**
# **Validates: Requirements 5.1, 5.3**
@settings(max_examples=200)
@given(st.sampled_from(SRC_MODULES))
def test_no_machine_specific_paths_in_reusable_code(module_path):
    """For any Python module under src/, no line contains an absolute
    machine-specific home path."""
    offenders = []
    for lineno, line in enumerate(
        (REPO_ROOT / module_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if MACHINE_PATH_RE.search(line):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        f"{module_path} contains machine-specific absolute paths: {offenders}"
    )
