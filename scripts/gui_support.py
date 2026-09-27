"""Shared helpers for the GUI entry-point scripts.

The GUI applications (`gcode_dev_v2.py`, `repair_editor.py`,
`visualizer_advanced.py`) load their inputs through Tk file dialogs. The entry
points in ``scripts/`` let a user pass those inputs as command-line arguments
instead (Requirements 5.2, 5.3), without changing the GUI code: while a file is
"preselected", the application's open-file dialog returns the given path, so the
application's own loading routine runs exactly as it does interactively.
"""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

# Repository root: the parent of clear_repo/. The GUI applications still live
# there and are imported as-is (the reorganization copies rather than moves).
CLEAR_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLEAR_REPO_ROOT.parent


def add_repo_root_to_path() -> Path:
    """Make the original GUI modules at the repository root importable."""
    for root in (str(CLEAR_REPO_ROOT), str(REPO_ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)
    return REPO_ROOT


def load_repo_module(module_name: str):
    """Import a GUI application module from the repository root by file path.

    Importing by path avoids a name clash between an entry point and the
    application module it wraps (for example ``repair_editor.py``).
    """
    cached = sys.modules.get(f"as4rcsd_app.{module_name}")
    if cached is not None:
        return cached

    source = REPO_ROOT / f"{module_name}.py"
    if not source.is_file():
        raise FileNotFoundError(f"GUI application module not found: {source}")

    add_repo_root_to_path()
    spec = importlib.util.spec_from_file_location(f"as4rcsd_app.{module_name}", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_file(parser, path, label: str) -> str:
    """Exit with a parser error unless ``path`` names an existing file."""
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        parser.error(f"{label} not found: {path}")
    return str(resolved)


@contextmanager
def preselected_file(module, path):
    """Make ``module``'s open-file dialog return ``path`` for one call.

    ``module`` is the GUI module (it exposes ``filedialog``). Restores the
    original dialog function afterwards, so later interactive use is unaffected.
    """
    filedialog = module.filedialog
    original = filedialog.askopenfilename
    filedialog.askopenfilename = lambda *args, **kwargs: path
    try:
        yield
    finally:
        filedialog.askopenfilename = original


def load_with_path(module, loader, path) -> None:
    """Invoke a GUI ``loader`` method as if ``path`` had been chosen in a dialog."""
    if not path:
        return
    with preselected_file(module, path):
        loader()


def run_gui(build_and_run) -> int:
    """Run a GUI callable, reporting a missing display as a clean error."""
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is not available in this Python installation.", file=sys.stderr)
        return 1

    try:
        return build_and_run() or 0
    except tk.TclError as error:
        print(
            f"Could not open a graphical display: {error}\n"
            f"These editors require a desktop session (a set DISPLAY).",
            file=sys.stderr,
        )
        return 1
