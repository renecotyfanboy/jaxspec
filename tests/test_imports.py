"""Guard that every module imports, and that import *order* never matters.

Two distinct failures live here:

* **Import cycles.** ``analysis.results`` imports from ``fit`` at runtime and the fitters
  annotate their return type with ``FitResult``. If any fitter imports ``FitResult``
  eagerly the cycle closes and importing ``jaxspec.analysis.results`` *first* raises
  ``ImportError``. Catching that needs a fresh interpreter per entry point — once another
  test module has imported jaxspec in a working order, ``sys.modules`` hides it.
* **Modules that simply do not import.** A leaf module nothing else imports can break
  (an upstream dropping a column, say) and no test notices, while it is still public and
  rendered in the API docs. ``jaxspec.util.abundance`` shipped broken exactly that way.
  One subprocess importing everything is enough, and is cheap.
"""

import os
import pkgutil
import subprocess
import sys

import pytest

# Entry points a user might reach for *first*. Order matters here, so each gets its own
# interpreter; keep the list to genuine entry points to bound the runtime.
ENTRY_POINTS = [
    "jaxspec",
    "jaxspec.analysis.results",
    "jaxspec.analysis.compare",
    "jaxspec.analysis._plot",
    "jaxspec.analysis._posterior_params",
    "jaxspec.analysis._ppc",
    "jaxspec.fit",
    "jaxspec.fit._fitter",
    "jaxspec.data",
    "jaxspec.model.abc",
]


@pytest.mark.fast
@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_module_imports_first_in_a_fresh_interpreter(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert (
        result.returncode == 0
    ), f"`import {module}` failed as the first jaxspec import:\n{result.stderr}"


@pytest.mark.fast
def test_from_import_of_fitresult_via_compare():
    """The exact line used in docs/examples/background.md."""
    result = subprocess.run(
        [sys.executable, "-c", "from jaxspec.analysis.compare import plot_corner_comparison"],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr


def _all_jaxspec_modules() -> list[str]:
    """Every importable module in the installed package, discovered — never listed.

    A hand-maintained list is exactly what let `jaxspec.util.abundance` rot: it was public,
    documented, and covered by nothing.
    """
    import jaxspec

    return sorted(
        name
        for _, name, _ in pkgutil.walk_packages(jaxspec.__path__, prefix="jaxspec.")
        if "__pycache__" not in name
    )


@pytest.mark.fast
def test_every_module_imports():
    """Import every module in one subprocess.

    Order-independence is covered by the per-entry-point tests above; this only asks
    whether each module imports *at all*.
    """
    modules = _all_jaxspec_modules()
    assert len(modules) > 20, f"module discovery looks broken: {modules}"

    script = (
        "import importlib, sys\n"
        f"mods = {modules!r}\n"
        "failed = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as exc:\n"
        "        failed.append(f'{m}: {type(exc).__name__}: {exc}')\n"
        "print('\\n'.join(failed))\n"
        "sys.exit(1 if failed else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, f"modules failed to import:\n{result.stdout}{result.stderr}"
