import subprocess
import sys

import pytest

from jaxspec.util.misc import catchtime


@pytest.mark.fast
def test_debug_info_script_runs():
    """`jaxspec-debug-info` is the command users run when their install is broken.

    It must therefore depend on nothing beyond the standard library and the packages it
    reports on. Run it out-of-process so an already-imported module cannot mask a
    missing dependency.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from jaxspec.scripts.debug import debug_info; debug_info()"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "jaxspec" in result.stdout
    assert "not installed" not in result.stdout.split("JAX")[0].split("Packages")[1]


def _dummy_task():
    """Trivial unit of work timed by the catchtime tests below."""
    return 1


def test_catchtime_measures_time_correctly():
    with catchtime(desc="Test Task", print_time=False) as get_time:
        result = _dummy_task()
    assert result == 1
    assert get_time() > 0


def test_catchtime_prints_time_when_print_time_is_true(capfd):
    with catchtime(desc="Test Task", print_time=True) as get_time:
        result = _dummy_task()
    out, err = capfd.readouterr()
    assert "Test Task" in out
    assert "seconds" in out


def test_catchtime_does_not_print_time_when_print_time_is_false(capfd):
    with catchtime(desc="Test Task", print_time=False) as get_time:
        result = _dummy_task()
    out, err = capfd.readouterr()
    assert out == ""


@pytest.mark.fast
def test_data_path_finder_resolves_siblings_when_not_required():
    """`require_*=False` must still *look* for the file, only not raise on a miss.

    The flag gated the lookup itself, so `Observation.from_pha_file` (which passes False
    for all three) always got `(None, None, None)` and produced an all-zero background
    even with the background file sitting next to the PHA.
    """
    import os

    from jaxspec.data.util import data_path_finder

    pha = "docs/examples/data/PN_spectrum_grp20.fits"
    if not os.path.exists(pha):
        pytest.skip("bundled example PHA not available")

    arf, rmf, bkg = data_path_finder(pha, require_arf=False, require_rmf=False, require_bkg=False)

    assert arf is not None and os.path.basename(arf) == "PN.arf"
    assert rmf is not None and os.path.basename(rmf) == "PN.rmf"
    assert bkg is not None and "background" in os.path.basename(bkg).lower()


@pytest.mark.fast
def test_from_pha_file_picks_up_the_background():
    import os

    import numpy as np

    from jaxspec.data import Observation

    pha = "docs/examples/data/PN_spectrum_grp20.fits"
    if not os.path.exists(pha):
        pytest.skip("bundled example PHA not available")

    observation = Observation.from_pha_file(pha)

    assert float(np.asarray(observation.folded_background.data).sum()) > 0
