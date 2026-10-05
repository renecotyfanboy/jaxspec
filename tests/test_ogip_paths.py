"""Named optional OGIP links are resolved independently of missing-file policy."""

import numpy as np
import pytest

from astropy.io import fits

from jaxspec.data import Observation
from jaxspec.data.util import data_path_finder, find_file_or_compressed_in_dir


def write_pha(path, counts=(2, 3), **headers):
    """Create a small event spectrum with complete scalar OGIP defaults."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[10, 12]),
            fits.Column(name="COUNTS", format="J", array=counts),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10.0, GROUPING=0, QUALITY=0, BACKSCAL=1.0, AREASCAL=1.0)
    spectrum.header.update(headers)
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    return path


def test_optional_links_are_found_and_direct_observation_loads_measured_background(tmp_path):
    """An optional background must not disappear merely because no error was requested."""
    source = write_pha(
        tmp_path / "source.pha", ANCRFILE="area.arf", RESPFILE="response.rmf", BACKFILE="back.pha"
    )
    (tmp_path / "area.arf").touch()
    (tmp_path / "response.rmf").touch()
    write_pha(tmp_path / "back.pha", counts=(5, 7))
    found = data_path_finder(source, require_arf=False, require_rmf=False, require_bkg=False)
    assert found == tuple(str(tmp_path / name) for name in ("area.arf", "response.rmf", "back.pha"))
    observation = Observation.from_pha_file(source)
    np.testing.assert_array_equal(observation.background, [5, 7])
    assert observation.attrs["background_file"] == str(tmp_path / "back.pha")


def test_optional_missing_file_and_required_missing_file_have_distinct_policies(tmp_path):
    """The require flag controls an error, rather than suppressing all discovery."""
    path = write_pha(tmp_path / "source.pha", RESPFILE="missing.rmf")
    assert data_path_finder(path, require_rmf=False) == (None, None, None)
    with pytest.raises(FileNotFoundError, match=r"missing\.rmf"):
        data_path_finder(path)


def test_named_gzip_fallback_is_exact_even_when_a_backup_exists(tmp_path):
    """A prefix match cannot select a different response or hide the correct gzip file."""
    (tmp_path / "response.rmf.backup").touch()
    compressed = tmp_path / "response.rmf.gz"
    compressed.touch()
    assert find_file_or_compressed_in_dir("response.rmf", tmp_path, True) == str(compressed)
    compressed.unlink()
    assert find_file_or_compressed_in_dir("response.rmf", tmp_path, False) is None
    with pytest.raises(FileNotFoundError, match=r"response\.rmf"):
        find_file_or_compressed_in_dir("response.rmf", tmp_path, True)


def test_absolute_links_and_none_markers_are_preserved(tmp_path):
    """An explicit absolute calibration path is not reinterpreted relative to the PHA."""
    response = tmp_path / "response.rmf"
    response.touch()
    spectra = tmp_path / "spectra"
    spectra.mkdir()
    path = write_pha(spectra / "source.pha", RESPFILE=str(response), ANCRFILE=" NONE ", BACKFILE="")
    assert data_path_finder(path) == (None, str(response), None)


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    ("name", "tried_names"),
    [
        ("missing.rmf", ("missing.rmf", "missing.rmf.gz")),
        ("missing.rmf.gz", ("missing.rmf.gz",)),
    ],
)
def test_missing_file_error_lists_only_the_paths_tried(tmp_path, absolute, name, tried_names):
    """Missing response errors identify real candidates, including absolute gzip paths."""
    directory = tmp_path / "spectra"
    parent = tmp_path / "calibration" if absolute else directory
    path = parent / name if absolute else name

    with pytest.raises(FileNotFoundError) as error:
        find_file_or_compressed_in_dir(path, directory, True)

    tried = ", ".join(str(parent / candidate) for candidate in tried_names)
    assert str(error.value) == f"Can't find {path} (tried: {tried})."
