"""PHA channel columns take precedence over scalar header defaults."""

from pathlib import Path

import numpy as np
import pytest

from astropy.io import fits

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataPHA


@pytest.mark.parametrize(
    "use_grouping,use_quality,expected_counts,expected_response",
    [
        (True, True, [4, 8], [[1, 0, 0, 0], [0, 0, 1, 0]]),
        (True, False, [904, 708], [[1, 1, 0, 0], [0, 0, 1, 1]]),
        (False, True, [4, 8], [[1, 0, 0, 0], [0, 0, 1, 0]]),
        (False, False, [4, 900, 8, 700], np.eye(4)),
    ],
)
def test_columns_override_header_defaults_in_counts_and_response(
    tmp_path, use_grouping, use_quality, expected_counts, expected_response
):
    """Stale zero keywords must not erase channel selection or grouping."""
    columns = [
        fits.Column(name="CHANNEL", format="J", array=[0, 1, 2, 3]),
        fits.Column(name="COUNTS", format="J", array=[4, 900, 8, 700]),
    ]
    if use_grouping:
        columns.append(fits.Column(name="GROUPING", format="I", array=[1, -1, 1, -1]))
    if use_quality:
        columns.append(fits.Column(name="QUALITY", format="I", array=[0, 1, 0, 1]))
    spectrum = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
    spectrum.header.update(EXPOSURE=10, GROUPING=0, QUALITY=0, BACKSCAL=1, AREASCAL=1)
    path = tmp_path / "flags.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)

    observation = Observation.from_ogip_container(DataPHA.from_file(path))
    edges = np.arange(1.0, 6.0)
    instrument = Instrument.from_matrix(
        np.eye(4), np.ones(4), edges[:-1], edges[1:], edges[:-1], edges[1:]
    )
    configuration = ObsConfiguration.from_instrument(instrument, observation)
    np.testing.assert_array_equal(configuration.folded_counts, expected_counts)
    np.testing.assert_array_equal(
        configuration.transfer_matrix.data.todense(), 10 * np.asarray(expected_response)
    )


@pytest.mark.parametrize("missing", ["GROUPING", "QUALITY"])
def test_missing_column_and_header_default_remains_an_error(tmp_path, missing):
    """Column precedence does not invent metadata when neither form is supplied."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[0, 1]),
            fits.Column(name="COUNTS", format="J", array=[4, 9]),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10, GROUPING=0, QUALITY=0, BACKSCAL=1, AREASCAL=1)
    del spectrum.header[missing]
    path = tmp_path / "missing-flags.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    with pytest.raises(ValueError, match="(?i)" + missing):
        DataPHA.from_file(path)


def test_bundled_xmm_spectrum_preserves_bad_channel_flags():
    """The bundled SAS product retains QUALITY=0 beside its actual flag column."""
    path = Path(__file__).resolve().parents[1] / "docs/examples/data/PN_spectrum_grp20.fits"
    with fits.open(path) as hdus:
        assert hdus["SPECTRUM"].header["QUALITY"] == 0
        quality = hdus["SPECTRUM"].data["QUALITY"].copy()
        counts = hdus["SPECTRUM"].data["COUNTS"].copy()
    pha = DataPHA.from_file(path)
    assert np.count_nonzero(quality) == 1116
    np.testing.assert_array_equal(pha.quality, quality)
    np.testing.assert_array_equal(pha.counts, counts)
