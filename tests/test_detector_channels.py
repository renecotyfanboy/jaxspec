"""Detector labels survive FITS quantities without rounding or integer overflow."""

import astropy.units as u
import numpy as np
import pytest
import sparse

from astropy.io import fits

from jaxspec.data import Instrument, Observation
from jaxspec.data.ogip import DataPHA


def construct_all(channels):
    """Apply one label vector to each public PHA, observation and response boundary."""
    constructors = (
        lambda: DataPHA(channels, [2, 3], 1.0).channel,
        lambda: Observation.from_matrix([2, 3], sparse.eye(2), channels, [0, 0], 1.0).channel.data,
        lambda: (
            Instrument.from_matrix(
                sparse.eye(2), [1, 1], [1, 2], [2, 3], [1, 2], [2, 3], channel=channels
            ).channel.data
        ),
    )
    return constructors


def test_integral_quantities_preserve_channel_identity_everywhere():
    """Astropy's Quantity float dtype must not exclude integer-valued channel labels."""
    channels = np.array([10, 12]) * u.def_unit("chan")
    for construct in construct_all(channels):
        labels = construct()
        np.testing.assert_array_equal(labels, [10, 12])
        assert labels.dtype == np.int64


@pytest.mark.parametrize(
    "channels",
    [
        [10, 12.5],
        [10, np.inf],
        [10, np.nan],
        [12, 10],
        [10, 10],
        [10],
        [True, False],
        np.array([10, 2**63], dtype=np.uint64),
        [10.0, float(2**63)],
    ],
)
def test_invalid_labels_fail_before_any_integer_cast(channels):
    """All construction routes reject labels that would otherwise be rounded or wrapped."""
    for construct in construct_all(channels):
        with pytest.raises(ValueError, match="channel identifiers"):
            construct()


def test_signed_extreme_labels_do_not_overflow_an_ordering_difference():
    """Ordering compares adjacent values without subtracting opposite int64 extremes."""
    values = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max])
    for construct in construct_all(values):
        np.testing.assert_array_equal(construct(), values)


def test_pha_fits_integer_channel_with_unit_loads(tmp_path):
    """The actual QTable file-reading path may attach a chan unit and promote dtype."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", unit="chan", array=[10, 12]),
            fits.Column(name="COUNTS", format="J", array=[2, 3]),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=1.0, GROUPING=0, QUALITY=0, BACKSCAL=1.0, AREASCAL=1.0)
    path = tmp_path / "channels.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    np.testing.assert_array_equal(DataPHA.from_file(path).channel, [10, 12])
