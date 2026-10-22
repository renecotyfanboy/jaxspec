"""Event counts and OGIP grouping must retain the supplied Poisson experiment."""

import warnings

import astropy.units as u
import numpy as np
import pytest
import sparse

from astropy.io import fits

from jaxspec.data import Observation
from jaxspec.data.ogip import DataPHA


@pytest.mark.parametrize(
    "counts",
    [
        [1.25, 2.0],
        [-1, 2],
        [np.nan, 2],
        [np.inf, 2],
        np.array([2**63, 2], dtype=np.uint64),
        [float(2**63), 2.0],
        [[1, 2]],
        [True, False],
    ],
)
def test_invalid_events_are_rejected_before_casting(counts):
    """Neither the file container nor array API may silently alter observations."""
    with pytest.raises(ValueError):
        DataPHA([0, 1], counts, 10.0)
    with pytest.raises(ValueError):
        Observation.from_matrix(counts, sparse.eye(2), [0, 1], [0, 0], 10.0)


def test_zero_group_flags_keep_individual_channels():
    """Mixed grouped/ungrouped channels retain every raw event exactly once."""
    pha = DataPHA(np.arange(6), [2, 3, 5, 7, 11, 13], 10.0, grouping=[0, 1, -1, 0, 1, -1])
    np.testing.assert_array_equal(pha.grouping @ pha.counts, [2, 8, 7, 24])
    np.testing.assert_array_equal(pha.grouping.sum(axis=0).todense(), np.ones(6))


def test_all_zero_group_flags_mean_no_grouping():
    """A column of zero flags has the same meaning as GROUPING=0 in a header."""
    pha = DataPHA([1, 2, 3], [7, 0, 11], 100.0, grouping=[0, 0, 0])
    np.testing.assert_array_equal(pha.grouping.todense(), np.eye(3))


@pytest.mark.parametrize("grouping", [[-1, 1], [1, 2], [1, -0.5], [1]])
def test_invalid_grouping_cannot_discard_or_reassign_events(grouping):
    with pytest.raises(ValueError, match="GROUPING"):
        DataPHA([0, 1], [2, 3], 10.0, grouping=grouping)


@pytest.mark.parametrize("channel", [[1, 1], [2, 1], [0.0, 1.5], [1]])
def test_channel_identifiers_are_unambiguous(channel):
    with pytest.raises(ValueError, match="channel identifiers"):
        DataPHA(channel, [2, 3], 10.0)


@pytest.mark.parametrize("exposure", [0.0, -1.0, np.nan, np.inf])
def test_pha_exposure_is_positive(exposure):
    with pytest.raises(ValueError, match="exposure"):
        DataPHA([0, 1], [2, 3], exposure)


def test_scalar_metadata_and_integer_valued_float_events_are_preserved():
    """Programmatic counts and scalar OGIP defaults produce a usable observation."""
    pha = DataPHA([0, 1], [2.0, 0.0], 20.0, backscal=0.25, areascal=0.8)
    observation = Observation.from_matrix(
        pha.counts, pha.grouping, pha.channel, pha.quality, pha.exposure
    )
    np.testing.assert_array_equal(observation.counts, [2, 0])
    np.testing.assert_array_equal(observation.folded_counts, [2, 0])
    np.testing.assert_array_equal(observation.backratio, [1.0, 1.0])
    np.testing.assert_array_equal(pha.backscal, [0.25, 0.25])
    np.testing.assert_array_equal(pha.areascal, [0.8, 0.8])


def test_background_events_are_not_rounded():
    with pytest.raises(ValueError, match=r"background counts.*fractional"):
        Observation.from_matrix([2, 3], sparse.eye(2), [0, 1], [0, 0], 10.0, background=[0.1, 1.9])


def test_explicit_count_units_are_converted_without_losing_integer_precision():
    counts = u.Quantity([2, 3], u.kct, dtype=np.int64)
    np.testing.assert_array_equal(DataPHA([0, 1], counts, 10).counts, [2000, 3000])
    large = u.Quantity([2**53 + 1, np.iinfo(np.int64).max], u.ct, dtype=np.int64)
    observation = Observation.from_matrix(large, sparse.eye(2), [0, 1], [0, 0], 10)
    np.testing.assert_array_equal(observation.folded_counts, large.value)
    np.testing.assert_array_equal(observation.folded_background, [0, 0])
    assert observation.folded_background.dtype == np.int64


@pytest.mark.parametrize("unit", [u.ct / u.s, u.keV])
def test_explicit_rates_and_non_count_units_are_rejected(unit):
    with pytest.raises(ValueError, match="units equivalent to counts"):
        DataPHA([0, 1], [2, 3] * unit, 10)


@pytest.mark.parametrize("background", [False, True])
def test_group_sums_cannot_wrap_back_to_positive_counts(background):
    counts = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    with pytest.raises(ValueError, match="signed 64-bit event-count range"):
        Observation.from_matrix(
            [1, 2, 3] if background else counts,
            [[1, 1, 1]],
            [0, 1, 2],
            [0, 0, 0],
            10,
            background=counts if background else None,
        )


def test_large_group_sum_at_int64_limit_is_exact():
    maximum = np.iinfo(np.int64).max
    observation = Observation.from_matrix(
        [maximum - 1, 1], [[1, 1]], [0, 1], [0, 0], 10, background=[maximum - 2, 2]
    )
    np.testing.assert_array_equal(observation.folded_counts, [maximum])
    np.testing.assert_array_equal(observation.folded_background, [maximum])


@pytest.mark.parametrize(
    "column,unit,values", [("COUNTS", "10**3 count", [2, 3]), ("RATE", "count/s", [200, 300])]
)
def test_unit_tagged_fits_counts_and_legacy_rate_columns(tmp_path, column, unit, values):
    """Legacy RATE columns without HDUCLAS3 still convert exposure in seconds."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[0, 1]),
            fits.Column(name=column, format="D", unit=unit, array=values),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10, GROUPING=0, QUALITY=0, BACKSCAL=1, AREASCAL=1)
    path = tmp_path / "counts.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    np.testing.assert_array_equal(DataPHA.from_file(path).counts, [2000, 3000])


@pytest.mark.parametrize("poisson", [None, False, True])
@pytest.mark.parametrize("supplied_errors", [False, True])
def test_supplied_errors_do_not_reject_raw_event_counts(tmp_path, poisson, supplied_errors):
    """Raw counts remain usable when a pipeline stores its own error estimates."""
    columns = [
        fits.Column(name="CHANNEL", format="J", array=[0, 1]),
        fits.Column(name="COUNTS", format="J", array=[4, 9]),
    ]
    if supplied_errors:
        columns.append(fits.Column(name="STAT_ERR", format="D", array=[20, 30]))
    spectrum = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
    spectrum.header.update(EXPOSURE=10, GROUPING=0, QUALITY=0, BACKSCAL=1, AREASCAL=1)
    if poisson is not None:
        spectrum.header["POISSERR"] = poisson
    path = tmp_path / "errors.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pha = DataPHA.from_file(path)
        observation = Observation.from_ogip_container(pha)
    expected_warning = poisson is False or (supplied_errors and poisson is not True)
    assert len(captured) == int(expected_warning)
    if expected_warning:
        assert captured[0].category is UserWarning
        assert "Poisson likelihood ignores these error settings" in str(captured[0].message)
        assert captured[0].filename == __file__
    assert "NONPOISSON" not in pha.flags
    np.testing.assert_array_equal(observation.counts, [4, 9])
    np.testing.assert_array_equal(observation.folded_counts, [4, 9])


@pytest.mark.parametrize("counts,classification", [([4.5, 9], "TOTAL"), ([4, 9], "NET")])
def test_supplied_errors_do_not_bypass_event_count_requirements(tmp_path, counts, classification):
    """The warning does not allow fractional or background-subtracted spectra."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[0, 1]),
            fits.Column(name="COUNTS", format="D", array=counts),
            fits.Column(name="STAT_ERR", format="D", array=[20, 30]),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10, GROUPING=0, QUALITY=0, BACKSCAL=1, AREASCAL=1)
    spectrum.header.update(POISSERR=False, HDUCLAS2=classification)
    path = tmp_path / "errors.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    message = "fractional" if classification == "TOTAL" else "background-subtracted"
    with pytest.warns(UserWarning, match="Poisson likelihood ignores"):
        with pytest.raises(ValueError, match=message):
            Observation.from_ogip_container(DataPHA.from_file(path))
