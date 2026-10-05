"""Reject nonphysical response inputs while preserving legitimate zero sensitivity."""

import numpy as np
import pytest
import sparse

from jaxspec.data import Instrument
from jaxspec.data.ogip import DataARF


def make_instrument(**changes):
    """A two-bin array response isolates one malformed physical input at a time."""
    values = dict(
        redistribution_matrix=sparse.eye(2),
        spectral_response=[1, 2],
        e_min_unfolded=[1, 2],
        e_max_unfolded=[2, 3],
        e_min_channel=[1, 2],
        e_max_channel=[2, 3],
    )
    values.update(changes)
    return Instrument.from_matrix(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"e_min_unfolded": [np.nan, 2]},
        {"e_max_unfolded": [2, np.inf]},
        {"e_min_unfolded": [-1, 2]},
        {"e_max_unfolded": [1, 3]},
        {"e_max_unfolded": [2.1, 3]},
        {"e_min_unfolded": [2, 1]},
        {"e_min_unfolded": [1]},
        {"e_min_unfolded": [[1, 2]]},
        {"e_max_channel": [0, 3]},
    ],
)
def test_invalid_energy_bins_are_rejected_before_folding(change):
    with pytest.raises(ValueError, match="energies"):
        make_instrument(**change)


@pytest.mark.parametrize("area", [[-1, 2], [np.nan, 2], [1, np.inf], [1]])
def test_effective_area_must_be_finite_nonnegative_and_match_photon_grid(area):
    with pytest.raises(ValueError, match="Effective area"):
        make_instrument(spectral_response=area)


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_bad_response_weights_are_not_discarded(bad, use_sparse):
    response = np.array([[bad, 0], [0, 1]])
    if use_sparse:
        response = sparse.COO(response)
    with pytest.raises(ValueError, match="Redistribution matrix"):
        make_instrument(redistribution_matrix=response)


def test_response_shape_must_match_both_energy_axes():
    with pytest.raises(ValueError, match=r"Redistribution matrix.*shape"):
        make_instrument(redistribution_matrix=np.ones((2, 3)))


def test_zero_area_and_unnormalized_nonnegative_response_remain_valid():
    instrument = make_instrument(
        redistribution_matrix=np.array([[2.0, 0], [3.0, 0]]), spectral_response=[0, 100]
    )
    np.testing.assert_array_equal(instrument.area, [0, 100])
    np.testing.assert_array_equal(instrument.redistribution.data.todense(), [[2, 0], [3, 0]])


def test_arf_validation_also_applies_without_instrument_construction():
    with pytest.raises(ValueError, match="ARF energies"):
        DataARF([1, 1.5], [2, 3], [1, 2])
    with pytest.raises(ValueError, match="ARF SPECRESP"):
        DataARF([1, 2], [2, 3], [1, -2])


def test_storage_roundoff_overlap_preserves_every_original_edge():
    low = np.array([1, np.nextafter(np.float32(2), np.float32(0))], dtype=np.float32)
    high = np.array([2, 3], dtype=np.float32)
    area = DataARF(low, high, [1, 2])
    np.testing.assert_array_equal(area.energ_lo, low.astype(np.float64))
    np.testing.assert_array_equal(area.energ_hi, high.astype(np.float64))
    # The same decimal values stored explicitly in float64 are a real overlap,
    # not an effect of that input's native precision.
    with pytest.raises(ValueError, match="storage roundoff"):
        DataARF(low.astype(np.float64), high.astype(np.float64), [1, 2])


def test_storage_tolerance_cannot_hide_overlap_in_very_narrow_bins():
    step = np.spacing(np.float32(1))
    low = np.array([1, 1 + 7 * step], dtype=np.float32)
    high = np.array([1 + 8 * step, 1 + 15 * step], dtype=np.float32)
    with pytest.raises(ValueError, match="storage roundoff"):
        DataARF(low, high, [1, 2])


def test_detector_bounds_follow_channel_labels_without_reordering_by_energy():
    instrument = make_instrument(e_min_channel=[2, 1], e_max_channel=[3, 2])
    np.testing.assert_array_equal(instrument.e_min_channel, [2, 1])
    np.testing.assert_array_equal(instrument.redistribution.data.todense(), np.eye(2))
