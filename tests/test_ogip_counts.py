"""Event counts and OGIP grouping must retain the supplied Poisson experiment."""

import numpy as np
import pytest
import sparse

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
