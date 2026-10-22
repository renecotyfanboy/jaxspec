"""Source/background aperture scaling must preserve the original counting model."""

import numpy as np
import pytest

from jaxspec.data import Observation
from jaxspec.data.ogip import DataPHA


def test_background_is_aligned_by_channel_before_scaling():
    source = DataPHA([5, 7], [20, 30], 10, backscal=[0.1, 0.2], areascal=[0.5, 0.8])
    background = DataPHA([4, 5, 6, 7], [2, 3, 5, 7], 20, backscal=0.4, areascal=2.0)
    observation = Observation.from_ogip_container(source, background)
    np.testing.assert_array_equal(observation.counts, [20, 30])
    np.testing.assert_array_equal(observation.background, [3, 7])
    np.testing.assert_allclose(observation.backratio, [0.03125, 0.1])
    np.testing.assert_allclose(observation.areascal, [0.5, 0.8])
    assert observation.attrs["background_exposure"] == 20


def test_missing_background_channel_is_not_assumed_to_have_zero_counts():
    with pytest.raises(ValueError, match="missing source detector channel"):
        Observation.from_ogip_container(
            DataPHA([5, 7], [20, 30], 10), DataPHA([4, 5, 6], [2, 3, 5], 20)
        )


def test_background_quality_is_respected_in_usable_channels():
    source = DataPHA([5, 7], [20, 30], 10)
    background = DataPHA([5, 7], [3, 700], 20, quality=[0, 2])
    observation = Observation.from_ogip_container(source, background)
    np.testing.assert_array_equal(observation.quality, [0, 2])


def test_varying_background_scaling_refines_groups_instead_of_averaging():
    source = DataPHA(np.arange(4), [5, 6, 7, 8], 10, grouping=[1, -1, -1, -1])
    observation = Observation.from_matrix(
        source.counts,
        source.grouping,
        source.channel,
        source.quality,
        source.exposure,
        background=[1, 2, 3, 4],
        backratio=[1, 1, 2, 2],
    )
    np.testing.assert_array_equal(observation.folded_counts, [11, 15])
    np.testing.assert_array_equal(observation.folded_background, [3, 7])
    np.testing.assert_array_equal(observation.folded_backratio, [1, 2])
    assert observation.attrs["background_scale_group_splits"] == 1
    assert float((observation.folded_background * observation.folded_backratio).sum()) == 17
    np.testing.assert_array_equal(observation.grouping.data.sum(axis=0).todense(), np.ones(4))


@pytest.mark.parametrize("flag", ["NET", "NONPOISSON"])
@pytest.mark.parametrize("which", ["source", "background"])
def test_non_poisson_data_are_not_reconstructed_by_adding_background(flag, which):
    source = DataPHA([0, 1], [20, 30], 10, flags=[flag] if which == "source" else [])
    background = DataPHA([0, 1], [3, 4], 20, flags=[flag] if which == "background" else [])
    with pytest.raises(ValueError, match="original TOTAL source and Poisson BKG"):
        Observation.from_ogip_container(source, background)


@pytest.mark.parametrize("scaling", [0.0, -1.0])
def test_nonpositive_aperture_scaling_is_not_silently_replaced(scaling):
    source = DataPHA([0, 1], [20, 30], 10, backscal=scaling)
    background = DataPHA([0, 1], [3, 4], 20)
    with pytest.raises(ValueError, match="BACKSCAL must be positive"):
        Observation.from_ogip_container(source, background)


@pytest.mark.parametrize("grouping", [[[1, 0], [1, 1]], [[0.5, 0.5]]])
def test_overlapping_or_weighted_counts_are_not_poisson_groups(grouping):
    with pytest.raises(ValueError, match="Grouping must sum disjoint"):
        Observation.from_matrix([2, 3], grouping, [0, 1], [0, 0], 10)
