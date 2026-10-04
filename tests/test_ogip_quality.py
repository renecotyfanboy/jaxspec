"""Quality masks must select the same physical channels in data and response."""

import numpy as np
import pytest
import sparse

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataPHA


def make_inputs(*, quality=(0, 1, 0, 1)):
    """Construct a two-group detector with contaminated rejected channels."""
    edges = np.arange(1.0, 6.0)
    instrument = Instrument.from_matrix(
        sparse.eye(4), np.ones(4), edges[:-1], edges[1:], edges[:-1], edges[1:]
    )
    pha = DataPHA(
        np.arange(4),
        [4, 900, 8, 700],
        10.0,
        grouping=[1, -1, 1, -1],
        quality=quality,
    )
    observation = Observation.from_matrix(
        pha.counts,
        pha.grouping,
        pha.channel,
        pha.quality,
        pha.exposure,
        background=[1, 800, 2, 600],
        backratio=[0.1, 9.0, 0.3, 7.0],
    )
    return instrument, observation


def test_rejected_raw_channels_do_not_remain_in_grouped_counts():
    instrument, observation = make_inputs()
    configured = ObsConfiguration.from_instrument(instrument, observation)
    np.testing.assert_array_equal(configured.folded_counts, [4, 8])
    np.testing.assert_array_equal(configured.folded_background, [1, 2])
    np.testing.assert_allclose(configured.folded_backratio, [0.1, 0.3])
    np.testing.assert_array_equal(configured.out_energies, [[1, 3], [2, 4]])
    np.testing.assert_array_equal(
        configured.transfer_matrix.data.todense(), [[10, 0, 0, 0], [0, 0, 10, 0]]
    )


def test_wholly_bad_group_is_removed_without_affecting_other_counts():
    instrument, observation = make_inputs(quality=[1, 1, 0, 1])
    configured = ObsConfiguration.from_instrument(instrument, observation)
    np.testing.assert_array_equal(configured.folded_counts, [8])
    np.testing.assert_array_equal(configured.folded_background, [2])


def test_exact_band_edges_are_included():
    instrument, observation = make_inputs()
    configured = ObsConfiguration.from_instrument(
        instrument, observation, low_energy=1.0, high_energy=2.0
    )
    np.testing.assert_array_equal(configured.folded_counts, [4])


@pytest.mark.parametrize("bounds", [(2, 1), (-1, 4), (np.nan, 4)])
def test_invalid_band_is_rejected(bounds):
    instrument, observation = make_inputs()
    with pytest.raises(ValueError, match="Energy bounds"):
        ObsConfiguration.from_instrument(
            instrument, observation, low_energy=bounds[0], high_energy=bounds[1]
        )


def test_empty_selected_band_has_an_actionable_error():
    instrument, observation = make_inputs()
    with pytest.raises(ValueError, match="No usable grouped channels"):
        ObsConfiguration.from_instrument(instrument, observation, low_energy=20, high_energy=30)


def test_empty_response_is_rejected_before_constructing_configuration():
    """An all-zero RMF cannot produce a meaningful source-count likelihood."""
    instrument, observation = make_inputs()
    instrument["redistribution"] = instrument.redistribution.copy(
        data=sparse.zeros(instrument.redistribution.shape)
    )
    with pytest.raises(ValueError, match="response has no energy bin"):
        ObsConfiguration.from_instrument(instrument, observation)
