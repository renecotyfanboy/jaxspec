"""Detector-background bands need raw bounds, including quality and grouping gaps."""

import numpy as np
import sparse

from jaxspec.data import Instrument, ObsConfiguration, Observation


def test_raw_energy_coordinates_follow_pha_labels_and_keep_every_grouping_column():
    """Offset labels, a rejected channel and an excluded group must retain alignment."""
    instrument = Instrument.from_matrix(
        sparse.eye(6),
        np.ones(6),
        np.arange(1.0, 7.0),
        np.arange(2.0, 8.0),
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        channel=[10, 11, 13, 20, 21, 30],
    )
    # The first three selected channels share one group but channel 13 is bad.
    # The last group lies above the requested band, yet its raw column remains.
    observation = Observation.from_matrix(
        [5, 99, 7, 9],
        [[1, 1, 1, 0], [0, 0, 0, 1]],
        [11, 13, 20, 30],
        [0, 1, 0, 0],
        10.0,
    )
    configured = ObsConfiguration.from_instrument(
        instrument, observation, low_energy=2, high_energy=5
    )
    np.testing.assert_array_equal(configured.channel, [11, 13, 20, 30])
    np.testing.assert_array_equal(configured.e_min_channel, [2, 3, 4, 6])
    np.testing.assert_array_equal(configured.e_max_channel, [3, 4, 5, 7])
    assert configured.e_min_channel.dims == ("instrument_channel",)
    assert configured.e_max_channel.attrs["units"] == "keV"
    np.testing.assert_array_equal(configured.grouping.data.todense(), [[1, 0, 1, 0]])
    np.testing.assert_array_equal(configured.folded_counts, [12])
    np.testing.assert_array_equal(configured.out_energies, [[2], [5]])
    # A detector component supported only in the 3-4 keV quality gap contributes
    # zero. Integrating merely across the folded 2-5 keV bounds would be wrong.
    overlap = np.maximum(
        0,
        np.minimum(configured.e_max_channel.data, 4) - np.maximum(configured.e_min_channel.data, 3),
    )
    np.testing.assert_array_equal(configured.grouping.data @ overlap, [0])
