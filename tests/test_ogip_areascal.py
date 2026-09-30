"""OGIP AREASCAL acts on detector channels before their counts are grouped."""

import numpy as np
import sparse

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataPHA


def scaled_observation():
    """A non-diagonal response distinguishes detector scaling from photon area."""
    edges = np.arange(1.0, 6.0)
    redistribution = np.array(
        [[0.8, 0.1, 0.0, 0.0], [0.2, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.2], [0.0, 0.0, 0.1, 0.8]]
    )
    instrument = Instrument.from_matrix(
        sparse.COO(redistribution),
        [20, 30, 40, 50],
        edges[:-1],
        edges[1:],
        edges[:-1],
        edges[1:],
    )
    pha = DataPHA(
        np.arange(4),
        [11, 12, 13, 14],
        10.0,
        grouping=[1, -1, 1, -1],
        areascal=[0.5, 2.0, 0.8, 1.2],
    )
    observation = Observation.from_ogip_container(pha)
    configured = ObsConfiguration.from_instrument(instrument, observation)
    return configured, redistribution, pha


def test_source_counts_use_detector_areascal_before_grouping():
    observation, redistribution, pha = scaled_observation()
    expected = pha.grouping @ (
        pha.areascal[:, None] * redistribution * np.array([20, 30, 40, 50]) * 10
    )
    np.testing.assert_allclose(observation.transfer_matrix.data.todense(), expected, rtol=1e-14)
    np.testing.assert_array_equal(observation.folded_counts, [23, 27])
    np.testing.assert_array_equal(observation.areascal, pha.areascal)
