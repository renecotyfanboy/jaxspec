"""OGIP AREASCAL acts on detector channels before their counts are grouped."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
import sparse

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataPHA
from jaxspec.fit import ForwardModel
from jaxspec.model.additive import Powerlaw
from jaxspec.model.instrument import PileupModel


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


def test_scaled_dimensionless_metadata_are_converted():
    pha = DataPHA([0, 1], [2, 3], 10, areascal=50 * u.percent, backscal=25 * u.percent)
    np.testing.assert_array_equal(pha.areascal, [0.5, 0.5])
    np.testing.assert_array_equal(pha.backscal, [0.25, 0.25])


@pytest.mark.parametrize("name", ["areascal", "backscal"])
def test_dimensional_extraction_scales_are_rejected(name):
    with pytest.raises(ValueError, match="must have units equivalent"):
        DataPHA([0, 1], [2, 3], 10, **{name: 2 * u.cm})


@pytest.mark.parametrize("sparse_matrix", [False, True])
def test_pileup_applies_detector_areascal_after_redistribution(sparse_matrix):
    """Component-based folding must honor the same detector scaling as linear folding."""
    observation, _, _ = scaled_observation()
    pileup = PileupModel(frac_expo=1.0, frame_time=3.2)
    forward = ForwardModel(
        Powerlaw(), observation, instrument_model={"data": pileup}, sparsify_matrix=sparse_matrix
    )
    cache = forward.obs_caches["data"]
    flux = jnp.full(4, 1e-4)
    actual = pileup.fold(flux, cache, eval_energies=observation.in_energies)
    # Keep all nonlinear photon-space pileup operations identical, then select
    # each raw detector channel independently to construct the expected sum.
    ungrouped_cache = dict(cache, grouping=jnp.eye(4), areascal=jnp.ones(4))
    ungrouped = pileup.fold(flux, ungrouped_cache, eval_energies=observation.in_energies)
    expected = cache["grouping"] @ (cache["areascal"] * ungrouped)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
