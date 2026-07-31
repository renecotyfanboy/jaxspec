import chex
import jax
import pytest

from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters


@pytest.fixture
def multidimensional_parameters():
    from numpy.random import default_rng

    rng = default_rng(42)

    num_params = (8, 8, 8)

    parameters = {
        "tbabs_1.nh": rng.uniform(0.1, 0.4, size=num_params),
        "powerlaw_1.alpha": rng.uniform(1, 3, size=num_params),
        "powerlaw_1.norm": rng.exponential(10 ** (-0.5), size=num_params),
        "blackbodyrad_1.kT": rng.uniform(0.1, 3.0, size=num_params),
        "blackbodyrad_1.norm": rng.exponential(10 ** (-3), size=num_params),
    }

    return parameters


@pytest.fixture
def unidimensional_parameters():
    from numpy.random import default_rng

    rng = default_rng(42)

    num_params = 16

    parameters = {
        "tbabs_1.nh": rng.uniform(0.1, 0.4, size=num_params),
        "powerlaw_1.alpha": rng.uniform(1, 3, size=num_params),
        "powerlaw_1.norm": rng.exponential(10 ** (-0.5), size=num_params),
        "blackbodyrad_1.kT": rng.uniform(0.1, 3.0, size=num_params),
        "blackbodyrad_1.norm": rng.exponential(10 ** (-3), size=num_params),
    }

    return parameters


@pytest.fixture
def model():
    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    return Tbabs() * (Powerlaw() + Blackbodyrad())


@pytest.fixture
def sharded_parameters(unidimensional_parameters):
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh((4,), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))

    return jax.device_put(unidimensional_parameters, sharding)


def test_fakeits_apply_stat(obsconfs, model, multidimensional_parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, multidimensional_parameters, apply_stat=False
    )
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(
        obsconf, model, multidimensional_parameters, apply_stat=True
    )
    chex.assert_type(spectra, int)


def test_fakeits_parallel(obsconfs, model, sharded_parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(obsconf, model, sharded_parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(obsconf, model, sharded_parameters, apply_stat=True)
    chex.assert_type(spectra, int)


def test_fakeits_sparsify(obsconfs, model, unidimensional_parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, unidimensional_parameters, apply_stat=False, sparsify_matrix=False
    )
    chex.assert_type(spectra, float)

    # JAX explicit sharding currently does not infer an output sharding for the
    # sparse scatter path used by BCOO matmul, so sparse coverage is kept on
    # unsharded parameters while sharding coverage stays in test_fakeits_parallel.
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, unidimensional_parameters, apply_stat=False, sparsify_matrix=True
    )
    chex.assert_type(spectra, float)


def test_mock_obsconf(instruments, model, multidimensional_parameters):
    for instrument in instruments:
        obsconf = ObsConfiguration.mock_from_instrument(instrument, exposure=1e5)
        fakeit_for_multiple_parameters(obsconf, model, multidimensional_parameters)


def test_fakeits_missing_parameter_raises(obsconfs, model, multidimensional_parameters):
    """A parameters dict missing a model parameter must raise a clear,
    parameter-centric error rather than silently using a default (old behavior)
    or the misleading prior-dict KeyError raised inside the JIT trace."""
    obsconf = obsconfs[0]
    incomplete = {
        k: v for k, v in multidimensional_parameters.items() if k != "blackbodyrad_1.norm"
    }
    with pytest.raises(ValueError, match="blackbodyrad_1"):
        fakeit_for_multiple_parameters(obsconf, model, incomplete)


def test_fakeit_noise_is_independent_across_observations():
    """Each observation must get its own PRNG stream.

    ``handlers.seed`` splits its key per sample site in trace order, so entering it
    inside the per-observation loop restarted from the same key every time and gave every
    observation *identical* Poisson noise — perfectly correlated counts across
    instruments in a joint simulation.
    """
    import jax.numpy as jnp
    import numpy as np

    from jaxspec.data.util import load_example_obsconf
    from jaxspec.model.additive import Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    obsconf = next(iter(load_example_obsconf("NGC7793_ULX4_ALL").values()))
    model = Tbabs() * Powerlaw()
    parameters = {
        "tbabs_1.nh": jnp.array([0.2]),
        "powerlaw_1.alpha": jnp.array([1.7]),
        "powerlaw_1.norm": jnp.array([1e-2]),
    }

    # The same observation twice: identical expected counts, so any correlation left in
    # the residuals is pure PRNG reuse.
    noisy = fakeit_for_multiple_parameters(
        [obsconf, obsconf], model, parameters, apply_stat=True, rng_key=0
    )
    expected = np.asarray(
        fakeit_for_multiple_parameters([obsconf], model, parameters, apply_stat=False)
    ).ravel()

    a, b = np.asarray(noisy[0]).ravel(), np.asarray(noisy[1]).ravel()
    assert not np.array_equal(a, b)

    scale = np.sqrt(np.maximum(expected, 1e-9))
    correlation = np.corrcoef((a - expected) / scale, (b - expected) / scale)[0, 1]
    assert abs(correlation) < 0.2, f"noise still correlated across observations: {correlation}"
