import pytest
import jax
from jaxspec.data.util import fakeit_for_multiple_parameters
import os
import sys
import chex

chex.set_n_cpu_devices(n=4)

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


@pytest.fixture
def parameters():
    from numpy.random import default_rng

    rng = default_rng(42)

    num_params = 1000

    parameters = {
        "tbabs_1": {"N_H": rng.uniform(0.1, 0.4, size=num_params)},
        "powerlaw_1": {"alpha": rng.uniform(1, 3, size=num_params), "norm": rng.exponential(10 ** (-0.5), size=num_params)},
        "blackbodyrad_1": {"kT": rng.uniform(0.1, 3.0, size=num_params), "norm": rng.exponential(10 ** (-3), size=num_params)},
    }

    return parameters


@pytest.fixture
def model():
    from jaxspec.model.additive import Powerlaw, Blackbodyrad
    from jaxspec.model.multiplicative import Tbabs

    return Tbabs() * (Powerlaw() + Blackbodyrad())


@pytest.fixture
def observations():
    from jaxspec.data.util import example_observations

    return example_observations["PN"]


@pytest.fixture
def sharded_parameters(parameters):
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding

    sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))

    return jax.device_put(parameters, sharding)


def test_fakeits_apply_stat(observations, model, parameters):
    spectra = fakeit_for_multiple_parameters(observations, model, parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(observations, model, parameters, apply_stat=True)
    chex.assert_type(spectra, int)


def test_fakeits_parallel(observations, model, sharded_parameters):
    spectra = fakeit_for_multiple_parameters(observations, model, sharded_parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(observations, model, sharded_parameters, apply_stat=True)
    chex.assert_type(spectra, int)
