import os
import sys

import chex
import jax
import pytest

from jaxspec.data.util import fakeit_for_multiple_parameters

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
        "powerlaw_1": {
            "alpha": rng.uniform(1, 3, size=num_params),
            "norm": rng.exponential(10 ** (-0.5), size=num_params),
        },
        "blackbodyrad_1": {
            "kT": rng.uniform(0.1, 3.0, size=num_params),
            "norm": rng.exponential(10 ** (-3), size=num_params),
        },
    }

    return parameters


@pytest.fixture
def model():
    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    return Tbabs() * (Powerlaw() + Blackbodyrad())


@pytest.fixture
def sharded_parameters(parameters):
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding

    sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))

    return jax.device_put(parameters, sharding)


def test_fakeits_apply_stat(obsconfs, model, parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(obsconf, model, parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(obsconf, model, parameters, apply_stat=True)
    chex.assert_type(spectra, int)


def test_fakeits_parallel(obsconfs, model, sharded_parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(obsconf, model, sharded_parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(obsconf, model, sharded_parameters, apply_stat=True)
    chex.assert_type(spectra, int)


def test_fakeits_multiple_observation(obsconfs, model, parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(obsconf, model, parameters, apply_stat=False)
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(obsconf, model, parameters, apply_stat=True)
    chex.assert_type(spectra, int)
