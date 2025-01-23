import os
import sys

import chex
import jax
import pytest

from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters

chex.set_n_cpu_devices(n=4)

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


@pytest.fixture
def multidimensional_parameters():
    from numpy.random import default_rng

    rng = default_rng(42)

    num_params = (8, 8, 8)

    parameters = {
        "tbabs_1_nh": rng.uniform(0.1, 0.4, size=num_params),
        "powerlaw_1_alpha": rng.uniform(1, 3, size=num_params),
        "powerlaw_1_norm": rng.exponential(10 ** (-0.5), size=num_params),
        "blackbodyrad_1_kT": rng.uniform(0.1, 3.0, size=num_params),
        "blackbodyrad_1_norm": rng.exponential(10 ** (-3), size=num_params),
    }

    return parameters


@pytest.fixture
def unidimensional_parameters():
    from numpy.random import default_rng

    rng = default_rng(42)

    num_params = 16

    parameters = {
        "tbabs_1_nh": rng.uniform(0.1, 0.4, size=num_params),
        "powerlaw_1_alpha": rng.uniform(1, 3, size=num_params),
        "powerlaw_1_norm": rng.exponential(10 ** (-0.5), size=num_params),
        "blackbodyrad_1_kT": rng.uniform(0.1, 3.0, size=num_params),
        "blackbodyrad_1_norm": rng.exponential(10 ** (-3), size=num_params),
    }

    return parameters


@pytest.fixture
def model():
    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    return Tbabs() * (Powerlaw() + Blackbodyrad())


@pytest.fixture
def sharded_parameters(unidimensional_parameters):
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding

    sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))

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


def test_fakeits_multiple_observation(obsconfs, model, multidimensional_parameters):
    obsconf = obsconfs[0]
    spectra = fakeit_for_multiple_parameters(
        obsconf, model, multidimensional_parameters, apply_stat=False
    )
    chex.assert_type(spectra, float)

    spectra = fakeit_for_multiple_parameters(
        obsconf, model, multidimensional_parameters, apply_stat=True
    )
    chex.assert_type(spectra, int)


def test_mock_obsconf(instruments, model, multidimensional_parameters):
    for instrument in instruments:
        obsconf = ObsConfiguration.mock_from_instrument(instrument, exposure=1e5)
        fakeit_for_multiple_parameters(obsconf, model, multidimensional_parameters)
