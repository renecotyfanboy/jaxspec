import chex
import pytest
import numpyro
from jax import config
from jaxspec.fit import BayesianModel

chex.set_n_cpu_devices(n=4)
config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


@pytest.fixture(scope="session")
def obsconfs():
    from jaxspec.data.util import load_example_foldings

    return list(load_example_foldings().values())


@pytest.fixture(scope="session")
def observations():
    from jaxspec.data.util import load_example_observations

    return list(load_example_observations().values())


@pytest.fixture(scope="session")
def instruments():
    from jaxspec.data.util import load_example_instruments

    return list(load_example_instruments().values())


@pytest.fixture(scope="session")
def obs_model_prior(obsconfs):
    from jaxspec.model.additive import Powerlaw, Blackbodyrad
    from jaxspec.model.multiplicative import Tbabs
    import numpyro.distributions as dist

    model = Tbabs() * (Powerlaw() + Blackbodyrad())
    prior = {
        "powerlaw_1": {"alpha": dist.Uniform(0, 10), "norm": dist.LogUniform(1e-6, 1)},
        "blackbodyrad_1": {"kT": dist.Uniform(0, 10), "norm": dist.LogUniform(1e-1, 1e4)},
        "tbabs_1": {"N_H": dist.Uniform(0, 0.2)},
    }

    return obsconfs, model, prior


@pytest.fixture(scope="session")
def get_individual_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [BayesianModel(model, obsconf).fit(prior, num_samples=5000) for obsconf in obsconfs]


@pytest.fixture(scope="session")
def get_joint_result(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [BayesianModel(model, obsconfs).fit(prior, num_samples=5000)]
