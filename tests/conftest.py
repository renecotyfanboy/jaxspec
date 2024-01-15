import chex
import pytest
import numpyro
from jax import config

chex.set_n_cpu_devices(n=4)
config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


@pytest.fixture
def folding_models():
    from jaxspec.data.util import load_example_foldings

    return list(load_example_foldings().values())


@pytest.fixture
def obs_model_prior(folding_models):
    from jaxspec.model.additive import Powerlaw, Cutoffpl
    from jaxspec.model.multiplicative import Tbabs
    import numpyro.distributions as dist

    model = Tbabs() * (Powerlaw() + Cutoffpl())
    prior = {
        "powerlaw_1": {"alpha": dist.Uniform(0, 5), "norm": dist.Exponential(1e4)},
        "tbabs_1": {"N_H": dist.Uniform(0, 5)},
        "cutoffpl_1": {"alpha": dist.Uniform(0, 5), "beta": dist.Uniform(0, 10), "norm": dist.Exponential(1e4)},
    }

    return folding_models, model, prior
