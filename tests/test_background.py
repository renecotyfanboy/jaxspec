import chex
import pytest
import numpyro
from jax.config import config

chex.set_n_cpu_devices(n=4)
config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


@pytest.fixture
def obs_model_prior():
    from jaxspec.model.additive import Powerlaw, Cutoffpl
    from jaxspec.model.multiplicative import Tbabs
    from jaxspec.data.util import load_example_observations
    import numpyro.distributions as dist

    obs_list = list(load_example_observations().values())
    model = Tbabs() * (Powerlaw() + Cutoffpl())
    prior = {
        "powerlaw_1": {"alpha": dist.Uniform(0, 5), "norm": dist.Exponential(1e4)},
        "tbabs_1": {"N_H": dist.Uniform(0, 5)},
        "cutoffpl_1": {"alpha": dist.Uniform(0, 5), "beta": dist.Uniform(0, 10), "norm": dist.Exponential(1e4)},
    }

    return obs_list, model, prior


def test_gp_bkg(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import GaussianProcessBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20)
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()


def test_subtract_bkg(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import SubtractedBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackground()
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()


def test_subtract_bkg_with_error(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import SubtractedBackgroundWithError

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackgroundWithError()
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()
