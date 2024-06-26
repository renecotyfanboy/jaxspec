import chex

chex.set_n_cpu_devices(n=4)

import numpyro  # noqa:E402
import pytest  # noqa:E402

from jax import config  # noqa:E402
from jaxspec.fit import BayesianFitter, MinimizationFitter  # noqa:E402

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

good_init_params = {
    "blackbodyrad_1": {"kT": 0.7504268020515429, "norm": 0.19813098808509122},
    "powerlaw_1": {"alpha": 2.0688055546595323, "norm": 0.00026474320883338153},
    "tbabs_1": {"N_H": 0.10091221590755987},
}


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
    import numpyro.distributions as dist

    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    model = Tbabs() * (Powerlaw() + Blackbodyrad())
    prior = {
        "powerlaw_1": {"alpha": dist.Uniform(0, 10), "norm": dist.LogUniform(1e-6, 1)},
        "blackbodyrad_1": {"kT": dist.Uniform(0, 10), "norm": dist.LogUniform(1e-1, 1e4)},
        "tbabs_1": {"N_H": dist.Uniform(0, 0.2)},
    }

    return obsconfs, model, prior


@pytest.fixture(scope="session")
def get_individual_mcmc_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [BayesianFitter(model, obsconf).fit(prior, num_samples=5000) for obsconf in obsconfs]


@pytest.fixture(scope="session")
def get_joint_mcmc_result(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [BayesianFitter(model, obsconfs).fit(prior, num_samples=5000)]


@pytest.fixture(scope="session")
def get_individual_fit_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [
        MinimizationFitter(model, obsconf).fit(
            prior, num_samples=20_000, init_params=good_init_params
        )
        for obsconf in obsconfs
    ]


@pytest.fixture(scope="session")
def get_joint_fit_result(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [
        MinimizationFitter(model, obsconfs).fit(
            prior, num_samples=20_000, init_params=good_init_params
        )
    ]


@pytest.fixture(scope="session")
def get_result_list(
    get_individual_mcmc_results,
    get_joint_mcmc_result,
    get_individual_fit_results,
    get_joint_fit_result,
):
    result_list = []
    result_list += get_individual_mcmc_results
    result_list += get_joint_mcmc_result
    result_list += get_individual_fit_results
    result_list += get_joint_fit_result

    name_list = []
    name_list += ["PN_mcmc", "MOS1_mcmc", "MOS2_mcmc"]
    name_list += ["Joint_mcmc"]
    name_list += ["PN_fit", "MOS1_fit", "MOS2_fit"]
    name_list += ["Joint_fit"]

    return name_list, result_list
