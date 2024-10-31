import chex

chex.set_n_cpu_devices(n=4)

import numpyro  # noqa:E402
import pytest  # noqa:E402

from jax import config  # noqa:E402
from jaxspec.fit import MCMCFitter  # noqa:E402

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
    from jaxspec.data.util import load_example_obsconf

    return list(load_example_obsconf("NGC7793_ULX4_ALL").values())


@pytest.fixture(scope="session")
def observations():
    from jaxspec.data.util import load_example_pha

    return list(load_example_pha("NGC7793_ULX4_ALL").values())


@pytest.fixture(scope="session")
def instruments():
    from jaxspec.data.util import load_example_instruments

    return list(load_example_instruments("NGC7793_ULX4_ALL").values())


@pytest.fixture(scope="session")
def obs_model_prior(obsconfs):
    import numpyro.distributions as dist

    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    model = Tbabs() * (Powerlaw() + Blackbodyrad())
    prior = {
        "powerlaw_1_alpha": dist.Uniform(0, 5),
        "powerlaw_1_norm": dist.LogUniform(1e-5, 1e-2),
        "blackbodyrad_1_kT": dist.Uniform(0, 5),
        "blackbodyrad_1_norm": dist.LogUniform(1e-2, 1e2),
        "tbabs_1_N_H": dist.Uniform(0, 1),
    }

    return obsconfs, model, prior


@pytest.fixture(scope="session")
def get_individual_mcmc_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [MCMCFitter(model, prior, obsconf).fit(num_samples=5000) for obsconf in obsconfs]


@pytest.fixture(scope="session")
def get_joint_mcmc_result(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [MCMCFitter(model, prior, obsconfs).fit(num_samples=5000)]


@pytest.fixture(scope="session")
def get_result_list(get_individual_mcmc_results, get_joint_mcmc_result):
    result_list = []
    result_list += get_individual_mcmc_results
    result_list += get_joint_mcmc_result

    name_list = []
    name_list += ["PN_mcmc", "MOS1_mcmc", "MOS2_mcmc"]
    name_list += ["Joint_mcmc"]

    return name_list, result_list
