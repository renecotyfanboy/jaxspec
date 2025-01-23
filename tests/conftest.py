# ruff: noqa: E402

import chex

chex.set_n_cpu_devices(n=4)

import os

from pathlib import Path

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pooch
import pytest
import yaml

from jax import config
from jaxspec.data.util import load_example_obsconf
from jaxspec.fit import MCMCFitter
from jaxspec.model.additive import Blackbodyrad, Powerlaw
from jaxspec.model.multiplicative import Tbabs

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

prior_shared_pars = {
    "powerlaw_1_alpha": dist.Uniform(0, 5),
    "powerlaw_1_norm": dist.LogUniform(1e-5, 1e-2),
    "blackbodyrad_1_kT": dist.Uniform(0, 5),
    "blackbodyrad_1_norm": dist.LogUniform(1e-2, 1e2),
    "tbabs_1_nh": dist.Uniform(0, 1),
}

prior_split_pars = {
    "powerlaw_1_alpha": dist.Uniform(0, 5),
    "powerlaw_1_norm": dist.LogUniform(1e-5 * jnp.ones(3), 1e-2 * jnp.ones(3)),
    "blackbodyrad_1_kT": dist.Uniform(0, 5),
    "blackbodyrad_1_norm": dist.LogUniform(1e-2, 1e2),
    "tbabs_1_nh": dist.Uniform(0, 1),
}

prior = {
    "powerlaw_1_alpha": dist.Uniform(0, 5),
    "powerlaw_1_norm": dist.LogUniform(1e-5, 1e-2),
    "blackbodyrad_1_kT": dist.Uniform(0, 5),
    "blackbodyrad_1_norm": dist.LogUniform(1e-2, 1e2),
    "tbabs_1_nh": dist.Uniform(0, 1),
}

single_obsconf = load_example_obsconf("NGC7793_ULX4_PN")
list_of_obsconf = list(load_example_obsconf("NGC7793_ULX4_ALL").values())
dict_of_obsconf = load_example_obsconf("NGC7793_ULX4_ALL")

# Dir containing 8 files
parent_directory = Path(__file__).parent.resolve()
data_directory = parent_directory / "data"

if not data_directory.exists():
    os.mkdir(data_directory)

with open(parent_directory / "data_files.yml") as file:
    data_collection = yaml.safe_load(file)

with open(parent_directory / "data_hash.yml") as file:
    data_hash = yaml.safe_load(file)

pooch_dataset = pooch.create(
    base_url="https://github.com/HEACIT/curated-test-data/raw/main/",
    path=str(data_directory),
    registry=data_hash,
)

for file in data_hash.keys():
    pooch_dataset.fetch(file)


@pytest.fixture(scope="session")
def obsconfs():
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
    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.multiplicative import Tbabs

    model = Tbabs() * (Powerlaw() + Blackbodyrad())

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


spectral_model = Tbabs() * (Powerlaw() + Blackbodyrad())


@pytest.fixture(scope="session")
def prior_with_free_pars_on_obs():
    """
    Prior distribution with a free normalisation for each observation
    """

    return


@pytest.fixture(scope="session")
@pytest.mark.parametrize("obsconf, model, prior", ["NGC7793_ULX4_ALL"])
def fitting_setup_multiple_obs_and_free_pars(obsconf, model, prior):
    model = Tbabs() * (Powerlaw() + Blackbodyrad())

    return obsconfs, model, prior
