# ruff: noqa: E402

import chex

chex.set_n_cpu_devices(n=4)

import matplotlib

matplotlib.use("Agg")  # headless backend — avoids the macOS _macosx segfault

import matplotlib.pyplot as plt
import numpyro
import pytest

from jax import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

# Imported after the device setup above so the example-data load it triggers
# sees the configured CPU device count and x64.
from helpers import list_of_obsconf, prior_shared_pars, spectral_model
from jaxspec.fit import MCMCFitter


def pytest_collection_modifyitems(items):
    """Complete the fast/slow taxonomy: anything not explicitly ``slow`` is ``fast``.

    Keeps the marker split exhaustive without sprinkling ``@pytest.mark.fast``
    everywhere, and auto-classifies future tests. CI's ``-m "not slow"`` selector
    is unaffected.
    """
    for item in items:
        if "slow" not in item.keywords:
            item.add_marker(pytest.mark.fast)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figures a test opened, preventing cross-test leakage."""
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def obsconfs():
    return list_of_obsconf


@pytest.fixture(scope="session")
def observations():
    from jaxspec.data.util import load_example_pha

    return list(load_example_pha("NGC7793_ULX4_ALL").values())


@pytest.fixture(scope="session")
def instruments():
    from jaxspec.data.util import load_example_instruments

    return list(load_example_instruments("NGC7793_ULX4_ALL").values())


@pytest.fixture(scope="session")
def curated_data_dir():
    """Download the HEACIT curated multi-mission files; only the curated-data
    tests depend on this, so unrelated runs never trigger the fetch."""
    from helpers import download_curated_data

    return download_curated_data()


@pytest.fixture(scope="session")
def obs_model_prior(obsconfs):
    return obsconfs, spectral_model, prior_shared_pars


@pytest.fixture(scope="session")
def get_individual_mcmc_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [MCMCFitter(model, prior, obsconf).fit(num_samples=1000) for obsconf in obsconfs]


@pytest.fixture(scope="session")
def get_failed_mcmc_results(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [
        MCMCFitter(model, prior, obsconf).fit(num_warmup=10, num_samples=10) for obsconf in obsconfs
    ]


@pytest.fixture(scope="session")
def get_joint_mcmc_result(obs_model_prior):
    obsconfs, model, prior = obs_model_prior

    return [MCMCFitter(model, prior, obsconfs).fit(num_samples=1000)]


@pytest.fixture(scope="session")
def get_result_list(get_individual_mcmc_results, get_joint_mcmc_result):
    result_list = []
    result_list += get_individual_mcmc_results
    result_list += get_joint_mcmc_result

    name_list = []
    name_list += ["PN_mcmc", "MOS1_mcmc", "MOS2_mcmc"]
    name_list += ["Joint_mcmc"]

    return name_list, result_list
