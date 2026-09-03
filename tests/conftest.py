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

    ``xspec`` tests are promoted to ``slow`` first: they shell out to a live PyXSPEC and
    take minutes, so on a machine that *has* HEASOFT they would otherwise be pulled into
    ``-m fast`` / ``-m "not slow"``.
    """
    for item in items:
        if "xspec" in item.keywords:
            item.add_marker(pytest.mark.slow)
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


# --- Fast fixtures for the structural smoke test -----------------------------
#
# A 20-warmup / 20-sample / 1-chain fit produces a fully functional FitResult in
# ~11-14 s. That is the whole point: the suites that exercise the numpyro trace
# path (test_inference / test_results) take ~23 minutes, and `-m "not slow"` does
# not cover that path at all — a change can pass the fast suite and still break
# every fit. These fixtures give the refactor work a ~30 s regression net.


def _quick_fit(model, prior, obsconf, **kwargs):
    return MCMCFitter(model, prior, obsconf, **kwargs).fit(
        num_warmup=20, num_samples=20, num_chains=1, mcmc_kwargs={"progress_bar": False}
    )


@pytest.fixture(scope="session")
def quick_result(obs_model_prior):
    """One observation, all-shared prior — the simplest complete FitResult."""
    obsconfs, model, prior = obs_model_prior
    return _quick_fit(model, prior, obsconfs[0])


@pytest.fixture(scope="session")
def quick_joint_result(obs_model_prior):
    """Three observations, all-shared prior — exercises the trailing obs axis."""
    obsconfs, model, prior = obs_model_prior
    return _quick_fit(model, prior, obsconfs)


@pytest.fixture(scope="session")
def quick_disk_result(obsconfs):
    """A fit whose model publishes a derived quantity (``Diskbb`` is the shipped one).

    Separate from ``quick_result``, whose exact chain-column set pins that models
    publishing nothing are unaffected.
    """
    import numpyro.distributions as dist

    from jaxspec.model.additive import Diskbb
    from jaxspec.model.multiplicative import Tbabs

    prior = {
        "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
        "spectrum.diskbb_1.Tin": dist.Uniform(0.1, 5.0),
        "spectrum.diskbb_1.norm": dist.LogUniform(1e-6, 1e-2),
    }
    return _quick_fit(Tbabs() * Diskbb(), prior, obsconfs[0])


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
