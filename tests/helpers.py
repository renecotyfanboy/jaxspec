"""Shared building blocks for the jaxspec test suite.

Constants, parametrize markers, and post-fit smoke helpers live here (rather
than in ``conftest``) so test modules can import them explicitly. The module is
kept import-time light: the only I/O is loading the bundled example
observations once — the ``@pytest.mark.parametrize`` markers below need those
objects at *collection* time, which rules out making them fixtures.

``conftest`` runs the chex / x64 / numpyro device setup before this module is
ever imported (pytest imports ``conftest`` first), so loading the example data
here is safe.
"""

from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpyro.distributions as dist
import pytest
import yaml

from jaxspec.data.util import load_example_obsconf
from jaxspec.model.additive import Blackbodyrad, Powerlaw
from jaxspec.model.multiplicative import Tbabs

# --- Spectral model + prior dicts --------------------------------------------

spectral_model = Tbabs() * (Powerlaw() + Blackbodyrad())

#: Every spectral parameter shared across all applicable observations.
prior_shared_pars = {
    "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
    "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
    "spectrum.blackbodyrad_1.kT": dist.Uniform(0, 5),
    "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
    "spectrum.tbabs_1.nh": dist.Uniform(0, 1),
}

#: As ``prior_shared_pars`` but with an independent per-obs powerlaw norm draw.
prior_split_pars = {
    "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
    "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
    "spectrum.blackbodyrad_1.kT": dist.Uniform(0, 5),
    "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
    "spectrum.tbabs_1.nh": dist.Uniform(0, 1),
}

# --- Example observations (loaded once, shared everywhere) --------------------

single_obsconf = load_example_obsconf("NGC7793_ULX4_PN")
dict_of_obsconf = load_example_obsconf("NGC7793_ULX4_ALL")
list_of_obsconf = list(dict_of_obsconf.values())

# --- Reusable parametrize markers --------------------------------------------

sparsify_marker = pytest.mark.parametrize(
    "sparse",
    [
        pytest.param(True, id="sparse matrix"),
        pytest.param(False, id="dense matrix"),
    ],
)

data_prior_marker = pytest.mark.parametrize(
    "model, prior, obsconf, expectation",
    [
        pytest.param(
            spectral_model,
            prior_shared_pars,
            single_obsconf,
            does_not_raise(),
            id="single observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            single_obsconf,
            does_not_raise(),
            id="single observation-split parameters",
        ),
        pytest.param(
            spectral_model,
            prior_shared_pars,
            list_of_obsconf,
            does_not_raise(),
            id="list of observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_shared_pars,
            dict_of_obsconf,
            does_not_raise(),
            id="dict of observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            list_of_obsconf,
            does_not_raise(),
            id="list of observation-split parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            dict_of_obsconf,
            does_not_raise(),
            id="dict of observation-split parameters",
        ),
    ],
)

mcmc_marker = pytest.mark.parametrize(
    "sampler",
    [
        pytest.param("nuts", id="NUTS"),
        pytest.param("aies", id="AIES"),
        pytest.param("ess", id="ESS"),
    ],
)

# --- Post-fit smoke helpers --------------------------------------------------

#: Short MCMC fit settings shared by the end-to-end inference smoke tests.
SHORT_MCMC_FIT = dict(
    num_chains=4,
    num_warmup=10,
    num_samples=10,
    mcmc_kwargs={"progress_bar": False},
)


def assert_result_smoke(result, *, redshift=0.01, e_min=0.7, e_max=1.2):
    """Exercise the common post-fit API surface used across inference tests.

    Computes/registers the standard fluxes, touches ``c_stat``, builds every
    per-obs PPC branch, and serialises to a chain. Returns the chain so callers
    can make additional assertions on its columns.
    """
    result.photon_flux(e_min, e_max, register=True)
    result.energy_flux(e_min, e_max, register=True)
    result.luminosity(e_min, e_max, redshift=redshift, register=True)
    result.c_stat
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    return result.to_chain("test")


# --- Curated multi-mission test data (HEACIT) --------------------------------

_TESTS_DIR = Path(__file__).parent.resolve()
data_directory = _TESTS_DIR / "data"

with open(_TESTS_DIR / "data_files.yml") as _fh:
    #: List of curated observations (name + ogip paths) used by test_instruments.
    data_collection = yaml.safe_load(_fh)

with open(_TESTS_DIR / "data_hash.yml") as _fh:
    _data_hash = yaml.safe_load(_fh)


def download_curated_data():
    """Fetch the HEACIT curated multi-mission test files via pooch (idempotent)."""
    import pooch

    data_directory.mkdir(exist_ok=True)
    dataset = pooch.create(
        base_url="https://github.com/HEACIT/curated-test-data/raw/main/",
        path=str(data_directory),
        registry=_data_hash,
        retry_if_failed=10,
    )
    for fname in _data_hash:
        dataset.fetch(fname)
    return data_directory


def curated_params(unsupported: set[str], reason: str):
    """Build a ``parametrize`` list over ``data_collection``.

    Observations whose ``name`` is in ``unsupported`` are tagged with a real
    ``skip`` mark (and ``reason``) so they surface as skipped rather than as a
    silent vacuous pass.
    """
    return [
        pytest.param(
            obs,
            id=obs["name"],
            marks=pytest.mark.skip(reason=reason) if obs["name"] in unsupported else (),
        )
        for obs in data_collection
    ]
