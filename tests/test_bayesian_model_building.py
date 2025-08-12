from contextlib import nullcontext as does_not_raise

import numpyro.distributions as dist
import pytest

from conftest import (
    dict_of_obsconf,
    list_of_obsconf,
    prior_shared_pars,
    prior_split_pars,
    single_obsconf,
    spectral_model,
)
from jaxspec.fit import BayesianModel, MCMCFitter, TiedParameter, VIFitter
from jaxspec.model.additive import Powerlaw
from jaxspec.model.instrument import ConstantGain, ConstantShift, InstrumentModel
from jaxspec.model.multiplicative import Tbabs
from numpyro.optim import optax_to_numpyro
from optax import adamw

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
            pytest.raises(ValueError),
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


@pytest.mark.fast
@sparsify_marker
@data_prior_marker
def test_build_model(model, prior, obsconf, expectation, sparse):
    """Try to build a model from the given combination of observation and priors"""
    with expectation:
        BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)


@pytest.mark.slow
@sparsify_marker
@data_prior_marker
def test_mock_obs(model, prior, obsconf, expectation, sparse):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        bayesian_model = BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)
        bayesian_model.mock_observations(bayesian_model.prior_samples())


@pytest.mark.slow
@mcmc_marker
@data_prior_marker
def test_run_mcmc(model, prior, obsconf, expectation, sampler):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = MCMCFitter(model, prior, obsconf, background_model=None)
        result = forward_model.fit(
            num_chains=4,
            num_warmup=10,
            num_samples=10,
            sampler=sampler,
            mcmc_kwargs={"progress_bar": False},
        )

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@data_prior_marker
def test_run_vi(model, prior, obsconf, expectation):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = VIFitter(model, prior, obsconf, background_model=None)
        optim = optax_to_numpyro(adamw(3e-4))

        result = forward_model.fit(
            num_steps=100,
            num_samples=10,
            optimizer=optim,
            plot_diagnostics=True,
        )

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@mcmc_marker
def test_instrument_model_building(sampler):
    forward_model = MCMCFitter(
        spectral_model,
        prior_shared_pars,
        dict_of_obsconf,
        background_model=None,
        instrument_model=InstrumentModel(
            "PN",
            gain_model=ConstantGain(dist.Uniform(0.8, 1.2)),
            shift_model=ConstantShift(dist.Uniform(-0.1, +0.1)),
        ),
    )

    result = forward_model.fit(
        num_chains=4,
        num_warmup=10,
        num_samples=10,
        sampler=sampler,
        mcmc_kwargs={"progress_bar": False},
    )

    result.photon_flux(0.7, 1.2, register=True)
    result.energy_flux(0.7, 1.2, register=True)
    result.luminosity(0.7, 1.2, redshift=0.01, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    result.to_chain("test")


@pytest.mark.slow
@mcmc_marker
def test_tied_parameters(sampler):
    spectral_model = Tbabs() * (Powerlaw() + Powerlaw())
    prior = {
        "powerlaw_1_alpha": dist.Uniform(0, 5),
        "powerlaw_1_norm": dist.LogUniform(1e-5, 1e-2),
        "powerlaw_2_alpha": TiedParameter("powerlaw_1_alpha", lambda x: 0.5 * x),
        "powerlaw_2_norm": dist.LogUniform(1e-5, 1e-2),
        "tbabs_1_nh": 0.6,
    }

    forward_model = MCMCFitter(spectral_model, prior, dict_of_obsconf)

    result = forward_model.fit(
        num_chains=4,
        num_warmup=10,
        num_samples=10,
        sampler=sampler,
        mcmc_kwargs={"progress_bar": False},
    )

    result.photon_flux(0.7, 1.2, register=True)
    result.energy_flux(0.7, 1.2, register=True)
    result.luminosity(0.7, 1.2, redshift=0.01, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    result.to_chain("test")
