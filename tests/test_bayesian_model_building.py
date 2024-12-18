from contextlib import nullcontext as does_not_raise

import pytest

from conftest import (
    dict_of_obsconf,
    list_of_obsconf,
    prior_shared_pars,
    prior_split_pars,
    single_obsconf,
    spectral_model,
)
from jaxspec.fit import BayesianModel, MCMCFitter

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
        forward_model.fit(
            num_chains=4,
            num_warmup=10,
            num_samples=10,
            sampler=sampler,
            mcmc_kwargs={"progress_bar": False},
        )
