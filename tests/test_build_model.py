"""Happy-path construction of :class:`BayesianModel` and prior-predictive checks.

Error-path / validation tests live in ``test_prior_validation.py``; private
prior-resolution helpers are unit-tested in ``test_prior_resolution.py``.
"""

import matplotlib.pyplot as plt
import numpyro.distributions as dist
import pytest

from helpers import (
    data_prior_marker,
    list_of_obsconf,
    prior_shared_pars,
    single_obsconf,
    sparsify_marker,
    spectral_model,
)
from jaxspec.fit import BayesianModel, MCMCFitter
from jaxspec.model.abc import SpectralModel
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import BackgroundWithError


@sparsify_marker
@data_prior_marker
def test_build_model(model, prior, obsconf, expectation, sparse):
    """Try to build a model from the given combination of observation and priors"""
    with expectation:
        BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)


def test_build_model_from_bare_component():
    """A single ModelComponent is auto-wrapped via SpectralModel.from_component."""
    model = Powerlaw()  # bare component, not a SpectralModel
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
    }
    bm = BayesianModel(model, prior, single_obsconf)
    assert isinstance(bm.spectral_model, SpectralModel)
    # Leaves nest under the conventional component name → prior keys resolve.
    bm.prior_samples(num_samples=1)


@pytest.mark.slow
@sparsify_marker
@data_prior_marker
def test_mock_obs(model, prior, obsconf, expectation, sparse):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        bayesian_model = BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)
        bayesian_model.mock_observations(bayesian_model.prior_samples())


def test_prior_samples_with_ragged_background_default_prior():
    fitter = MCMCFitter(
        spectral_model,
        prior_shared_pars,
        list_of_obsconf,
        background_model=BackgroundWithError(),
    )
    prior_samples = fitter.prior_samples(num_samples=1)

    for i, obsconf in enumerate(list_of_obsconf):
        # Per-obs sites are registered under the literal "forward." prefix.
        site_name = f"forward.background.data_{i}.countrate"
        assert site_name in prior_samples
        assert prior_samples[site_name].shape == (1, len(obsconf.folded_background))


@pytest.mark.slow
def test_prior_predictive_coverage(obs_model_prior):
    obsconfs, model, prior = obs_model_prior
    BayesianModel(model, prior, obsconfs).prior_predictive_coverage()


@pytest.mark.parametrize(
    ("kwargs", "show_calls"),
    [
        pytest.param({"min_counts": 20}, 1, id="min-counts"),
        pytest.param({"grouping": 4}, 1, id="grouping"),
    ],
)
def test_prior_predictive_coverage_rebinning(monkeypatch, kwargs, show_calls):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    BayesianModel(spectral_model, prior_shared_pars, single_obsconf).prior_predictive_coverage(
        num_samples=4, **kwargs
    )

    assert calls == ["show"] * show_calls
