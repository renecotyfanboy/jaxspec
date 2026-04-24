import matplotlib.pyplot as plt
import pytest

from conftest import prior_shared_pars, single_obsconf, spectral_model
from jaxspec.fit import BayesianModel


@pytest.mark.slow
def test_convergence(get_individual_mcmc_results, get_joint_mcmc_result, get_failed_mcmc_results):
    for result in get_individual_mcmc_results + get_joint_mcmc_result:
        assert result.converged

    for result in get_failed_mcmc_results:
        assert not result.converged


@pytest.mark.slow
def test_ns(obs_model_prior):
    NSFitter = pytest.importorskip("jaxspec.fit.NSFitter")

    obsconfs, model, prior = obs_model_prior

    obsconf = obsconfs[0]
    fitter = NSFitter(model, prior, obsconf)
    fitter.fit(num_samples=10000, num_live_points=1000, plot_diagnostics=True)


@pytest.mark.slow
def test_prior_predictive_coverage(obs_model_prior):
    obsconfs, model, prior = obs_model_prior
    BayesianModel(model, prior, obsconfs).prior_predictive_coverage()


@pytest.mark.fast
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
