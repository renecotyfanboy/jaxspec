import pytest

from jaxspec.fit import BayesianModel


def test_convergence(get_individual_mcmc_results, get_joint_mcmc_result):
    for result in get_individual_mcmc_results + get_joint_mcmc_result:
        assert result.converged


def test_ns(obs_model_prior):
    NSFitter = pytest.importorskip("jaxspec.fit.NSFitter")

    obsconfs, model, prior = obs_model_prior

    obsconf = obsconfs[0]
    fitter = NSFitter(model, prior, obsconf)
    fitter.fit(num_samples=10000, num_live_points=1000, plot_diagnostics=True)


def test_prior_predictive_coverage(obs_model_prior):
    obsconfs, model, prior = obs_model_prior
    BayesianModel(model, prior, obsconfs).prior_predictive_coverage()
