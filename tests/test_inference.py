"""End-to-end inference smoke tests across the MCMC / NS / VI backends, plus
instrument-calibration, tied-parameter and per-obs-chain coverage. Each fit is
deliberately tiny; the shared :func:`assert_result_smoke` helper exercises the
common post-fit API surface (fluxes, c-stat, PPC branches, chain export)."""

import numpy as np
import numpyro.distributions as dist
import pytest

from helpers import (
    SHORT_MCMC_FIT,
    assert_result_smoke,
    data_prior_marker,
    dict_of_obsconf,
    mcmc_marker,
    prior_shared_pars,
    spectral_model,
)
from jaxspec.fit import MCMCFitter, NSFitter, TiedParameter, VIFitter
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import BackgroundWithError, SpectralModelBackground
from jaxspec.model.instrument import ConstantGain, ConstantShift, InstrumentModel
from jaxspec.model.multiplicative import Tbabs
from numpyro.optim import optax_to_numpyro
from optax import adamw


@pytest.mark.slow
@mcmc_marker
@data_prior_marker
def test_run_mcmc(model, prior, obsconf, expectation, sampler):
    """Fit each (observation, prior, sampler) combination through the MCMC backend."""
    with expectation:
        forward_model = MCMCFitter(model, prior, obsconf, background_model=BackgroundWithError())
        result = forward_model.fit(**SHORT_MCMC_FIT, sampler=sampler)
        assert_result_smoke(result)


@pytest.mark.slow
@data_prior_marker
def test_run_ns(model, prior, obsconf, expectation):
    """Fit through the nested-sampling backend with a spectral-model background."""
    with expectation:
        bkg_spectral_model = Tbabs() * Powerlaw()
        # SpectralModelBackground exposes its inner spec's components directly
        # under the ``background.`` prefix via the ``user_path`` hook — users
        # write the same keys they would for the source spectrum.
        prior_with_backgrounds = {
            **prior,
            "background.powerlaw_1.alpha": dist.Uniform(0, 5),
            "background.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
            "background.tbabs_1.nh": dist.Uniform(0, 1),
        }
        forward_model = NSFitter(
            model,
            prior_with_backgrounds,
            obsconf,
            background_model=SpectralModelBackground(bkg_spectral_model),
        )
        # Aggressive stopping criterion: keep this on par with the MCMC/VI tests.
        result = forward_model.fit(
            num_samples=10,
            num_live_points=50,
            termination_kwargs={"max_samples": 200},
            verbose=False,
        )
        assert_result_smoke(result)


@pytest.mark.slow
@data_prior_marker
def test_run_vi(model, prior, obsconf, expectation):
    """Fit through the variational-inference backend."""
    with expectation:
        forward_model = VIFitter(model, prior, obsconf, background_model=BackgroundWithError())
        optim = optax_to_numpyro(adamw(3e-4))

        result = forward_model.fit(
            num_steps=100,
            num_samples=10,
            optimizer=optim,
            plot_diagnostics=True,
        )
        assert_result_smoke(result)


@pytest.mark.slow
@mcmc_marker
def test_instrument_model_building(sampler):
    prior_with_instruments = {
        **prior_shared_pars,
        "instrument.gain.factor[*]": dist.Uniform(0.8, 1.2),
        "instrument.shift.offset[*]": dist.Uniform(-0.1, +0.1),
    }
    forward_model = MCMCFitter(
        spectral_model,
        prior_with_instruments,
        dict_of_obsconf,
        background_model=None,
        instrument_model={
            "PN": None,  # explicit reference
            "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
        },
    )

    result = forward_model.fit(**SHORT_MCMC_FIT, sampler=sampler)
    assert_result_smoke(result)


@pytest.mark.slow
@mcmc_marker
def test_tied_parameters(sampler):
    spectral_model = Tbabs() * (Powerlaw() + Powerlaw())
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_2.alpha": TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 0.5 * x),
        "spectrum.powerlaw_2.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.tbabs_1.nh": 0.6,
    }

    forward_model = MCMCFitter(spectral_model, prior, dict_of_obsconf)
    result = forward_model.fit(**SHORT_MCMC_FIT, sampler=sampler)
    chain = assert_result_smoke(result)

    # Tied destinations are deterministic functions of their source; the user
    # convention is to exclude them from corner plots so the displayed
    # parameter set matches the genuinely sampled posterior.
    cols = list(chain.samples.columns)
    assert not any(
        "powerlaw_2.alpha" in c for c in cols
    ), f"tied destination 'powerlaw_2.alpha' should not be in chain columns: {cols}"
    assert any(
        "powerlaw_1.alpha" in c for c in cols
    ), f"tied source 'powerlaw_1.alpha' should remain in chain columns: {cols}"


@pytest.mark.slow
def test_scoped_tied_parameters_fit_and_extraction():
    """Scoped TiedParameter entries must survive the full fit → posterior
    extraction path (``input_parameters`` / fluxes / PPC), not just prior
    sampling — extraction used to look up ``tied_to`` without parsing its
    ``[obs]`` scope and drop the direct entries of a mixed direct+tied base."""
    prior = {
        **{k: v for k, v in prior_shared_pars.items() if k != "spectrum.powerlaw_1.norm"},
        "spectrum.powerlaw_1.norm[MOS1]": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_1.norm[MOS2]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.5 * x
        ),
        "spectrum.powerlaw_1.norm[PN]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.25 * x
        ),
    }

    fitter = MCMCFitter(spectral_model, prior, dict_of_obsconf)
    result = fitter.fit(
        num_chains=2,
        num_warmup=5,
        num_samples=5,
        sampler="nuts",
        mcmc_kwargs={"progress_bar": False},
    )

    norm = np.asarray(result.input_parameters["spectrum.powerlaw_1.norm"])
    obs_order = list(dict_of_obsconf.keys())
    mos1 = norm[..., obs_order.index("MOS1")]
    np.testing.assert_allclose(norm[..., obs_order.index("MOS2")], 0.5 * mos1, rtol=1e-6)
    np.testing.assert_allclose(norm[..., obs_order.index("PN")], 0.25 * mos1, rtol=1e-6)

    result.photon_flux(0.7, 1.2, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    result.to_chain("test")


@pytest.mark.slow
def test_to_chain_includes_per_obs_columns():
    """[*] per-obs parameters should appear in to_chain output, one column per obs.

    Pinned to NUTS — the column structure is sampler-independent and ensemble
    samplers (AIES / ESS) want n_chains ≥ 2·n_params which we'd need to scale
    up for, defeating the point of a fast smoke test.
    """
    prior = {
        **{k: v for k, v in prior_shared_pars.items() if k != "spectrum.powerlaw_1.norm"},
        "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
    }
    forward = MCMCFitter(spectral_model, prior, dict_of_obsconf)
    result = forward.fit(
        num_chains=2,
        num_warmup=5,
        num_samples=5,
        sampler="nuts",
        mcmc_kwargs={"progress_bar": False},
    )
    chain = result.to_chain("test")
    cols = list(chain.samples.columns)

    # One per-obs column per observation, labeled "<param>\n[<obs>]"
    per_obs_norm_cols = [c for c in cols if "powerlaw_1.norm" in c and "\n[" in c]
    assert len(per_obs_norm_cols) == len(
        dict_of_obsconf
    ), f"expected {len(dict_of_obsconf)} per-obs norm columns, got {per_obs_norm_cols}"
    obs_names_in_cols = {c.split("\n[")[1].rstrip("]") for c in per_obs_norm_cols}
    assert obs_names_in_cols == set(dict_of_obsconf.keys())


@pytest.mark.slow
def test_convergence(get_individual_mcmc_results, get_joint_mcmc_result, get_failed_mcmc_results):
    for result in get_individual_mcmc_results + get_joint_mcmc_result:
        assert result.converged

    for result in get_failed_mcmc_results:
        assert not result.converged


@pytest.mark.slow
def test_sparsify_matrix_in_model(obs_model_prior):
    obsconfigurations, model, prior = obs_model_prior

    for obsconf in obsconfigurations:
        forward_model = MCMCFitter(
            model, prior, obsconf, background_model=None, sparsify_matrix=True
        )
        forward_model.fit(**SHORT_MCMC_FIT)
