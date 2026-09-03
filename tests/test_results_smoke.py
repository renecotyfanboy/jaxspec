"""Structural smoke test over the whole public ``FitResult`` surface.

Why this exists: the suites that actually exercise the numpyro trace path
(``test_inference.py`` + ``test_results.py``) take ~23 minutes, and
``pytest -m "not slow"`` does **not** cover that path — a change can pass the fast suite
and still break every fit. These tests run a ~8 s fit once per session and then assert on
the *structure* of everything ``FitResult`` exposes: key sets, shapes, dimension names,
column names. They deliberately assert nothing about float values, so they stay stable
across samplers and platforms while still failing loudly on a refactor that drops a
variable, renames a site, or changes an array's shape.
"""

import numpy as np
import pytest

# (chain, draw) of the quick fixtures — see conftest._quick_fit.
CHAIN_DRAW = (1, 20)
N_OBS = 3

SHARED_PARAMS = {
    "spectrum.blackbodyrad_1.kT",
    "spectrum.blackbodyrad_1.norm",
    "spectrum.powerlaw_1.alpha",
    "spectrum.powerlaw_1.norm",
    "spectrum.tbabs_1.nh",
}


def test_parameter_names(quick_joint_result):
    """Only free parameters — no deterministic / observed sites."""
    assert set(quick_joint_result.bayesian_fitter.parameter_names) == SHARED_PARAMS


def test_input_parameters_keys_and_shapes(quick_joint_result):
    """Shared params carry a trailing observation axis."""
    params = quick_joint_result.input_parameters

    assert set(params) == SHARED_PARAMS
    for name, value in params.items():
        assert np.shape(value) == (*CHAIN_DRAW, N_OBS), name


def test_spectrum_parameters_is_the_spectrum_subset(quick_joint_result):
    spectrum = quick_joint_result.spectrum_parameters

    assert set(spectrum) == SHARED_PARAMS
    assert all(name.startswith("spectrum.") for name in spectrum)


def test_to_chain_columns(quick_joint_result):
    """Column set is the public contract for corner plots and the LaTeX table."""
    columns = set(quick_joint_result.to_chain("smoke").samples.columns)

    # chainconsumer adds its own weight column.
    assert columns == {name.removeprefix("spectrum.") for name in SHARED_PARAMS} | {"weight"}


def test_c_stat_variables(quick_joint_result):
    variables = set(quick_joint_result.c_stat.data_vars)

    assert {"full", "observed.all"} <= variables
    assert sum(name.startswith("observed.data_") for name in variables) == N_OBS


def test_log_likelihood_variables(quick_joint_result):
    """`log_likelihood` is an xr.Dataset carrying one term per obs plus the aggregates."""
    log_likelihood = quick_joint_result.log_likelihood
    variables = set(log_likelihood.data_vars)

    assert {"full", "observed.all"} <= variables
    assert sum(name.startswith("observed.data_") for name in variables) == N_OBS
    assert np.all(np.isfinite(log_likelihood["full"].to_numpy()))


def test_converged_is_a_bool(quick_joint_result):
    assert isinstance(quick_joint_result.converged, bool)


@pytest.mark.parametrize("kind", ["photon_flux", "energy_flux"])
def test_flux_shapes(quick_joint_result, kind):
    value = getattr(quick_joint_result, kind)(1.0, 5.0)

    assert np.shape(value) == (*CHAIN_DRAW, N_OBS)


@pytest.mark.parametrize("kind", ["photon_flux", "energy_flux"])
def test_flux_register_adds_a_posterior_variable(quick_result, kind):
    """`register=True` writes a `derived.*` variable with named dimensions."""
    getattr(quick_result, kind)(2.0, 6.0, register=True)

    name = f"derived.{kind}_2.0_6.0"
    posterior = quick_result.inference_data.posterior

    assert name in posterior.data_vars
    assert posterior[name].dims[:2] == ("chain", "draw")


def test_luminosity_shape(quick_joint_result):
    value = quick_joint_result.luminosity(1.0, 5.0, redshift=0.01)

    assert np.shape(value) == (*CHAIN_DRAW, N_OBS)


# Chainconsumer summary statistics need more than this smoke fixture's 20 broad-prior
# draws. Slow result tests cover rendering; ``test_to_chain_columns`` covers its input.


@pytest.mark.parametrize("scale", ["linear", "semilogx", "semilogy", "loglog"])
def test_plot_ppc_scales(quick_result, scale):
    figures = quick_result.plot_ppc(scale=scale)

    assert len(figures) == 1


@pytest.mark.parametrize("y_type", ["counts", "countrate", "photon_flux", "photon_flux_density"])
def test_plot_ppc_y_types(quick_result, y_type):
    assert len(quick_result.plot_ppc(y_type=y_type)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_components": True},
        {"plot_background": True},
        {"min_counts": 10},
        {"grouping": 10},
        {"x_lims": (1.0, 8.0)},
        {"n_sigmas": 2},
        {"x_unit": "angstrom"},
    ],
    ids=["components", "background", "min_counts", "grouping", "x_lims", "n_sigmas", "x_unit"],
)
def test_plot_ppc_options(quick_result, kwargs):
    """Every documented ``plot_ppc`` option remains wired."""
    assert len(quick_result.plot_ppc(**kwargs)) == 1


def test_plot_ppc_joint_returns_one_figure_per_observation(quick_joint_result):
    assert len(quick_joint_result.plot_ppc()) == N_OBS


def test_shared_instrument_prior_reaches_the_chain(obs_model_prior):
    """A shared instrument prior remains visible under its bare chain key."""
    import numpyro.distributions as dist

    from jaxspec.fit import MCMCFitter
    from jaxspec.model.instrument import ConstantGain, InstrumentModel

    obsconfs, model, prior = obs_model_prior
    result = MCMCFitter(
        model,
        {**prior, "instrument.gain.factor": dist.Uniform(0.9, 1.1)},
        obsconfs,
        # List inputs are auto-named data_0, data_1, ...
        instrument_model={
            f"data_{i}": InstrumentModel(gain=ConstantGain()) for i in range(len(obsconfs))
        },
    ).fit(num_warmup=20, num_samples=20, num_chains=1, mcmc_kwargs={"progress_bar": False})

    assert any("gain" in column for column in result.to_chain("smoke").samples.columns)


# --- Component-published derived quantities ----------------------------------


def test_published_quantity_reaches_posterior_prior_and_chain(quick_disk_result):
    """A published quantity remains visible across every result surface."""
    site = "derived.spectrum.diskbb_1.norm_xspec"
    inference_data = quick_disk_result.inference_data

    assert site in inference_data.posterior.data_vars
    assert site in inference_data.prior.data_vars
    assert inference_data.posterior[site].dims[:2] == ("chain", "draw")
    assert "diskbb_1.norm_xspec" in quick_disk_result.to_chain("smoke").samples.columns


def test_published_quantity_tracks_the_posterior_draws(quick_disk_result):
    """Published values are finite and follow the parameters in each posterior draw."""
    from jaxspec.model._additive.disk import STANDARD_RADIAL_EXPONENT, _disk_band_photon_flux

    posterior = quick_disk_result.inference_data.posterior
    published = posterior["derived.spectrum.diskbb_1.norm_xspec"].values
    tin = posterior["spectrum.diskbb_1.Tin"].values
    norm = posterior["spectrum.diskbb_1.norm"].values

    assert np.all(np.isfinite(published))
    band_flux = _disk_band_photon_flux(tin, STANDARD_RADIAL_EXPONENT, 0.5, 10.0)
    assert np.allclose(published, norm / np.asarray(band_flux))


def test_published_quantity_is_not_a_free_parameter(quick_disk_result):
    parameter_names = quick_disk_result.bayesian_fitter.parameter_names

    assert not any(name.startswith("derived.") for name in parameter_names)
