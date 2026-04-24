from contextlib import nullcontext as does_not_raise

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
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
from jaxspec.fit import BayesianModel, MCMCFitter, NSFitter, PerObs, TiedParameter, VIFitter
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import (
    BackgroundWithError,
    SpectralModelBackground,
)
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


@pytest.mark.fast
def test_prior_samples_with_ragged_background_default_prior():
    fitter = MCMCFitter(
        spectral_model,
        prior_shared_pars,
        list_of_obsconf,
        background_model=BackgroundWithError(),
    )
    prior_samples = fitter.prior_samples(num_samples=1)

    for i, obsconf in enumerate(list_of_obsconf):
        site_name = f"background.countrate.data_{i}"
        assert site_name in prior_samples
        assert prior_samples[site_name].shape == (1, len(obsconf.folded_background))


@pytest.mark.slow
@mcmc_marker
@data_prior_marker
def test_run_mcmc(model, prior, obsconf, expectation, sampler):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = MCMCFitter(model, prior, obsconf, background_model=BackgroundWithError())
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
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@data_prior_marker
def test_run_ns(model, prior, obsconf, expectation):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        bkg_spectral_model = Tbabs() * Powerlaw()
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

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@data_prior_marker
def test_run_vi(model, prior, obsconf, expectation):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = VIFitter(model, prior, obsconf, background_model=BackgroundWithError())
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
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.fast
def test_nested_per_obs_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm": PerObs(PerObs(dist.LogUniform(1e-5, 1e-2))),
    }

    with pytest.raises(TypeError, match="PerObs inside PerObs"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_tied_parameter_inside_per_obs_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm": PerObs(
            TiedParameter("spectrum.blackbodyrad_1.norm", lambda x: 0.5 * x)
        ),
    }

    with pytest.raises(TypeError, match="TiedParameter inside PerObs"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_missing_per_obs_entries_raise_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm": PerObs({"data_0": dist.LogUniform(1e-5, 1e-2)}),
    }

    with pytest.raises(ValueError, match="missing observations"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.slow
@mcmc_marker
def test_instrument_model_building(sampler):
    prior_with_instruments = {
        **prior_shared_pars,
        "instrument.gain.factor": dist.Uniform(0.8, 1.2),
        "instrument.shift.offset": dist.Uniform(-0.1, +0.1),
    }
    forward_model = MCMCFitter(
        spectral_model,
        prior_with_instruments,
        dict_of_obsconf,
        background_model=None,
        instrument_model=InstrumentModel(
            "PN",
            gain_model=ConstantGain(),
            shift_model=ConstantShift(),
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
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_2.alpha": TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 0.5 * x),
        "spectrum.powerlaw_2.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.tbabs_1.nh": 0.6,
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


@pytest.mark.fast
def test_invalid_fixed_prior_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alpha": object(),
    }

    with pytest.raises(TypeError, match="Invalid fixed prior value"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_invalid_per_obs_fixed_prior_raises_at_build_time():
    class NotCastableAsArray:
        def __array__(self, dtype=None):
            raise TypeError("cannot convert to array")

    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm": PerObs(NotCastableAsArray()),
    }

    with pytest.raises(TypeError, match="Invalid fixed prior value"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_background_model_without_background_raises_on_sampling():
    obsconf_without_background = single_obsconf.copy(deep=True)
    del obsconf_without_background["folded_background"]
    prior = {
        **prior_shared_pars,
        "background.powerlaw_1.alpha": dist.Uniform(0, 5),
        "background.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
    }

    bayesian_model = BayesianModel(
        spectral_model,
        prior,
        obsconf_without_background,
        background_model=SpectralModelBackground(Powerlaw()),
    )

    with pytest.raises(ValueError, match="Trying to fit a background model but no background"):
        bayesian_model.prior_samples(num_samples=1)


@pytest.mark.fast
def test_extract_posterior_samples_stacks_matching_per_obs_shapes():
    observation_names = list(dict_of_obsconf)
    alpha = np.arange(6, dtype=float).reshape(2, 3)
    norms = {
        f"spectrum.powerlaw_1.norm.{obs_name}": np.full((2, 3), i + 1.0)
        for i, obs_name in enumerate(observation_names)
    }
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": PerObs(dist.LogUniform(1e-5, 1e-2)),
        "spectrum.blackbodyrad_1.kT": 1.5,
        "spectrum.blackbodyrad_1.norm": 2.5,
        "spectrum.tbabs_1.nh": 0.6,
    }
    inference_data = az.from_dict(
        posterior={
            "spectrum.powerlaw_1.alpha": alpha,
            **norms,
        }
    )

    extracted = spectral_model.extract_posterior_samples(
        inference_data,
        prior,
        observation_names,
    )

    np.testing.assert_allclose(
        extracted["spectrum.powerlaw_1.alpha"],
        np.broadcast_to(alpha[..., None], (*alpha.shape, len(observation_names))),
    )
    np.testing.assert_allclose(
        extracted["spectrum.powerlaw_1.norm"],
        np.stack(
            [norms[f"spectrum.powerlaw_1.norm.{obs_name}"] for obs_name in observation_names],
            axis=-1,
        ),
    )


@pytest.mark.fast
def test_extract_posterior_samples_keeps_ragged_per_obs_values_as_dict():
    observation_names = list(dict_of_obsconf)
    prior = {
        "background.countrate": PerObs(
            {
                observation_names[0]: np.array([1.0, 2.0]),
                observation_names[1]: np.array([3.0]),
                observation_names[2]: np.array([4.0, 5.0, 6.0]),
            }
        )
    }
    inference_data = az.from_dict(posterior={"dummy": np.zeros((1, 1))})

    extracted = BackgroundWithError().extract_posterior_samples(
        inference_data,
        prior,
        observation_names,
    )

    assert isinstance(extracted["background.countrate"], dict)
    assert list(extracted["background.countrate"]) == observation_names


@pytest.mark.fast
def test_register_priors_fixed_shared_value():
    observation_names = list(dict_of_obsconf)
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT": dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.tbabs_1.nh": 0.6,
    }

    with numpyro.handlers.seed(rng_seed=0):
        result = spectral_model.register_priors(prior, observation_names)

    assert set(result["spectrum.tbabs_1.nh"]) == set(observation_names)
    for obs_name in observation_names:
        np.testing.assert_allclose(result["spectrum.tbabs_1.nh"][obs_name], 0.6)


@pytest.mark.fast
def test_register_priors_fixed_per_obs_value():
    observation_names = list(dict_of_obsconf)
    per_obs_values = {obs: float(i + 1) for i, obs in enumerate(observation_names)}
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT": dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.tbabs_1.nh": PerObs(per_obs_values),
    }

    with numpyro.handlers.seed(rng_seed=0):
        result = spectral_model.register_priors(prior, observation_names)

    for obs_name, expected in per_obs_values.items():
        np.testing.assert_allclose(result["spectrum.tbabs_1.nh"][obs_name], expected)


@pytest.mark.fast
def test_register_priors_tied_parameter():
    observation_names = list(dict_of_obsconf)
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT": TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 2.0 * x),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.tbabs_1.nh": dist.Uniform(0, 1),
    }

    with numpyro.handlers.seed(rng_seed=0):
        result = spectral_model.register_priors(prior, observation_names)

    for obs_name in observation_names:
        np.testing.assert_allclose(
            result["spectrum.blackbodyrad_1.kT"][obs_name],
            2.0 * result["spectrum.powerlaw_1.alpha"][obs_name],
        )


@pytest.mark.fast
def test_register_priors_tied_parameter_unknown_source():
    observation_names = list(dict_of_obsconf)
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT": TiedParameter("spectrum.does_not_exist", lambda x: x),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.tbabs_1.nh": dist.Uniform(0, 1),
    }

    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(ValueError, match="unknown source"):
            spectral_model.register_priors(prior, observation_names)


@pytest.mark.fast
def test_extract_posterior_samples_tied_parameter_shared_source():
    observation_names = list(dict_of_obsconf)
    alpha = np.arange(6, dtype=float).reshape(2, 3)
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": 1e-3,
        "spectrum.blackbodyrad_1.kT": TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 2.0 * x),
        "spectrum.blackbodyrad_1.norm": 1.0,
        "spectrum.tbabs_1.nh": 0.6,
    }
    inference_data = az.from_dict(posterior={"spectrum.powerlaw_1.alpha": alpha})

    extracted = spectral_model.extract_posterior_samples(inference_data, prior, observation_names)

    assert not isinstance(extracted["spectrum.blackbodyrad_1.kT"], dict)
    np.testing.assert_allclose(
        extracted["spectrum.blackbodyrad_1.kT"],
        2.0 * extracted["spectrum.powerlaw_1.alpha"],
    )


@pytest.mark.fast
def test_extract_posterior_samples_tied_parameter_ragged_source():
    observation_names = list(dict_of_obsconf)
    ragged_values = {
        observation_names[0]: np.array([1.0, 2.0]),
        observation_names[1]: np.array([3.0]),
        observation_names[2]: np.array([4.0, 5.0, 6.0]),
    }
    prior = {
        "background.countrate": PerObs(ragged_values),
        "background.derived": TiedParameter("background.countrate", lambda x: 10.0 * x),
    }
    inference_data = az.from_dict(posterior={"dummy": np.zeros((1, 1))})

    extracted = BackgroundWithError().extract_posterior_samples(
        inference_data, prior, observation_names
    )

    assert isinstance(extracted["background.derived"], dict)
    for obs_name, source_value in ragged_values.items():
        source_leaf = extracted["background.countrate"][obs_name]
        np.testing.assert_allclose(extracted["background.derived"][obs_name], 10.0 * source_leaf)
        np.testing.assert_allclose(jnp.squeeze(source_leaf), jnp.asarray(source_value))


@pytest.mark.fast
def test_extract_posterior_samples_tied_parameter_unknown_source():
    observation_names = list(dict_of_obsconf)
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.kT": TiedParameter("spectrum.does_not_exist", lambda x: x),
        "spectrum.powerlaw_1.norm": 1e-3,
        "spectrum.blackbodyrad_1.norm": 1.0,
        "spectrum.tbabs_1.nh": 0.6,
    }
    inference_data = az.from_dict(posterior={"spectrum.powerlaw_1.alpha": np.zeros((2, 3))})

    with pytest.raises(ValueError, match="unknown source"):
        spectral_model.extract_posterior_samples(inference_data, prior, observation_names)
