"""Public-interface tests for :class:`BayesianModel`: the likelihood / posterior
methods and the flat array <-> dict parameter interfaces used by external samplers.

Forward-model normalisation/validation helpers are tested in ``test_forward_model.py``;
prior-resolution internals in ``test_prior_resolution.py``."""

import operator

import jax
import jax.numpy as jnp

from helpers import prior_shared_pars, single_obsconf, spectral_model
from jaxspec.fit import BayesianModel


def _example_parameter_dict() -> dict[str, float]:
    return {
        "spectrum.powerlaw_1.alpha": 1.7,
        "spectrum.powerlaw_1.norm": 3e-4,
        "spectrum.blackbodyrad_1.kT": 0.7,
        "spectrum.blackbodyrad_1.norm": 10.0,
        "spectrum.tbabs_1.nh": 0.2,
    }


def test_likelihood():
    bayesian_model = BayesianModel(spectral_model, prior_shared_pars, single_obsconf)

    parameter_array = bayesian_model.dict_to_array(_example_parameter_dict())
    parameters = bayesian_model.array_to_dict(parameter_array)

    total_likelihood = bayesian_model.log_likelihood(parameters)
    splitted_likelihood = bayesian_model.log_likelihood_per_obs(parameters)
    total_likelihood_from_splitted = jax.tree.reduce(
        operator.add, jax.tree.map(jnp.sum, splitted_likelihood)
    )

    assert jnp.isclose(total_likelihood_from_splitted, total_likelihood)


def test_external_sampler_style_interfaces():
    bayesian_model = BayesianModel(spectral_model, prior_shared_pars, single_obsconf)

    assert bayesian_model.parameter_names == sorted(prior_shared_pars)

    theta = bayesian_model.dict_to_array(_example_parameter_dict())
    parameters = bayesian_model.array_to_dict(theta)

    assert set(parameters) == set(bayesian_model.parameter_names)
    assert jnp.allclose(bayesian_model.dict_to_array(parameters), theta)

    @jax.jit
    def log_likelihood_from_array(theta):
        return bayesian_model.log_likelihood(bayesian_model.array_to_dict(theta))

    @jax.jit
    def log_likelihood_per_obs_from_array(theta):
        return bayesian_model.log_likelihood_per_obs(bayesian_model.array_to_dict(theta))

    @jax.jit
    def log_posterior_from_array(theta):
        return bayesian_model.log_posterior_prob(bayesian_model.array_to_dict(theta))

    total_likelihood = log_likelihood_from_array(theta)
    splitted_likelihood = log_likelihood_per_obs_from_array(theta)
    total_likelihood_from_splitted = jax.tree.reduce(
        operator.add, jax.tree.map(jnp.sum, splitted_likelihood)
    )

    assert jnp.isclose(total_likelihood_from_splitted, total_likelihood)
    assert jnp.isfinite(log_posterior_from_array(theta))

    batched_log_posterior = jax.vmap(log_posterior_from_array)(
        jnp.stack([theta, theta * jnp.asarray([1.0, 1.1, 1.0, 0.9, 1.0])])
    )

    assert batched_log_posterior.shape == (2,)
