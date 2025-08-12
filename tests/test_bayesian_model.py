import operator

import jax
import jax.numpy as jnp

from jaxspec.fit import BayesianModel


def test_likelihood(obs_model_prior):
    obsconf, model, prior = obs_model_prior
    bayesian_model = BayesianModel(model, prior, obsconf)

    parameter_array = jnp.asarray([[0.7, 0.2, 2, 3e-4]])
    parameters = bayesian_model.array_to_dict(parameter_array)

    total_likelihood = bayesian_model.log_likelihood(parameters)
    splitted_likelihood = bayesian_model.log_likelihood_per_obs(parameters)
    total_likelihood_from_splitted = jax.tree.reduce(
        operator.add, jax.tree.map(jnp.sum, splitted_likelihood)
    )

    assert jnp.isclose(total_likelihood_from_splitted, total_likelihood)
