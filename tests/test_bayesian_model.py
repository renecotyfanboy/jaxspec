import operator

import jax
import jax.numpy as jnp

from jaxspec.fit import BayesianModel

'''
def test_model_building(obs_model_prior):
    """
    Check that all parameters are built correctly within the numpyro model.
    """
    obs, spectral_model, prior_distributions = obs_model_prior

    def numpyro_model():
        params = build_prior(prior_distributions, expand_shape=())
        lower_model = build_numpyro_model_for_single_obs(obs[0], spectral_model, None)

        lower_model(params)

    relations = get_model_relations(numpyro_model)

    assert {
        key for key in relations["sample_param"].keys() if key not in relations["observed"]
    } == set(prior_distributions.keys())
'''


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
