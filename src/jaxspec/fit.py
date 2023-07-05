import haiku as hk
import jax.numpy as jnp
import arviz as az
import numpyro
import jax
import numpyro.distributions as dist
from abc import ABC, abstractmethod
from jax import random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from .model.abc import SpectralModel
from .data.observation import Observation
from numpyro.infer import MCMC, NUTS
from numpyro.infer.mcmc import MCMCKernel
from numpyro.distributions import Distribution, Poisson
from typing import Union


def build_prior(prior):
    """
    Build the prior distribution for the model parameters.
    """

    parameters = hk.data_structures.to_haiku_dict(prior)

    for i, (m, n, to_set) in enumerate(hk.data_structures.traverse(prior)):

        if isinstance(to_set, Distribution):
            parameters[m][n] = numpyro.sample(f'{m}_{n}', to_set)

    return parameters


class CountForwardModel(hk.Module):
    """
    A haiku module which allows to build the function that simulates the measured counts
    """

    def __init__(self, model: SpectralModel, observation: Observation):
        super().__init__()
        self.model = model
        self.energies = jnp.asarray(observation.in_energies)
        self.transfer_matrix = jnp.asarray(observation.transfer_matrix)

    def __call__(self, parameters):
        """
        Compute the count functions for a given observation.
        """

        expected_counts = self.transfer_matrix @ self.model(parameters, *self.energies)

        return jnp.clip(expected_counts, a_min=1e-6)


class ForwardModelFit(ABC):
    """
    Abstract class to fit a model to a given set of observation.
    """

    model: SpectralModel
    observation: Union[Observation, list[Observation]]
    count_function: hk.Transformed
    pars: dict

    def __init__(self, model: SpectralModel, observation: Union[Observation, list[Observation]]):

        self.model = model
        self.observation = [observation] if isinstance(observation, Observation) else observation
        self.pars = tree_map(lambda x: jnp.float64(x), self.model.params)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract method to fit the model to the data.
        """
        pass


class BayesianModel(ForwardModelFit):
    """
    Class to fit a model to a given set of observation using a Bayesian approach.
    """

    samples: dict = {}

    def __init__(self, model, observation):
        super().__init__(model, observation)

    def numpyro_model(self, prior_distributions):

        def model():

            prior_params = build_prior(prior_distributions)

            for i, obs in enumerate(self.observation):

                transformed_model = hk.without_apply_rng(
                    hk.transform(lambda par: CountForwardModel(self.model, obs)(par))
                )

                obs_model = jax.jit(lambda p: transformed_model.apply(None, p))

                with numpyro.plate(f'obs_{i}', len(obs.observed_counts)):

                    numpyro.sample(
                        f'likelihood_obs_{i}',
                        Poisson(obs_model(prior_params)),
                        obs=obs.observed_counts
                    )

            return prior_params

        return model

    def fit(self,
            prior_distributions,
            rng_key: int = 0,
            num_chains: int = 4,
            num_warmup: int = 1000,
            num_samples: int = 1000,
            jit_model: bool = False,
            mcmc_kwargs: dict = {},
            return_inference_data: bool = True):

        # Instantiate Bayesian model
        bayesian_model = self.numpyro_model(prior_distributions)

        chain_kwargs = {
            'num_warmup': num_warmup,
            'num_samples': num_samples,
            'num_chains': num_chains
        }

        kernel = NUTS(bayesian_model, max_tree_depth=7)
        mcmc = MCMC(kernel, **(chain_kwargs | mcmc_kwargs))

        mcmc.run(random.PRNGKey(rng_key))

        self.samples = mcmc.get_samples()

        if return_inference_data:

            return az.from_numpyro(posterior=mcmc)

        else:

            return self.samples
