import haiku as hk
import jax.numpy as jnp
import arviz as az
import numpyro
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

    def model_observation(self, parameters, observation: Observation):
        """
        Compute the count functions for a given observation.
        """

        energies = jnp.asarray(observation.energies, dtype=jnp.float64)
        transfer_matrix = jnp.asarray(observation.transfer_matrix, dtype=jnp.float64)

        return jnp.clip(transfer_matrix @ jnp.trapz(self.model(parameters, energies), x=energies, axis=0), a_min=1e-6)

    def log_likelihood_observation(self, parameters, observation):
        """
        Compute the log likelihood for a given observation.
        """

        model = self.model_observation(parameters, observation)
        observed = observation.observed_counts

        return -Poisson(model).log_prob(observed).sum()

    def log_likelihood(self, parameters):
        """
        Compute the joint likelihood for all the observations.
        """

        return sum(tree_map(lambda x: self.log_likelihood_observation(parameters, x), self.observation))

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract method to fit the model to the data.
        """
        pass


class FrequentistModel(ForwardModelFit):
    """
    Class to fit a model to a given set of observation using a frequentist approach.
    """

    def __init__(self, model, observation):
        super().__init__(model, observation)

    def fit(self):
        pass


class BayesianModel(ForwardModelFit):
    """
    Class to fit a model to a given set of observation using a Bayesian approach.
    """

    def __init__(self, model, observation):
        super().__init__(model, observation)

    def build_prior(self, prior):
        """
        Build the prior distribution for the model parameters.
        """

        parameters = hk.data_structures.to_haiku_dict(prior)

        for i, (m, n, to_set) in enumerate(hk.data_structures.traverse(prior)):

            if isinstance(to_set, Distribution):
                parameters[m][n] = numpyro.sample(f'{m}_{n}', to_set)

        return parameters

    def fit(self,
            prior_params,
            rng_key: int = 0,
            num_chains: int = 4,
            num_warmup: int = 1000,
            num_samples: int = 1000,
            likelihood: Distribution = Poisson,
            kernel: MCMCKernel = NUTS,
            kernel_kwargs: dict = {},
            return_inference_data: bool = True):

        def model():

            parameters = self.build_prior(prior_params)

            for i, obs in enumerate(self.observation):

                numpyro.sample(f'likelihood_obs_{i}',
                               likelihood(self.model_observation(parameters, obs)),
                               obs=obs.observed_counts)

        mcmc_kwargs = {
            'num_warmup': num_warmup,
            'num_samples': num_samples,
            'num_chains': num_chains
        }

        kernel = kernel(model, **kernel_kwargs)
        mcmc = MCMC(kernel, **mcmc_kwargs)

        mcmc.run(random.PRNGKey(rng_key))

        if return_inference_data:

            return az.from_numpyro(posterior=mcmc)

        else:

            return mcmc.get_samples()
