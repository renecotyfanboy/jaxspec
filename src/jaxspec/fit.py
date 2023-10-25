import haiku as hk
import jax.numpy as jnp
import numpyro
import jax
from typing import Callable, Mapping
from abc import ABC, abstractmethod
from jax import random
from jax.tree_util import tree_map
from .analysis.results import ChainResult
from .model.abc import SpectralModel
from .data.instrument import Instrument
from .data.observation import Observation
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Distribution, Poisson


def build_prior(prior):
    """
    Build the prior distribution for the model parameters.

    to_set is supposed to be a numpyro distribution.

    It turns the distribution into a numpyro sample.
    """

    parameters = hk.data_structures.to_haiku_dict(prior)

    for i, (m, n, to_set) in enumerate(hk.data_structures.traverse(prior)):
        if isinstance(to_set, Distribution):
            parameters[m][n] = numpyro.sample(f"{m}_{n}", to_set)

    return parameters


class CountForwardModel(hk.Module):
    """
    A haiku module which allows to build the function that simulates the measured counts
    """

    def __init__(self, model: SpectralModel, instrument: Instrument):
        super().__init__()
        self.model = model
        self.energies = jnp.asarray(instrument.in_energies)
        self.transfer_matrix = jnp.asarray(instrument.transfer_matrix)

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
    """The model to fit to the data."""
    observations: list[Observation]
    """The observations to fit the model to."""
    count_function: hk.Transformed
    """A function enabling the forward modelling of observations with the given instrumental setup."""
    pars: dict

    def __init__(
        self, model: SpectralModel, observations: Observation | list[Observation]
    ):
        self.model = model
        self.observations = (
            [observations] if isinstance(observations, Observation) else observations
        )
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

    def __init__(self, model, observations):
        super().__init__(model, observations)

    def numpyro_model(
        self, prior_distributions: Mapping[str, Mapping[str, Distribution]]
    ) -> Callable:
        """
        Build the numpyro model for the Bayesian fit. It returns a callable which can be used
        to fit the model using numpyro's various samplers.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.
        """

        def model():
            prior_params = build_prior(prior_distributions)

            for i, obs in enumerate(self.observations):
                transformed_model = hk.without_apply_rng(
                    hk.transform(lambda par: CountForwardModel(self.model, obs)(par))
                )

                obs_model = jax.jit(lambda p: transformed_model.apply(None, p))

                with numpyro.plate(f"obs_{i}", len(obs.observed_counts)):
                    numpyro.sample(
                        f"likelihood_obs_{i}",
                        Poisson(obs_model(prior_params)),
                        obs=obs.observed_counts,
                    )

        return model

    def fit(
        self,
        prior_distributions: Mapping[str, Mapping[str, Distribution]],
        rng_key: int = 0,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        jit_model: bool = False,
        mcmc_kwargs: dict = {},
    ) -> ChainResult:
        """
        Fit the model to the data using NUTS sampler from numpyro. This is the default sampler in JAXspec.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.
            rng_key: the random key used to initialize the sampler.
            num_chains: the number of chains to run.
            num_warmup: the number of warmup steps.
            num_samples: the number of samples to draw.
            jit_model: whether to jit the model or not.
            mcmc_kwargs: additional arguments to pass to the MCMC sampler. See [`MCMC`][numpyro.infer.mcmc.MCMC] for more details.

        Returns:
            A [`ChainResult`][jaxspec.analysis.results.ChainResult] instance containing the results of the fit.
        """
        # Instantiate Bayesian model
        bayesian_model = self.numpyro_model(prior_distributions)

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        kernel = NUTS(bayesian_model, max_tree_depth=7)
        mcmc = MCMC(kernel, **(chain_kwargs | mcmc_kwargs))

        mcmc.run(random.PRNGKey(rng_key))

        return ChainResult(self.model, self.observations, mcmc, self.model.params)
