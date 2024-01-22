import haiku as hk
import jax.numpy as jnp
import numpyro
import arviz as az
import jax
from typing import Callable, Mapping, List
from abc import ABC, abstractmethod
from jax import random
from jax.tree_util import tree_map
from .analysis.results import ChainResult
from .model.abc import SpectralModel
from .data import FoldingMatrix
from .model.background import BackgroundModel
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import Distribution
from numpyro.distributions import Poisson


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

    def __init__(self, model: SpectralModel, folding: FoldingMatrix):
        super().__init__()
        self.model = model
        self.energies = jnp.asarray(folding.in_energies)
        self.transfer_matrix = jnp.asarray(folding.transfer_matrix.data)

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
    observations: list[FoldingMatrix]
    """The observations to fit the model to."""
    count_function: hk.Transformed
    """A function enabling the forward modelling of observations with the given instrumental setup."""
    pars: dict

    def __init__(self, model: SpectralModel, observations: FoldingMatrix | list[FoldingMatrix]):
        self.model = model
        self.observations = [observations] if isinstance(observations, FoldingMatrix) else observations
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

    def __init__(self, model, observations, background_model: BackgroundModel = None):
        super().__init__(model, observations)
        self.background_model = background_model

    def numpyro_model(self, prior_distributions: Mapping[str, Mapping[str, Distribution]]) -> Callable:
        """
        Build the numpyro model for the Bayesian fit. It returns a callable which can be used
        to fit the model using numpyro's various samplers.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.
        """

        def model(observed=True):
            prior_params = build_prior(prior_distributions)

            for i, obs in enumerate(self.observations):
                transformed_model = hk.without_apply_rng(hk.transform(lambda par: CountForwardModel(self.model, obs)(par)))

                if (getattr(obs, "folded_background", None) is not None) and (self.background_model is not None):
                    # TODO : Raise warning when setting a background model but there is no background spectra loaded
                    bkg_countrate = self.background_model.numpyro_model(
                        obs.out_energies, obs.folded_background.data, name=f"bkg_{i}", observed=observed
                    )
                else:
                    bkg_countrate = 0.0

                obs_model = jax.jit(lambda p: transformed_model.apply(None, p))
                countrate = obs_model(prior_params)

                # This is the case where we fit a model to a TOTAL spectrum as defined in OGIP standard
                with numpyro.plate(f"obs_{i}_plate", len(obs.folded_counts)):
                    numpyro.sample(
                        f"obs_{i}",
                        Poisson(countrate + bkg_countrate),
                        obs=obs.folded_counts.data if observed else None,
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
    ) -> List[ChainResult]:
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

        kernel = NUTS(bayesian_model, max_tree_depth=6)
        mcmc = MCMC(kernel, **(chain_kwargs | mcmc_kwargs))

        keys = random.split(random.PRNGKey(rng_key), 3)

        mcmc.run(keys[0])
        posterior_predictive = Predictive(bayesian_model, mcmc.get_samples())(keys[1], observed=False)
        prior = Predictive(bayesian_model, num_samples=num_samples)(keys[2], observed=False)

        # Extract results and divide the inference data into one per observation
        result_list = []

        global_id = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)

        for index, observation in enumerate(self.observations):
            mapping_dict = (
                {f"obs_{index}": "obs", f"bkg_{index}": "bkg"} if self.background_model is not None else {f"obs_{index}": "obs"}
            )
            parameters = [f"obs_{index}", f"bkg_{index}"] if self.background_model is not None else [f"obs_{index}"]
            obs_id = global_id.copy()
            obs_id.posterior_predictive = obs_id.posterior_predictive[parameters].rename_vars(mapping_dict)
            result_list.append(
                ChainResult(
                    self.model, observation, obs_id, mcmc.get_samples(), self.model.params, background_model=self.background_model
                )
            )

        return result_list
