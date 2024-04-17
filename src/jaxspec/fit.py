import haiku as hk
import jax.numpy as jnp
import numpyro
import arviz as az
import jax
from typing import Callable, TypeVar
from abc import ABC, abstractmethod
from jax import random
from jax.tree_util import tree_map
from jax.experimental.sparse import BCSR
from .analysis.results import ChainResult
from .model.abc import SpectralModel
from .data import ObsConfiguration
from .model.background import BackgroundModel
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import Distribution, TransformedDistribution
from numpyro.distributions import Poisson
from jax.typing import ArrayLike
from numpyro.infer.reparam import TransformReparam


T = TypeVar("T")


class HaikuDict(dict[str, dict[str, T]]): ...


def build_prior(prior: HaikuDict[Distribution | ArrayLike], expand_shape: tuple = ()):
    parameters = dict(hk.data_structures.to_haiku_dict(prior))

    for i, (m, n, sample) in enumerate(hk.data_structures.traverse(prior)):
        match sample:
            case Distribution():
                parameters[m][n] = jnp.ones(expand_shape) * numpyro.sample(f"{m}_{n}", sample)
                # parameters[m][n] = numpyro.sample(f"{m}_{n}", sample.expand(expand_shape)) build a free parameter for each obs
            case float() | ArrayLike():
                parameters[m][n] = jnp.ones(expand_shape) * sample
            case _:
                raise ValueError(f"Invalid prior type {type(sample)} for parameter {m}_{n} : {sample}")

    return parameters


def build_numpyro_model(
    obs: ObsConfiguration,
    model: SpectralModel,
    background_model: BackgroundModel,
    name: str = "",
    sparse: bool = False,
) -> Callable:
    def numpro_model(prior_params, observed=True):
        # prior_params = build_prior(prior_distributions, name=name)
        transformed_model = hk.without_apply_rng(hk.transform(lambda par: CountForwardModel(model, obs, sparse=sparse)(par)))

        if (getattr(obs, "folded_background", None) is not None) and (background_model is not None):
            bkg_countrate = background_model.numpyro_model(
                obs.out_energies, obs.folded_background.data, name=name + "bkg", observed=observed
            )
        elif (getattr(obs, "folded_background", None) is None) and (background_model is not None):
            raise ValueError("Trying to fit a background model but no background is linked to this observation")

        else:
            bkg_countrate = 0.0

        obs_model = jax.jit(lambda p: transformed_model.apply(None, p))
        countrate = obs_model(prior_params)

        # This is the case where we fit a model to a TOTAL spectrum as defined in OGIP standard
        with numpyro.plate(name + "obs_plate", len(obs.folded_counts)):
            numpyro.sample(
                name + "obs",
                Poisson(countrate + bkg_countrate / obs.folded_backratio.data),
                obs=obs.folded_counts.data if observed else None,
            )

    return numpro_model


class CountForwardModel(hk.Module):
    """
    A haiku module which allows to build the function that simulates the measured counts
    """

    def __init__(self, model: SpectralModel, folding: ObsConfiguration, sparse=False):
        super().__init__()
        self.model = model
        self.energies = jnp.asarray(folding.in_energies)

        if sparse:  # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
            self.transfer_matrix = BCSR.from_scipy_sparse(folding.transfer_matrix.data.to_scipy_sparse().tocsr())  #

        else:
            self.transfer_matrix = jnp.asarray(folding.transfer_matrix.data.todense())

    def __call__(self, parameters):
        """
        Compute the count functions for a given observation.
        """

        expected_counts = self.transfer_matrix @ self.model(parameters, *self.energies)

        return jnp.clip(expected_counts, a_min=1e-6)


class BayesianModelAbstract(ABC):
    """
    Abstract class to fit a model to a given set of observation.
    """

    model: SpectralModel
    """The model to fit to the data."""
    numpyro_model: Callable
    """The numpyro model defining the likelihood."""
    background_model: BackgroundModel
    """The background model."""
    pars: dict

    def __init__(self, model: SpectralModel):
        self.model = model
        self.pars = tree_map(lambda x: jnp.float64(x), self.model.params)

    @property
    @abstractmethod
    def observation_container(self): ...

    def fit(
        self,
        prior_distributions: HaikuDict[Distribution],
        rng_key: int = 0,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        max_tree_depth: int = 10,
        target_accept_prob: float = 0.8,
        dense_mass=False,
        mcmc_kwargs: dict = {},
    ) -> ChainResult:
        """
        Fit the model to the data using NUTS sampler from numpyro. This is the default sampler in jaxspec.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.
            rng_key: the random key used to initialize the sampler.
            num_chains: the number of chains to run.
            num_warmup: the number of warmup steps.
            num_samples: the number of samples to draw.
            max_tree_depth: the recursion depth of NUTS sampler.
            target_accept_prob: the target acceptance probability for the NUTS sampler.
            dense_mass: whether to use a dense mass for the NUTS sampler.
            mcmc_kwargs: additional arguments to pass to the MCMC sampler. See [`MCMC`][numpyro.infer.mcmc.MCMC] for more details.

        Returns:
            A [`ChainResult`][jaxspec.analysis.results.ChainResult] instance containing the results of the fit.
        """

        transform_dict = {}

        for m, n, val in hk.data_structures.traverse(prior_distributions):
            if isinstance(val, TransformedDistribution):
                transform_dict[f"{m}_{n}"] = TransformReparam()

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        bayesian_model = numpyro.handlers.reparam(self.numpyro_model(prior_distributions), config=transform_dict)

        kernel = NUTS(bayesian_model, max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob, dense_mass=dense_mass)

        mcmc = MCMC(kernel, **(chain_kwargs | mcmc_kwargs))

        keys = random.split(random.PRNGKey(rng_key), 3)

        mcmc.run(keys[0])
        posterior_predictive = Predictive(bayesian_model, mcmc.get_samples())(keys[1], observed=False)
        prior = Predictive(bayesian_model, num_samples=num_samples)(keys[2], observed=False)
        inference_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)

        predictive_parameters = []

        for key, value in self.observation_container.items():
            if self.background_model is not None:
                predictive_parameters.append(f"{key}_obs")
                predictive_parameters.append(f"{key}_bkg")
            else:
                predictive_parameters.append(f"{key}_obs")

        inference_data.posterior_predictive = inference_data.posterior_predictive[predictive_parameters]

        parameters = [x for x in inference_data.posterior.keys() if not x.endswith("_base")]
        inference_data.posterior = inference_data.posterior[parameters]
        inference_data.prior = inference_data.prior[parameters]

        return ChainResult(
            self.model,
            self.observation_container,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )


class BayesianModel(BayesianModelAbstract):
    def __init__(
        self,
        model,
        observations: list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel = None,
        sparsify_matrix: bool = False,
    ):
        super().__init__(model)
        self._observations = observations
        self.pars = tree_map(lambda x: jnp.float64(x), self.model.params)
        self.background_model = background_model
        self.sparse = sparsify_matrix

    @property
    def observation_container(self) -> dict[str, ObsConfiguration]:
        """
        The observations used in the fit as a dictionary of observations.
        """

        if isinstance(self._observations, dict):
            return self._observations

        elif isinstance(self._observations, list):
            return {f"Data_{i}": obs for i, obs in enumerate(self._observations)}

        elif isinstance(self._observations, ObsConfiguration):
            return {"Data": self._observations}

        else:
            raise ValueError(f"Invalid type for observations : {type(self._observations)}")

    def numpyro_model(self, prior_distributions: HaikuDict[Distribution]) -> Callable:
        def model(observed=True):
            prior_params = build_prior(prior_distributions, expand_shape=(len(self.observation_container),))

            for i, (key, observation) in enumerate(self.observation_container.items()):
                params = tree_map(lambda x: x[i], prior_params)

                obs_model = build_numpyro_model(
                    observation, self.model, self.background_model, name=key + "_", sparse=self.sparse
                )

                obs_model(params, observed=observed)

        return model
