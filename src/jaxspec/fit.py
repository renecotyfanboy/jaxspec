import haiku as hk
import jax.numpy as jnp
import numpyro
import arviz as az
import jax
from typing import Callable, TypeVar
from abc import ABC, abstractmethod
from jax import random
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.experimental.sparse import BCSR
from .analysis.results import FitResult
from .model.abc import SpectralModel
from .data import ObsConfiguration
from .model.background import BackgroundModel
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import Distribution, TransformedDistribution
from numpyro.distributions import Poisson
from jax.typing import ArrayLike
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import initialize_model
from jax.random import PRNGKey
import jaxopt


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
                obs.out_energies, obs.folded_background.data, name="bkg_" + name, observed=observed
            )
        elif (getattr(obs, "folded_background", None) is None) and (background_model is not None):
            raise ValueError("Trying to fit a background model but no background is linked to this observation")

        else:
            bkg_countrate = 0.0

        obs_model = jax.jit(lambda p: transformed_model.apply(None, p))
        countrate = obs_model(prior_params)

        # This is the case where we fit a model to a TOTAL spectrum as defined in OGIP standard
        with numpyro.plate("obs_plate_" + name, len(obs.folded_counts)):
            numpyro.sample(
                "obs_" + name,
                Poisson(countrate + bkg_countrate / obs.folded_backratio.data),
                obs=obs.folded_counts.data if observed else None,
            )

    return numpro_model


def filter_inference_data(inference_data, observation_container, background_model=None) -> az.InferenceData:
    predictive_parameters = []

    for key, value in observation_container.items():
        if background_model is not None:
            predictive_parameters.append(f"obs_{key}")
            predictive_parameters.append(f"bkg_{key}")
        else:
            predictive_parameters.append(f"obs_{key}")

    inference_data.posterior_predictive = inference_data.posterior_predictive[predictive_parameters]

    parameters = [x for x in inference_data.posterior.keys() if not x.endswith("_base")]
    inference_data.posterior = inference_data.posterior[parameters]
    inference_data.prior = inference_data.prior[parameters]

    return inference_data


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

        expected_counts = self.transfer_matrix @ self.model.photon_flux(parameters, *self.energies)

        return jnp.clip(expected_counts, a_min=1e-6)


class ModelFitter(ABC):
    """
    Abstract class to fit a model to a given set of observation.
    """

    def __init__(
        self,
        model: SpectralModel,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel = None,
        sparsify_matrix: bool = False,
    ):
        """
        Initialize the fitter.

        Parameters:
            model: the spectral model to fit.
            observations: the observations to fit the model to.
            background_model: the background model to fit.
            sparsify_matrix: whether to sparsify the transfer matrix.
        """
        self.model = model
        self._observations = observations
        self.background_model = background_model
        self.pars = tree_map(lambda x: jnp.float64(x), self.model.params)
        self.sparse = sparsify_matrix

    @property
    def _observation_container(self) -> dict[str, ObsConfiguration]:
        """
        The observations used in the fit as a dictionary of observations.
        """

        if isinstance(self._observations, dict):
            return self._observations

        elif isinstance(self._observations, list):
            return {f"data_{i}": obs for i, obs in enumerate(self._observations)}

        elif isinstance(self._observations, ObsConfiguration):
            return {"data": self._observations}

        else:
            raise ValueError(f"Invalid type for observations : {type(self._observations)}")

    def numpyro_model(self, prior_distributions: HaikuDict[Distribution]) -> Callable:
        """
        Build the numpyro model using the observed data, the prior distributions and the spectral model.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.

        Returns:
            A model function that can be used with numpyro.
        """

        def model(observed=True):
            prior_params = build_prior(prior_distributions, expand_shape=(len(self._observation_container),))

            for i, (key, observation) in enumerate(self._observation_container.items()):
                params = tree_map(lambda x: x[i], prior_params)

                obs_model = build_numpyro_model(observation, self.model, self.background_model, name=key, sparse=self.sparse)
                obs_model(params, observed=observed)

        return model

    @abstractmethod
    def fit(self, prior_distributions: HaikuDict[Distribution], **kwargs) -> FitResult: ...


class BayesianFitter(ModelFitter):
    """
    A class to fit a model to a given set of observation using a Bayesian approach. This class uses the NUTS sampler
    from numpyro to perform the inference on the model parameters.
    """

    def fit(
        self,
        prior_distributions: HaikuDict[Distribution],
        rng_key: int = 0,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        max_tree_depth: int = 10,
        target_accept_prob: float = 0.8,
        dense_mass: bool = False,
        mcmc_kwargs: dict = {},
    ) -> FitResult:
        """
        Fit the model to the data using NUTS sampler from numpyro.

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
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
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

        inference_data = filter_inference_data(inference_data, self._observation_container, self.background_model)

        return FitResult(
            self.model,
            self._observation_container,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )


class MinimizationFitter(ModelFitter):
    """
    A class to fit a model to a given set of observation using a minimization algorithm. This class uses the L-BFGS
    algorithm from jaxopt to perform the minimization on the model parameters. The uncertainties are computed using the
    Hessian of the log-likelihood, assuming that it is a multivariate Gaussian in the unbounded space defined by
    numpyro.
    """

    def fit(
        self,
        prior_distributions: HaikuDict[Distribution],
        rng_key: int = 0,
        num_iter_max: int = 10_000,
        num_samples: int = 1_000,
    ) -> FitResult:
        """
        Fit the model to the data using L-BFGS algorithm.

        Parameters:
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters.
            rng_key: the random key used to initialize the sampler.
            num_iter_max: the maximum number of iteration in the minimization algorithm.
            num_samples: the number of sample to draw from the best-fit covariance.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = self.numpyro_model(prior_distributions)

        param_info, potential_fn, postprocess_fn, *_ = initialize_model(
            PRNGKey(0),
            bayesian_model,
            model_args=tuple(),
            dynamic_args=True,  # <- this is important!
        )

        # get negative log-density from the potential function
        @jax.jit
        def nll_fn(position):
            func = potential_fn()
            return func(position)

        solver = jaxopt.LBFGS(fun=nll_fn, maxiter=10_000)
        params, state = solver.run(param_info.z)
        keys = random.split(random.PRNGKey(rng_key), 3)

        value_flat, unflatten_fun = ravel_pytree(params)
        covariance = jnp.linalg.inv(jax.hessian(lambda p: nll_fn(unflatten_fun(p)))(value_flat))

        samples_flat = jax.random.multivariate_normal(keys[0], value_flat, covariance, shape=(num_samples,))
        samples = jax.vmap(unflatten_fun)(samples_flat.block_until_ready())
        posterior_samples = postprocess_fn()(samples)

        posterior_predictive = Predictive(bayesian_model, posterior_samples)(keys[1], observed=False)
        prior = Predictive(bayesian_model, num_samples=num_samples)(keys[2], observed=False)
        log_likelihood = numpyro.infer.log_likelihood(bayesian_model, posterior_samples)

        def sanitize_chain(chain):
            return tree_map(lambda x: x[None, ...], chain)

        inference_data = az.from_dict(
            sanitize_chain(posterior_samples),
            prior=sanitize_chain(prior),
            posterior_predictive=sanitize_chain(posterior_predictive),
            log_likelihood=sanitize_chain(log_likelihood),
        )

        inference_data = filter_inference_data(inference_data, self._observation_container, self.background_model)

        return FitResult(
            self.model,
            self._observation_container,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )
