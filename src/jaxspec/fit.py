from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import arviz as az
import haiku as hk
import jax
import jax.numpy as jnp
import numpyro
import optimistix as optx

from jax import random
from jax.experimental.sparse import BCOO
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import Distribution, Poisson, TransformedDistribution
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_value
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import constrain_fn
from scipy.stats import Covariance, multivariate_normal

from .analysis.results import FitResult
from .data import ObsConfiguration
from .model.abc import SpectralModel
from .model.background import BackgroundModel
from .util import catchtime
from .util.typing import PriorDictModel, PriorDictType


def build_prior(prior: PriorDictType, expand_shape: tuple = (), prefix=""):
    """
    Transform a dictionary of prior distributions into a dictionary of parameters sampled from the prior.
    Must be used within a numpyro model.
    """
    parameters = dict(hk.data_structures.to_haiku_dict(prior))

    for i, (m, n, sample) in enumerate(hk.data_structures.traverse(prior)):
        if isinstance(sample, Distribution):
            parameters[m][n] = jnp.ones(expand_shape) * numpyro.sample(f"{prefix}{m}_{n}", sample)

        elif isinstance(sample, ArrayLike):
            parameters[m][n] = jnp.ones(expand_shape) * sample

        else:
            raise ValueError(
                f"Invalid prior type {type(sample)} for parameter {prefix}{m}_{n} : {sample}"
            )

    return parameters


def build_numpyro_model_for_single_obs(
    obs: ObsConfiguration,
    model: SpectralModel,
    background_model: BackgroundModel,
    name: str = "",
    sparse: bool = False,
) -> Callable:
    """
    Build a numpyro model for a given observation and spectral model.
    """

    def numpyro_model(prior_params, observed=True):
        # prior_params = build_prior(prior_distributions, name=name)
        transformed_model = hk.without_apply_rng(
            hk.transform(lambda par: CountForwardModel(model, obs, sparse=sparse)(par))
        )

        if (getattr(obs, "folded_background", None) is not None) and (background_model is not None):
            bkg_countrate = background_model.numpyro_model(
                obs, model, name="bkg_" + name, observed=observed
            )
        elif (getattr(obs, "folded_background", None) is None) and (background_model is not None):
            raise ValueError(
                "Trying to fit a background model but no background is linked to this observation"
            )

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

    return numpyro_model


def filter_inference_data(
    inference_data, observation_container, background_model=None
) -> az.InferenceData:
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

        if (
            sparse
        ):  # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
            self.transfer_matrix = BCOO.from_scipy_sparse(
                folding.transfer_matrix.data.to_scipy_sparse().tocsr()
            )

        else:
            self.transfer_matrix = jnp.asarray(folding.transfer_matrix.data.todense())

    def __call__(self, parameters):
        """
        Compute the count functions for a given observation.
        """

        expected_counts = self.transfer_matrix @ self.model.photon_flux(parameters, *self.energies)

        return jnp.clip(expected_counts, a_min=1e-6)


class BayesianModel:
    """
    Class to fit a model to a given set of observation.
    """

    def __init__(
        self,
        model: SpectralModel,
        prior_distributions: PriorDictType | Callable,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel = None,
        sparsify_matrix: bool = False,
    ):
        """
        Initialize the fitter.

        Parameters:
            model: the spectral model to fit.
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters, or a
                callable function that returns parameter samples.
            observations: the observations to fit the model to.
            background_model: the background model to fit.
            sparsify_matrix: whether to sparsify the transfer matrix.
        """
        self.model = model
        self._observations = observations
        self.background_model = background_model
        self.pars = tree_map(lambda x: jnp.float64(x), self.model.params)
        self.sparse = sparsify_matrix

        if not callable(prior_distributions):
            # Validate the entry with pydantic
            prior = PriorDictModel(nested_dict=prior_distributions).nested_dict

            def prior_distributions_func():
                return build_prior(prior, expand_shape=(len(self.observation_container),))

        else:
            prior_distributions_func = prior_distributions

        self.prior_distributions_func = prior_distributions_func

    @property
    def observation_container(self) -> dict[str, ObsConfiguration]:
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

    @property
    def numpyro_model(self) -> Callable:
        """
        Build the numpyro model using the observed data, the prior distributions and the spectral model.

        Returns:
        -------
            A model function that can be used with numpyro.
        """

        def model(observed=True):
            prior_params = self.prior_distributions_func()

            # Iterate over all the observations in our container and build a single numpyro model for each observation
            for i, (key, observation) in enumerate(self.observation_container.items()):
                # We expect that prior_params contains an array of parameters for each observation
                # They can be identical or different for each observation
                params = tree_map(lambda x: x[i], prior_params)

                obs_model = build_numpyro_model_for_single_obs(
                    observation, self.model, self.background_model, name=key, sparse=self.sparse
                )

                obs_model(params, observed=observed)

        return model

    @property
    def transformed_numpyro_model(self) -> Callable:
        transform_dict = {}

        relations = get_model_relations(self.numpyro_model)
        distributions = {
            parameter: getattr(numpyro.distributions, value, None)
            for parameter, value in relations["sample_dist"].items()
        }

        for parameter, distribution in distributions.items():
            if isinstance(distribution, TransformedDistribution):
                transform_dict[parameter] = TransformReparam()

        return numpyro.handlers.reparam(self.numpyro_model, config=transform_dict)


class BayesianModelFitter(BayesianModel, ABC):
    @abstractmethod
    def fit(self, **kwargs) -> FitResult: ...


class NUTSFitter(BayesianModelFitter):
    """
    A class to fit a model to a given set of observation using a Bayesian approach. This class uses the NUTS sampler
    from numpyro to perform the inference on the model parameters.
    """

    def fit(
        self,
        rng_key: int = 0,
        num_chains: int = len(jax.devices()),
        num_warmup: int = 1000,
        num_samples: int = 1000,
        max_tree_depth: int = 10,
        target_accept_prob: float = 0.8,
        dense_mass: bool = False,
        kernel_kwargs: dict = {},
        mcmc_kwargs: dict = {},
    ) -> FitResult:
        """
        Fit the model to the data using NUTS sampler from numpyro.

        Parameters:
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

        bayesian_model = self.transformed_numpyro_model
        # bayesian_model = self.numpyro_model(prior_distributions)

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        kernel = NUTS(
            bayesian_model,
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob,
            dense_mass=dense_mass,
            **kernel_kwargs,
        )

        mcmc = MCMC(kernel, **(chain_kwargs | mcmc_kwargs))
        keys = random.split(random.PRNGKey(rng_key), 3)

        mcmc.run(keys[0])

        posterior_predictive = Predictive(bayesian_model, mcmc.get_samples())(
            keys[1], observed=False
        )

        prior = Predictive(bayesian_model, num_samples=num_samples)(keys[2], observed=False)

        inference_data = az.from_numpyro(
            mcmc, prior=prior, posterior_predictive=posterior_predictive
        )

        inference_data = filter_inference_data(
            inference_data, self.observation_container, self.background_model
        )

        return FitResult(
            self,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )


class MinimizationFitter(BayesianModelFitter):
    """
    A class to fit a model to a given set of observation using a minimization algorithm. This class uses the L-BFGS
    algorithm from jaxopt to perform the minimization on the model parameters. The uncertainties are computed using the
    Hessian of the log-log_likelihood, assuming that it is a multivariate Gaussian in the unbounded space defined by
    numpyro.
    """

    def fit(
        self,
        rng_key: int = 0,
        num_iter_max: int = 100_000,
        num_samples: int = 1_000,
        solver: Literal["bfgs", "levenberg_marquardt"] = "bfgs",
        init_params=None,
        refine_first_guess=True,
    ) -> FitResult:
        """
        Fit the model to the data using L-BFGS algorithm.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_iter_max: the maximum number of iteration in the minimization algorithm.
            num_samples: the number of sample to draw from the best-fit covariance.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = self.numpyro_model
        keys = jax.random.split(PRNGKey(rng_key), 4)

        if init_params is not None:
            # We initialize the parameters by randomly sampling from the prior
            local_keys = jax.random.split(keys[0], 2)

            with numpyro.handlers.seed(rng_seed=local_keys[0]):
                starting_value = self.prior_distributions_func()

            # We update the starting value with the provided init_params
            for m, n, val in hk.data_structures.traverse(init_params):
                if f"{m}_{n}" in starting_value.keys():
                    starting_value[f"{m}_{n}"] = val

            init_params, _ = numpyro.infer.util.find_valid_initial_params(
                local_keys[1], bayesian_model, init_strategy=init_to_value(values=starting_value)
            )

        else:
            init_params, _ = numpyro.infer.util.find_valid_initial_params(keys[0], bayesian_model)

        init_params = init_params[0]

        @jax.jit
        def nll(unconstrained_params, _):
            constrained_params = constrain_fn(
                bayesian_model, tuple(), dict(observed=True), unconstrained_params
            )

            log_likelihood = numpyro.infer.util.log_likelihood(
                model=bayesian_model, posterior_samples=constrained_params
            )

            # We solve a least square problem, this function ensure that the total residual is indeed the nll
            return jax.tree.map(lambda x: jnp.sqrt(-x), log_likelihood)

        """
        if refine_first_guess:
            with catchtime("Refine_first"):
                solution = optx.least_squares(
                    nll,
                    optx.BestSoFarMinimiser(optx.OptaxMinimiser(optax.adam(1e-4), 1e-6, 1e-6)),
                    init_params,
                    max_steps=1000,
                    throw=False
                )
            init_params = solution.value
        """

        if solver == "bfgs":
            solver = optx.BestSoFarMinimiser(optx.BFGS(1e-6, 1e-6))
        elif solver == "levenberg_marquardt":
            solver = optx.BestSoFarLeastSquares(optx.LevenbergMarquardt(1e-6, 1e-6))
        else:
            raise NotImplementedError(f"{solver} is not implemented")

        with catchtime("Minimization"):
            solution = optx.least_squares(
                nll,
                solver,
                init_params,
                max_steps=num_iter_max,
            )

        params = solution.value
        value_flat, unflatten_fun = ravel_pytree(params)

        with catchtime("Compute error"):
            precision = jax.hessian(
                lambda p: jnp.sum(ravel_pytree(nll(unflatten_fun(p), None))[0] ** 2)
            )(value_flat)

            cov = Covariance.from_precision(precision)

            samples_flat = multivariate_normal.rvs(mean=value_flat, cov=cov, size=num_samples)

        samples = jax.vmap(unflatten_fun)(samples_flat)
        posterior_samples = jax.jit(
            jax.vmap(lambda p: constrain_fn(bayesian_model, tuple(), dict(observed=True), p))
        )(samples)

        with catchtime("Posterior"):
            posterior_predictive = Predictive(bayesian_model, posterior_samples)(
                keys[2], observed=False
            )
            prior = Predictive(bayesian_model, num_samples=num_samples)(keys[3], observed=False)
            log_likelihood = numpyro.infer.log_likelihood(bayesian_model, posterior_samples)

        def sanitize_chain(chain):
            """
            reshape the samples so that it is arviz compliant with an extra starting dimension
            """
            return tree_map(lambda x: x[None, ...], chain)

        # We export the observed values to the inference_data
        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(bayesian_model, jax.random.PRNGKey(0)),
            substitute_fn=numpyro.infer.init_to_sample,
        )
        trace = numpyro.handlers.trace(seeded_model).get_trace()
        observations = {
            name: site["value"]
            for name, site in trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

        with catchtime("InferenceData wrapping"):
            inference_data = az.from_dict(
                sanitize_chain(posterior_samples),
                prior=sanitize_chain(prior),
                posterior_predictive=sanitize_chain(posterior_predictive),
                log_likelihood=sanitize_chain(log_likelihood),
                observed_data=observations,
            )

        inference_data = filter_inference_data(
            inference_data, self.observation_container, self.background_model
        )

        return FitResult(
            self,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )


class NestedSamplingFitter(BayesianModelFitter):
    r"""
    A class to fit a model to a given set of observation using the Nested Sampling algorithm. This class uses the
    [`DefaultNestedSampler`][jaxns.DefaultNestedSampler] from [`jaxns`](https://jaxns.readthedocs.io/en/latest/) which
    implements the [Phantom-Powered Nested Sampling](https://arxiv.org/abs/2312.11330) algorithm.
    Add Citation to jaxns
    """

    def fit(
        self,
        rng_key: int = 0,
        num_samples: int = 1000,
        plot_diagnostics=False,
        termination_kwargs: dict | None = None,
        verbose=True,
    ) -> FitResult:
        """
        Fit the model to the data using the Phantom-Powered nested sampling algorithm.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_samples: the number of samples to draw.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = self.transformed_numpyro_model
        keys = random.split(random.PRNGKey(rng_key), 4)

        ns = NestedSampler(
            bayesian_model,
            constructor_kwargs=dict(
                num_parallel_workers=1,
                verbose=verbose,
                difficult_model=True,
                max_samples=1e6,
                parameter_estimation=True,
                num_live_points=1_000,
            ),
            termination_kwargs=termination_kwargs if termination_kwargs else dict(),
        )

        ns.run(keys[0])

        if plot_diagnostics:
            ns.diagnostics()

        posterior_samples = ns.get_samples(keys[1], num_samples=num_samples)
        log_likelihood = numpyro.infer.log_likelihood(bayesian_model, posterior_samples)
        posterior_predictive = Predictive(bayesian_model, posterior_samples)(
            keys[2], observed=False
        )

        prior = Predictive(bayesian_model, num_samples=num_samples)(keys[3], observed=False)

        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(bayesian_model, jax.random.PRNGKey(0)),
            substitute_fn=numpyro.infer.init_to_sample,
        )
        trace = numpyro.handlers.trace(seeded_model).get_trace()
        observations = {
            name: site["value"]
            for name, site in trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

        def sanitize_chain(chain):
            """
            reshape the samples so that it is arviz compliant with an extra starting dimension
            """
            return tree_map(lambda x: x[None, ...], chain)

        inference_data = az.from_dict(
            sanitize_chain(posterior_samples),
            prior=sanitize_chain(prior),
            posterior_predictive=sanitize_chain(posterior_predictive),
            log_likelihood=sanitize_chain(log_likelihood),
            observed_data=observations,
        )

        inference_data = filter_inference_data(
            inference_data, self.observation_container, self.background_model
        )

        return FitResult(
            self,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )
