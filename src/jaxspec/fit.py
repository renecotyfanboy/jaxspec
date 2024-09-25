import operator
import warnings

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Literal

import arviz as az
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro

from jax import random
from jax.experimental.sparse import BCOO
from jax.random import PRNGKey
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import Distribution, Poisson, TransformedDistribution
from numpyro.infer import AIES, ESS, MCMC, NUTS, Predictive
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import log_density

from .analysis._plot import _plot_poisson_data_with_error
from .analysis.results import FitResult
from .data import ObsConfiguration
from .model.abc import SpectralModel
from .model.background import BackgroundModel
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
    Base class for a Bayesian model. This class contains the necessary methods to build a model, sample from the prior
    and compute the log-likelihood and posterior probability.
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
        Build a Bayesian model for a given spectral model and observations.

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
            prior = PriorDictModel.from_dict(prior_distributions).nested_dict

            def prior_distributions_func():
                return build_prior(prior, expand_shape=(len(self.observation_container),))

        else:
            prior_distributions_func = prior_distributions

        self.prior_distributions_func = prior_distributions_func
        self.init_params = self.prior_samples()

    @cached_property
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

    @cached_property
    def numpyro_model(self) -> Callable:
        """
        Build the numpyro model using the observed data, the prior distributions and the spectral model.
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

    @cached_property
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

    @cached_property
    def log_likelihood_per_obs(self) -> Callable:
        """
        Build the log likelihood function for each bins in each observation.
        """

        @jax.jit
        def log_likelihood_per_obs(constrained_params):
            log_likelihood = numpyro.infer.util.log_likelihood(
                model=self.numpyro_model, posterior_samples=constrained_params
            )

            return jax.tree.map(lambda x: jnp.where(jnp.isnan(x), -jnp.inf, x), log_likelihood)

        return log_likelihood_per_obs

    @cached_property
    def log_likelihood(self) -> Callable:
        """
        Build the total log likelihood function. Takes a dictionary of parameters where the keys are the parameter names
        that can be fetched with the [`parameter_names`][jaxspec.fit.BayesianModel.parameter_names].
        """

        @jax.jit
        def log_likelihood(constrained_params):
            log_likelihood = self.log_likelihood_per_obs(constrained_params)

            return jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, log_likelihood))

        return log_likelihood

    @cached_property
    def log_posterior_prob(self) -> Callable:
        """
        Build the posterior probability. Takes a dictionary of parameters where the keys are the parameter names
        that can be fetched with the [`parameter_names`][jaxspec.fit.BayesianModel.parameter_names].
        """

        # This is required as numpyro.infer.util.log_densities does not check parameter validity by itself
        numpyro.enable_validation()

        @jax.jit
        def log_posterior_prob(constrained_params):
            log_posterior_prob, _ = log_density(
                self.numpyro_model, (), dict(observed=True), constrained_params
            )
            return jnp.where(jnp.isnan(log_posterior_prob), -jnp.inf, log_posterior_prob)

        return log_posterior_prob

    @cached_property
    def parameter_names(self) -> list[str]:
        """
        A list of parameter names for the model.
        """
        relations = get_model_relations(self.numpyro_model)
        all_sites = relations["sample_sample"].keys()
        observed_sites = relations["observed"]
        return [site for site in all_sites if site not in observed_sites]

    @cached_property
    def observation_names(self) -> list[str]:
        """
        List of the observations.
        """
        relations = get_model_relations(self.numpyro_model)
        all_sites = relations["sample_sample"].keys()
        observed_sites = relations["observed"]
        return [site for site in all_sites if site in observed_sites]

    def array_to_dict(self, theta):
        """
        Convert an array of parameters to a dictionary of parameters.
        """
        input_params = {}

        for index, key in enumerate(self.parameter_names):
            input_params[key] = theta[index]

        return input_params

    def dict_to_array(self, dict_of_params):
        """
        Convert a dictionary of parameters to an array of parameters.
        """

        theta = jnp.zeros(len(self.parameter_names))

        for index, key in enumerate(self.parameter_names):
            theta = theta.at[index].set(dict_of_params[key])

        return theta

    def prior_samples(self, key: PRNGKey = PRNGKey(0), num_samples: int = 100):
        """
        Get initial parameters for the model by sampling from the prior distribution

        Parameters:
            key: the random key used to initialize the sampler.
            num_samples: the number of samples to draw from the prior.
        """

        @jax.jit
        def prior_sample(key):
            return Predictive(
                self.numpyro_model, return_sites=self.parameter_names, num_samples=num_samples
            )(key, observed=False)

        return prior_sample(key)

    def mock_observations(self, parameters, key: PRNGKey = PRNGKey(0)):
        @jax.jit
        def fakeit(key, parameters):
            return Predictive(
                self.numpyro_model,
                return_sites=self.observation_names,
                posterior_samples=parameters,
            )(key, observed=False)

        return fakeit(key, parameters)

    def prior_predictive_coverage(
        self, key: PRNGKey = PRNGKey(0), num_samples: int = 1000, percentiles: tuple = (16, 84)
    ):
        """
        Check if the prior distribution include the observed data.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior_params = self.prior_samples(key=key_prior, num_samples=num_samples)
        posterior_observations = self.mock_observations(prior_params, key=key_posterior)

        for key, value in self.observation_container.items():
            fig, axs = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(8, 8), height_ratios=[3, 1]
            )

            _plot_poisson_data_with_error(
                axs[0],
                value.out_energies,
                value.folded_counts.values,
                percentiles=percentiles,
            )

            axs[0].stairs(
                np.max(posterior_observations["obs_" + key], axis=0),
                edges=[*list(value.out_energies[0]), value.out_energies[1][-1]],
                baseline=np.min(posterior_observations["obs_" + key], axis=0),
                alpha=0.3,
                fill=True,
                color=(0.15, 0.25, 0.45),
            )

            # rank = np.vstack((posterior_observations["obs_" + key], value.folded_counts.values)).argsort(axis=0)[-1] / (num_samples) * 100
            counts = posterior_observations["obs_" + key]
            observed = value.folded_counts.values

            num_samples = counts.shape[0]

            less_than_obs = (counts < observed).sum(axis=0)
            equal_to_obs = (counts == observed).sum(axis=0)

            rank = (less_than_obs + 0.5 * equal_to_obs) / num_samples * 100

            axs[1].stairs(rank, edges=[*list(value.out_energies[0]), value.out_energies[1][-1]])

            axs[1].plot(
                (value.out_energies.min(), value.out_energies.max()),
                (50, 50),
                color="black",
                linestyle="--",
            )

            axs[1].set_xlabel("Energy (keV)")
            axs[0].set_ylabel("Counts")
            axs[1].set_ylabel("Rank (%)")
            axs[1].set_ylim(0, 100)
            axs[0].set_xlim(value.out_energies.min(), value.out_energies.max())
            axs[0].loglog()
            plt.suptitle(f"Prior Predictive coverage for {key}")
            plt.show()


class BayesianModelFitter(BayesianModel, ABC):
    def build_inference_data(
        self,
        posterior_samples,
        num_chains: int = 1,
        num_predictive_samples: int = 1000,
        key: PRNGKey = PRNGKey(42),
        use_transformed_model: bool = False,
        filter_inference_data: bool = True,
    ) -> az.InferenceData:
        """
        Build an [InferenceData][arviz.InferenceData] object from posterior samples.

        Parameters:
            posterior_samples: the samples from the posterior distribution.
            num_chains: the number of chains used to sample the posterior.
            num_predictive_samples: the number of samples to draw from the prior.
            key: the random key used to initialize the sampler.
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            filter_inference_data: whether to filter the InferenceData to keep only the relevant parameters.
        """

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        keys = random.split(key, 3)

        posterior_predictive = Predictive(numpyro_model, posterior_samples)(keys[0], observed=False)

        prior = Predictive(numpyro_model, num_samples=num_predictive_samples * num_chains)(
            keys[1], observed=False
        )

        log_likelihood = numpyro.infer.log_likelihood(numpyro_model, posterior_samples)

        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(numpyro_model, keys[3]),
            substitute_fn=numpyro.infer.init_to_sample,
        )

        observations = {
            name: site["value"]
            for name, site in numpyro.handlers.trace(seeded_model).get_trace().items()
            if site["type"] == "sample" and site["is_observed"]
        }

        def reshape_first_dimension(arr):
            new_dim = arr.shape[0] // num_chains
            new_shape = (num_chains, new_dim) + arr.shape[1:]
            reshaped_array = arr.reshape(new_shape)

            return reshaped_array

        posterior_samples = {
            key: reshape_first_dimension(value) for key, value in posterior_samples.items()
        }
        prior = {key: value[None, :] for key, value in prior.items()}
        posterior_predictive = {
            key: reshape_first_dimension(value) for key, value in posterior_predictive.items()
        }
        log_likelihood = {
            key: reshape_first_dimension(value) for key, value in log_likelihood.items()
        }

        inference_data = az.from_dict(
            posterior_samples,
            prior=prior,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            observed_data=observations,
        )

        return (
            self.filter_inference_data(inference_data) if filter_inference_data else inference_data
        )

    def filter_inference_data(
        self,
        inference_data: az.InferenceData,
    ) -> az.InferenceData:
        """
        Filter the inference data to keep only the relevant parameters for the observations.

        - Removes predictive parameters from deterministic random variables (e.g. kernel of background GP)
        - Removes parameters build from reparametrised variables (e.g. ending with `"_base"`)
        """

        predictive_parameters = []

        for key, value in self.observation_container.items():
            if self.background_model is not None:
                predictive_parameters.append(f"obs_{key}")
                predictive_parameters.append(f"bkg_{key}")
            else:
                predictive_parameters.append(f"obs_{key}")

        inference_data.posterior_predictive = inference_data.posterior_predictive[
            predictive_parameters
        ]

        parameters = [x for x in inference_data.posterior.keys() if not x.endswith("_base")]
        inference_data.posterior = inference_data.posterior[parameters]
        inference_data.prior = inference_data.prior[parameters]

        return inference_data

    @abstractmethod
    def fit(self, **kwargs) -> FitResult: ...


class MCMCFitter(BayesianModelFitter):
    """
    A class to fit a model to a given set of observation using a Bayesian approach. This class uses samplers
    from numpyro to perform the inference on the model parameters.
    """

    kernel_dict = {
        "nuts": NUTS,
        "aies": AIES,
        "ess": ESS,
    }

    def fit(
        self,
        rng_key: int = 0,
        num_chains: int = len(jax.devices()),
        num_warmup: int = 1000,
        num_samples: int = 1000,
        sampler: Literal["nuts", "aies", "ess"] = "nuts",
        use_transformed_model: bool = True,
        kernel_kwargs: dict = {},
        mcmc_kwargs: dict = {},
    ) -> FitResult:
        """
        Fit the model to the data using a MCMC sampler from numpyro.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_chains: the number of chains to run.
            num_warmup: the number of warmup steps.
            num_samples: the number of samples to draw.
            sampler: the sampler to use. Can be one of "nuts", "aies" or "ess".
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            kernel_kwargs: additional arguments to pass to the kernel. See [`NUTS`][numpyro.infer.mcmc.MCMCKernel] for more details.
            mcmc_kwargs: additional arguments to pass to the MCMC sampler. See [`MCMC`][numpyro.infer.mcmc.MCMC] for more details.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        kernel = self.kernel_dict[sampler](bayesian_model, **kernel_kwargs)

        mcmc_kwargs = chain_kwargs | mcmc_kwargs

        if sampler in ["aies", "ess"] and mcmc_kwargs.get("chain_method", None) != "vectorized":
            mcmc_kwargs["chain_method"] = "vectorized"
            warnings.warn("The chain_method is set to 'vectorized' for AIES and ESS samplers")

        mcmc = MCMC(kernel, **mcmc_kwargs)
        keys = random.split(random.PRNGKey(rng_key), 3)

        mcmc.run(keys[0])

        posterior = mcmc.get_samples()

        inference_data = self.build_inference_data(
            posterior, num_chains=num_chains, use_transformed_model=True
        )

        return FitResult(
            self,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )


class NSFitter(BayesianModelFitter):
    r"""
    A class to fit a model to a given set of observation using the Nested Sampling algorithm. This class uses the
    [`DefaultNestedSampler`][jaxns.DefaultNestedSampler] from [`jaxns`](https://jaxns.readthedocs.io/en/latest/) which
    implements the [Phantom-Powered Nested Sampling](https://arxiv.org/abs/2312.11330) algorithm.

    !!! info
        Ensure large prior volume is covered by the prior distributions to ensure the algorithm yield proper results.

    """

    def fit(
        self,
        rng_key: int = 0,
        num_samples: int = 1000,
        num_live_points: int = 1000,
        plot_diagnostics=False,
        termination_kwargs: dict | None = None,
        verbose=True,
    ) -> FitResult:
        """
        Fit the model to the data using the Phantom-Powered nested sampling algorithm.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_samples: the number of samples to draw.
            num_live_points: the number of live points to use at the start of the NS algorithm.
            plot_diagnostics: whether to plot the diagnostics of the NS algorithm.
            termination_kwargs: additional arguments to pass to the termination criterion of the NS algorithm.
            verbose: whether to print the progress of the NS algorithm.

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
                num_live_points=num_live_points,
            ),
            termination_kwargs=termination_kwargs if termination_kwargs else dict(),
        )

        ns.run(keys[0])

        if plot_diagnostics:
            ns.diagnostics()

        posterior = ns.get_samples(keys[1], num_samples=num_samples)
        inference_data = self.build_inference_data(
            posterior, num_chains=1, use_transformed_model=True
        )

        return FitResult(
            self,
            inference_data,
            self.model.params,
            background_model=self.background_model,
        )
