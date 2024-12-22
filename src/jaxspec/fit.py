import operator
import warnings

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Literal

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro

from jax import random
from jax.random import PRNGKey
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import Poisson, TransformedDistribution
from numpyro.infer import AIES, ESS, MCMC, NUTS, Predictive
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import log_density

from ._fit._build_model import build_prior, forward_model
from .analysis._plot import (
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
)
from .analysis.results import FitResult
from .data import ObsConfiguration
from .model.abc import SpectralModel
from .model.background import BackgroundModel
from .util.typing import PriorDictType


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
        self.sparse = sparsify_matrix

        if not callable(prior_distributions):
            # Validate the entry with pydantic
            # prior = PriorDictModel.from_dict(prior_distributions).

            def prior_distributions_func():
                return build_prior(
                    prior_distributions, expand_shape=(len(self.observation_container),)
                )

        else:
            prior_distributions_func = prior_distributions

        self.prior_distributions_func = prior_distributions_func
        self.init_params = self.prior_samples()

        # Check the priors are suited for the observations
        split_parameters = [
            (param, shape[-1])
            for param, shape in jax.tree.map(lambda x: x.shape, self.init_params).items()
            if (len(shape) > 1)
            and not param.startswith("_")
            and not param.startswith("bkg")  # hardcoded for subtracted background
        ]

        for parameter, proposed_number_of_obs in split_parameters:
            if proposed_number_of_obs != len(self.observation_container):
                raise ValueError(
                    f"Invalid splitting in the prior distribution. "
                    f"Expected {len(self.observation_container)} but got {proposed_number_of_obs} for {parameter}"
                )

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

        def numpyro_model(observed=True):
            # Instantiate and register the parameters of the spectral model and the background
            prior_params = self.prior_distributions_func()

            # Iterate over all the observations in our container and build a single numpyro model for each observation
            for i, (name, observation) in enumerate(self.observation_container.items()):
                # Check that we can indeed fit a background
                if (getattr(observation, "folded_background", None) is not None) and (
                    self.background_model is not None
                ):
                    # This call should register the parameter and observation of our background model
                    bkg_countrate = self.background_model.numpyro_model(
                        observation, name=name, observed=observed
                    )

                elif (getattr(observation, "folded_background", None) is None) and (
                    self.background_model is not None
                ):
                    raise ValueError(
                        "Trying to fit a background model but no background is linked to this observation"
                    )

                else:
                    bkg_countrate = 0.0

                # We expect that prior_params contains an array of parameters for each observation
                # They can be identical or different for each observation
                params = jax.tree.map(lambda x: x[i], prior_params)

                # Forward model the observation and get the associated countrate
                obs_model = jax.jit(
                    lambda par: forward_model(self.model, par, observation, sparse=self.sparse)
                )
                obs_countrate = obs_model(params)

                # Register the observation as an observed site
                with numpyro.plate("obs_plate_" + name, len(observation.folded_counts)):
                    numpyro.sample(
                        "obs_" + name,
                        Poisson(
                            obs_countrate + bkg_countrate
                        ),  # / observation.folded_backratio.data
                        obs=observation.folded_counts.data if observed else None,
                    )

        return numpyro_model

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
        self,
        key: PRNGKey = PRNGKey(0),
        num_samples: int = 1000,
    ):
        """
        Check if the prior distribution include the observed data.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior_params = self.prior_samples(key=key_prior, num_samples=num_samples)
        posterior_observations = self.mock_observations(prior_params, key=key_posterior)

        for key, value in self.observation_container.items():
            fig, ax = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(5, 6), height_ratios=[3, 1]
            )

            y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                value.folded_counts.values, 1.0, "ct"
            )

            true_data_plot = _plot_poisson_data_with_error(
                ax[0],
                value.out_energies,
                y_observed.value,
                y_observed_low.value,
                y_observed_high.value,
                alpha=0.7,
            )

            prior_plot = _plot_binned_samples_with_error(
                ax[0], value.out_energies, posterior_observations["obs_" + key], n_sigmas=3
            )

            # rank = np.vstack((posterior_observations["obs_" + key], value.folded_counts.values)).argsort(axis=0)[-1] / (num_samples) * 100
            counts = posterior_observations["obs_" + key]
            observed = value.folded_counts.values

            num_samples = counts.shape[0]

            less_than_obs = (counts < observed).sum(axis=0)
            equal_to_obs = (counts == observed).sum(axis=0)

            rank = (less_than_obs + 0.5 * equal_to_obs) / num_samples * 100

            ax[1].stairs(rank, edges=[*list(value.out_energies[0]), value.out_energies[1][-1]])

            ax[1].plot(
                (value.out_energies.min(), value.out_energies.max()),
                (50, 50),
                color="black",
                linestyle="--",
            )

            ax[1].set_xlabel("Energy (keV)")
            ax[0].set_ylabel("Counts")
            ax[1].set_ylabel("Rank (%)")
            ax[1].set_ylim(0, 100)
            ax[0].set_xlim(value.out_energies.min(), value.out_energies.max())
            ax[0].loglog()
            ax[0].legend(loc="upper right")
            plt.suptitle(f"Prior Predictive coverage for {key}")
            plt.tight_layout()
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

        parameters = [
            x
            for x in inference_data.posterior.keys()
            if not x.endswith("_base") or x.startswith("_")
        ]
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
                verbose=verbose,
                difficult_model=True,
                max_samples=1e5,
                parameter_estimation=True,
                gradient_guided=True,
                devices=jax.devices(),
                # init_efficiency_threshold=0.01,
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
            background_model=self.background_model,
        )
