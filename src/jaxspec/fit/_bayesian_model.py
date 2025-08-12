import operator

from collections.abc import Callable
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro

from flax import nnx
from jax.experimental import mesh_utils
from jax.random import PRNGKey
from jax.sharding import PositionalSharding
from numpyro.distributions import Poisson, TransformedDistribution
from numpyro.infer import Predictive
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import log_density

from ..analysis._plot import (
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
)
from ..data import ObsConfiguration
from ..model.abc import SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel
from ..util.typing import PriorDictType
from ._build_model import build_prior, forward_model


class BayesianModel(nnx.Module):
    """
    Base class for a Bayesian model. This class contains the necessary methods to build a model, sample from the prior
    and compute the log-likelihood and posterior probability.
    """

    settings: dict[str, Any]

    def __init__(
        self,
        model: SpectralModel,
        prior_distributions: PriorDictType | Callable,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel = None,
        instrument_model: InstrumentModel = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
    ):
        """
        Build a Bayesian model for a given spectral model and observations.

        Parameters:
            model: the spectral model to fit.
            prior_distributions: a nested dictionary containing the prior distributions for the model parameters, or a
                callable function that returns parameter samples.
            observations: the observations to fit the model to.
            background_model: the background model to fit.
            instrument_model: the instrument model to fit.
            sparsify_matrix: whether to sparsify the transfer matrix.
        """

        self.spectral_model = model
        self._observations = observations
        self.background_model = background_model
        self.instrument_model = instrument_model
        self.settings = {"sparse": sparsify_matrix}

        if not callable(prior_distributions):

            def prior_distributions_func():
                return build_prior(
                    prior_distributions,
                    expand_shape=(len(self._observation_container),),
                    prefix="mod/~/",
                )

        else:
            prior_distributions_func = prior_distributions

        self.prior_distributions_func = prior_distributions_func

        def numpyro_model(observed=True):
            # Instantiate and register the parameters of the spectral model and the background
            prior_params = self.prior_distributions_func()

            # Iterate over all the observations in our container and build a single numpyro model for each observation
            for i, (name, observation) in enumerate(self._observation_container.items()):
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

                if self.instrument_model is not None:
                    gain, shift = self.instrument_model.get_gain_and_shift_model(name)
                else:
                    gain, shift = None, None

                # Forward model the observation and get the associated countrate
                obs_model = jax.jit(
                    lambda par: forward_model(
                        self.spectral_model,
                        par,
                        observation,
                        sparse=self.settings.get("sparse", False),
                        gain=gain,
                        shift=shift,
                        n_points=n_points,
                    )
                )

                obs_countrate = obs_model(params)

                # Register the observation as an observed site
                with numpyro.plate("obs_plate/~/" + name, len(observation.folded_counts)):
                    numpyro.sample(
                        "obs/~/" + name,
                        Poisson(obs_countrate + bkg_countrate / observation.folded_backratio.data),
                        obs=observation.folded_counts.data if observed else None,
                    )

        self.numpyro_model = numpyro_model
        self._init_params = self.prior_samples()
        # Check the priors are suited for the observations
        split_parameters = [
            (param, shape[-1])
            for param, shape in jax.tree.map(lambda x: x.shape, self._init_params).items()
            if (len(shape) > 1)
            and not param.startswith("_")
            and not param.startswith("bkg")  # hardcoded for subtracted background
            and not param.startswith("ins")
        ]

        for parameter, proposed_number_of_obs in split_parameters:
            if proposed_number_of_obs != len(self._observation_container):
                raise ValueError(
                    f"Invalid splitting in the prior distribution. "
                    f"Expected {len(self._observation_container)} but got {proposed_number_of_obs} for {parameter}"
                )

    @cached_property
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
    def _parameter_names(self) -> list[str]:
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

        for index, key in enumerate(self._parameter_names):
            input_params[key] = theta[index]

        return input_params

    def dict_to_array(self, dict_of_params):
        """
        Convert a dictionary of parameters to an array of parameters.
        """

        theta = jnp.zeros(len(self._parameter_names))

        for index, key in enumerate(self._parameter_names):
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
                self.numpyro_model, return_sites=self._parameter_names, num_samples=num_samples
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
        n_devices = len(jax.local_devices())
        sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,)))

        # Sample from prior and correct if the number of samples is not a multiple of the number of devices
        if num_samples % n_devices != 0:
            num_samples = num_samples + n_devices - (num_samples % n_devices)

        prior_params = self.prior_samples(key=key_prior, num_samples=num_samples)

        # Split the parameters on every device
        sharded_parameters = jax.device_put(prior_params, sharding)
        posterior_observations = self.mock_observations(sharded_parameters, key=key_posterior)

        for key, value in self._observation_container.items():
            fig, ax = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(5, 6), height_ratios=[3, 1]
            )

            legend_plots = []
            legend_labels = []

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
                ax[0], value.out_energies, posterior_observations["obs/~/" + key], n_sigmas=3
            )

            legend_plots.append((true_data_plot,))
            legend_labels.append("Observed")
            legend_plots += prior_plot
            legend_labels.append("Prior Predictive")

            # rank = np.vstack((posterior_observations["obs_" + key], value.folded_counts.values)).argsort(axis=0)[-1] / (num_samples) * 100
            counts = posterior_observations["obs/~/" + key]
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
            ax[0].legend(legend_plots, legend_labels)
            plt.suptitle(f"Prior Predictive coverage for {key}")
            plt.tight_layout()
            plt.show()
