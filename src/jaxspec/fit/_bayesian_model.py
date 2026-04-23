import operator

from collections.abc import Callable
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro

from jax import Array
from jax.random import key
from numpyro.distributions import Poisson, TransformedDistribution
from numpyro.infer import Predictive
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import log_density

from ..analysis._plot import (
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
    _rebin_xbins,
    adaptive_bin_1d,
    rebin_counts,
)
from ..data import ObsConfiguration
from ..model.abc import SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel
from ..util.typing import PriorDictType
from ._forward_model import ForwardModel


class BayesianModel:
    """
    Bayesian spectral model that composes a deterministic forward model with
    numpyro priors and Poisson likelihoods.

    Parameters:
        model: The spectral model to fit.
        prior_distributions: Unified dictionary mapping parameter names to
            numpyro distributions, fixed values,
            :class:`~jaxspec.fit.TiedParameter`, or
            :class:`~jaxspec.fit.PerObs` wrappers.

            Dict keys are routed by prefix:

            - ``"spectrum."`` prefix (e.g. ``"spectrum.powerlaw_1.alpha"``) →
              spectral model.
            - ``"instrument."`` prefix (e.g. ``"instrument.gain.factor"``) →
              instrument model.
            - ``"background."`` prefix (e.g.
              ``"background.powerlaw_1.alpha"``) → background model.
        observations: One or more observation configurations. Accepts a single
            :class:`~jaxspec.data.ObsConfiguration`, a list (auto-named
            ``data_0``, ``data_1``, ...), or a ``{name: obs}`` dict.
        background_model: Optional background model.
        instrument_model: Optional instrument calibration model.
        sparsify_matrix: Whether to use sparse transfer matrices.
        n_points: Number of quadrature points per energy bin.
    """

    def __init__(
        self,
        model: SpectralModel,
        prior_distributions: PriorDictType,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel | None = None,
        instrument_model: InstrumentModel | None = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
    ):
        self.forward_model = ForwardModel(
            model,
            observations,
            background_model=background_model,
            instrument_model=instrument_model,
            sparsify_matrix=sparsify_matrix,
            n_points=n_points,
        )
        self._prior = dict(prior_distributions)
        self._effective_prior = self._build_prior_dict()

    @property
    def spectral_model(self) -> SpectralModel:
        return self.forward_model.spectrum

    @property
    def background_model(self) -> BackgroundModel | None:
        return self.forward_model.background_model

    @property
    def instrument_model(self) -> InstrumentModel | None:
        return self.forward_model.instrument_model

    @property
    def settings(self) -> dict[str, Any]:
        return self.forward_model.settings

    @staticmethod
    def _validate_prior_dict(prior: dict, observation_names: list[str]) -> None:
        from ._parameter import PerObs, TiedParameter

        def _validate_leaf(site_name: str, value) -> None:
            if isinstance(value, PerObs):
                raise TypeError(f"PerObs inside PerObs is not supported (at {site_name!r}).")
            if isinstance(value, TiedParameter):
                raise TypeError(f"TiedParameter inside PerObs is not supported (at {site_name!r}).")
            if isinstance(value, numpyro.distributions.Distribution):
                return
            try:
                jnp.asarray(value)
            except Exception as exc:
                raise TypeError(f"Invalid fixed prior value for {site_name!r}: {value!r}") from exc

        required = set(observation_names)

        for site_name, entry in prior.items():
            if isinstance(entry, TiedParameter):
                continue

            if not isinstance(entry, PerObs):
                if not isinstance(entry, numpyro.distributions.Distribution):
                    try:
                        jnp.asarray(entry)
                    except Exception as exc:
                        raise TypeError(
                            f"Invalid fixed prior value for {site_name!r}: {entry!r}"
                        ) from exc
                continue

            if entry.is_homogeneous:
                _validate_leaf(site_name, entry.value)
                continue

            missing = required - set(entry.value.keys())
            if missing:
                raise ValueError(
                    f"PerObs entry for {site_name!r} is missing observations: {sorted(missing)}"
                )
            for obs_name in observation_names:
                _validate_leaf(f"{site_name}.{obs_name}", entry.value[obs_name])

    def _build_prior_dict(self) -> dict:
        """Return the validated prior dict used to sample/extract parameters.

        Merges the background model's ``default_prior`` (user entries win on
        collision), then validates the resulting dict against the observations.
        """
        prior = self._prior
        fm = self.forward_model
        if fm.background_model is not None:
            defaults = fm.background_model.default_prior(fm.observations)
            prior = {**defaults, **prior}

        self._validate_prior_dict(prior, list(fm.observations.keys()))
        return prior

    def _sample_priors(self) -> tuple[dict, dict | None, dict | None]:
        """Sample the priors for spectrum, instrument, background.

        Returns a ``(source_params, instrument_params, background_params)``
        tuple. ``instrument_params`` / ``background_params`` are ``None`` when
        the corresponding model is absent.
        """
        fm = self.forward_model
        obs_names = list(fm.observations.keys())

        prior = self._effective_prior
        source_params = fm.spectrum.register_priors(prior, obs_names)
        instrument_params = (
            fm.instrument_model.register_priors(prior, obs_names)
            if fm.instrument_model is not None
            else None
        )
        background_params = (
            fm.background_model.register_priors(prior, obs_names)
            if fm.background_model is not None
            else None
        )
        return source_params, instrument_params, background_params

    def numpyro_model(self, observed: bool = True):
        """Build the full numpyro model: source + instrument + background + likelihoods.

        Parameters:
            observed: If ``True``, condition on the observed data (fitting mode).
                If ``False``, sample from the prior predictive distribution.
        """
        fm = self.forward_model
        source_params, instrument_params, background_params = self._sample_priors()

        per_obs = fm.expected_counts(
            source_params=source_params,
            instrument_params=instrument_params,
            background_params=background_params,
        )

        # Background observation sites (stochastic) or deterministic record
        if fm.background_model is not None:
            for name, obs in fm.observations.items():
                if getattr(obs, "folded_background", None) is None:
                    raise ValueError(
                        "Trying to fit a background model but no background is "
                        "linked to this observation"
                    )
                bkg_rate = per_obs[name]["background_rate"]
                if fm.background_model.is_stochastic:
                    with numpyro.plate(
                        f"observed_background_plate.{name}", len(obs.folded_background)
                    ):
                        numpyro.sample(
                            f"observed_background.{name}",
                            Poisson(bkg_rate),
                            obs=obs.folded_background.data if observed else None,
                        )
                else:
                    numpyro.deterministic(f"observed_background.{name}", bkg_rate)

        # Source observation sites
        for name, obs in fm.observations.items():
            with numpyro.plate(f"observed_plate.{name}", len(obs.folded_counts)):
                numpyro.sample(
                    f"observed.{name}",
                    Poisson(per_obs[name]["total"]),
                    obs=obs.folded_counts.data if observed else None,
                )

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
        List of parameter names for the model.
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

        for index, parameter_key in enumerate(self.parameter_names):
            input_params[parameter_key] = theta[index]

        return input_params

    def dict_to_array(self, dict_of_params):
        """
        Convert a dictionary of parameters to an array of parameters.
        """

        theta = jnp.zeros(len(self.parameter_names))
        for index, parameter_key in enumerate(self.parameter_names):
            theta = theta.at[index].set(dict_of_params[parameter_key])
        return theta

    def prior_samples(self, key: Array = key(0), num_samples: int = 100):
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

    def mock_observations(self, parameters, key: Array = key(0)):
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
        key: Array = key(0),
        num_samples: int = 1000,
        min_counts: int | None = None,
        grouping: int | None = None,
    ):
        """
        Check if the prior distribution includes the observed data.

        Parameters:
            key: Random key for sampling.
            num_samples: Number of prior predictive samples.
            min_counts: Minimum number of observed counts per grouped bin.
                Adjacent bins are merged until the threshold is reached.
                Mutually exclusive with *grouping*.
            grouping: Number of consecutive bins to merge into each group.
                Mutually exclusive with *min_counts*.
        """
        if min_counts is not None and grouping is not None:
            raise ValueError("min_counts and grouping are mutually exclusive")

        key_prior, key_posterior = jax.random.split(key, 2)
        prior_params = self.prior_samples(key=key_prior, num_samples=num_samples)
        posterior_observations = self.mock_observations(prior_params, key=key_posterior)

        for key, value in self.forward_model.observations.items():
            fig, ax = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(5, 6), height_ratios=[3, 1]
            )

            legend_plots = []
            legend_labels = []

            observed = value.folded_counts.values
            counts = np.asarray(posterior_observations[f"observed.{key}"])
            out_energies = value.out_energies

            if min_counts is not None:
                bin_ids = adaptive_bin_1d(observed, min_counts)
            elif grouping is not None:
                n_bins = len(observed)
                bin_ids = np.arange(n_bins) // grouping
            else:
                bin_ids = None

            if bin_ids is not None:
                observed = rebin_counts(observed, bin_ids)
                counts = rebin_counts(counts, bin_ids)
                out_energies = _rebin_xbins(out_energies, bin_ids)

            y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                observed, 1.0, "ct"
            )

            true_data_plot = _plot_poisson_data_with_error(
                ax[0],
                out_energies,
                y_observed.value,
                y_observed_low.value,
                y_observed_high.value,
                alpha=0.7,
            )

            prior_plot = _plot_binned_samples_with_error(ax[0], out_energies, counts, n_sigmas=3)

            legend_plots.append((true_data_plot,))
            legend_labels.append("Observed")
            legend_plots += prior_plot
            legend_labels.append("Prior Predictive")

            num_samples = counts.shape[0]

            less_than_obs = (counts < observed).sum(axis=0)
            equal_to_obs = (counts == observed).sum(axis=0)

            rank = (less_than_obs + 0.5 * equal_to_obs) / num_samples * 100

            ax[1].stairs(rank, edges=[*list(out_energies[0]), out_energies[1][-1]])

            ax[1].plot(
                (out_energies.min(), out_energies.max()),
                (50, 50),
                color="black",
                linestyle="--",
            )

            ax[1].set_xlabel("Energy (keV)")
            ax[0].set_ylabel("Counts")
            ax[1].set_ylabel("Rank (%)")
            ax[1].set_ylim(0, 100)
            ax[0].set_xlim(out_energies.min(), out_energies.max())
            ax[0].loglog()
            ax[0].legend(legend_plots, legend_labels)
            plt.suptitle(f"Prior Predictive coverage for {key}")
            plt.tight_layout()
            plt.show()
