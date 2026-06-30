from __future__ import annotations

import operator

from collections.abc import Callable
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist

from flax import nnx
from jax import Array
from jax.random import key as rng_key
from jaxtyping import ArrayLike
from numpyro.distributions import Poisson, TransformedDistribution
from numpyro.infer import Predictive
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import log_density

from ..analysis._plot import (
    _compute_bin_ids,
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
    _rebin_xbins,
    rebin_counts,
)
from ..data import ObsConfiguration
from ..model.abc import ModelComponent, SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel
from ._forward_model import ForwardModel
from ._parameter import TiedParameter
from ._prior_resolution import (
    _KNOWN_PREFIXES,
    _enumerate_leaves,
    _prefix_to_obs_names,
    _resolve_targets,
    _unmatched_key_message,
    parse_prior_key,
    sample_prior,
)


class BayesianModel:
    """
    Bayesian spectral model that composes a deterministic forward model with
    numpyro priors and Poisson likelihoods.

    Parameters:
        model: The spectral model to fit (cloned per observation inside the
            internal :class:`~jaxspec.fit._forward_model.ForwardModel`). A single
            bare component (e.g. ``Powerlaw()``) is accepted and auto-wrapped via
            :meth:`~jaxspec.model.abc.SpectralModel.from_component`.
        prior: Either a unified prior dict using the ``[obs]`` / ``[*]``
            scoping syntax (see module docs), or a factory callable
            ``() -> ((leaf_path, shape) -> Distribution)``. The factory form
            runs inside the numpyro trace, letting it sample shared /
            hierarchical params before returning the leaf callable.
        observations: One or more observation configurations.
        background_model: ``None``, a singleton ``BackgroundModel``, or a
            ``{obs_name: BackgroundModel | None}`` dict.
        instrument_model: ``None``, or a ``{obs_name: InstrumentModel | None}``
            dict. ``None`` entries (and observations omitted from the dict)
            apply the identity fold (no instrument calibration).
        sparsify_matrix: Whether to use sparse transfer matrices.
        n_points: Number of quadrature points per energy bin.
    """

    def __init__(
        self,
        model: SpectralModel | ModelComponent,
        prior: dict | Callable,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel | dict[str, BackgroundModel | None] | None = None,
        instrument_model: dict[str, InstrumentModel | None] | None = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
        energy_grid: ArrayLike | None = None,
    ):
        self.forward_model = ForwardModel(
            model,
            observations,
            background_model=background_model,
            instrument_model=instrument_model,
            sparsify_matrix=sparsify_matrix,
            n_points=n_points,
            energy_grid=energy_grid,
        )

        self._user_prior = prior
        self._effective_prior = self._build_prior_dict()
        self._validate_prior_dict()

        # Tell the ForwardModel whether the eval-once-then-fold fast path is
        # safe for this fit. Only safe when an explicit energy_grid is set AND
        # every spectral prior entry is shared (no [obs] / [*] scopes), since
        # any per-obs spectral param means each replica produces a different
        # flux on the grid. Callable priors are conservatively treated as
        # non-shared (we can't statically inspect them).
        self.forward_model.settings["spectrum_shared"] = self._spectrum_is_shared()

    @property
    def spectral_model(self) -> SpectralModel:
        """A representative spectral model replica (PN's, or whichever obs is first).

        All per-obs replicas share the same structure; this is provided for
        callers (e.g. :class:`~jaxspec.analysis.results.FitResult`) that want
        a single ``SpectralModel`` to introspect topology or compute fluxes.
        Per-obs *parameter values* live on the bound replicas after sampling.
        """
        return next(iter(self.forward_model.spectrum.values()))

    @property
    def spectrum(self) -> dict[str, SpectralModel]:
        """Per-obs replicas of the spectral model held on the ForwardModel."""
        return self.forward_model.spectrum

    @property
    def instrument(self) -> dict[str, InstrumentModel]:
        return self.forward_model.instrument

    @property
    def background(self) -> dict[str, BackgroundModel]:
        return self.forward_model.background

    @property
    def settings(self) -> dict[str, Any]:
        return self.forward_model.settings

    # ----- Prior dict validation + default-merge -----

    def _build_prior_dict(self) -> dict | Callable:
        """Merge per-obs background and instrument defaults into the user prior
        (user wins). For callable priors, pass through unchanged."""
        if not isinstance(self._user_prior, dict):
            return self._user_prior

        prior = dict(self._user_prior)
        for modules in (self.forward_model.background, self.forward_model.instrument):
            for obs_name, module in modules.items():
                obs = self.forward_model.observations[obs_name]
                for key, value in module.default_prior(obs, obs_name).items():
                    prior.setdefault(key, value)
        return prior

    def _applicable_obs(self, prefix: str) -> set[str]:
        return set(_prefix_to_obs_names(self.forward_model).get(prefix, []))

    def _validate_prior_dict(self) -> None:
        """Validate the (effective) prior dict against the forward model.

        Callable priors are not validated structurally — the user is on the hook
        for their callable's correctness.
        """
        if not isinstance(self._effective_prior, dict):
            return
        leaves = _enumerate_leaves(self.forward_model)
        applicable = {prefix: self._applicable_obs(prefix) for prefix in _KNOWN_PREFIXES}
        for raw_key, value in self._effective_prior.items():
            self._validate_prior_entry(raw_key, value, leaves, applicable)

    def _validate_prior_entry(self, raw_key, value, leaves, applicable_by_prefix) -> None:
        # Catch flat keys like "tbabs_1_nh" before parse_prior_key — the
        # regex would accept them but downstream errors would be cryptic.
        if "." not in raw_key.split("[", 1)[0]:
            raise ValueError(
                f"Prior key {raw_key!r} has no module prefix. Expected a "
                f"dotted path like 'spectrum.<component>.<param>' (or "
                f"'instrument.<...>' / 'background.<...>'), optionally with "
                f"an [obs] or [*] suffix."
            )

        path, scope = parse_prior_key(raw_key)
        prefix = path.split(".", 1)[0]

        if prefix not in _KNOWN_PREFIXES:
            raise ValueError(
                f"Prior key {raw_key!r} starts with unknown module {prefix!r}. "
                f"The first dotted segment must be one of {_KNOWN_PREFIXES}. "
                f"Did you mean 'spectrum.{path}'?"
            )

        applicable = applicable_by_prefix[prefix]
        if not applicable:
            hint = (
                "Did you forget to pass instrument_model= to the fitter?"
                if prefix == "instrument"
                else "Did you forget to pass background_model= to the fitter?"
                if prefix == "background"
                else ""
            )
            raise ValueError(
                f"Prior key {raw_key!r} has prefix {prefix!r} but no observations "
                f"are attached to the {prefix!r} model. {hint}".rstrip()
            )
        if scope is not None and scope != "*" and scope not in applicable:
            raise ValueError(
                f"Prior key {raw_key!r} references observation {scope!r} which is "
                f"not in the {prefix!r} applicable set {sorted(applicable)}."
            )
        # Strict leaf-existence check: a key that resolves to zero leaves is a
        # typo'd parameter path — surface it at build time, not silently drop it.
        if not _resolve_targets(path, scope, leaves, applicable_by_prefix):
            raise KeyError(_unmatched_key_message(path, scope, leaves))
        if isinstance(value, dist.Distribution | TiedParameter):
            return
        try:
            jnp.asarray(value)
        except Exception as exc:
            raise TypeError(f"Invalid fixed prior value for {raw_key!r}: {value!r}") from exc

    # ----- numpyro model wiring -----

    def numpyro_model(self, observed: bool = True):
        """Sample the prior, evaluate the forward model, register likelihoods.

        Thin wrapper around :meth:`ForwardModel.evaluate`: this method owns
        only the numpyro-specific concerns (sample sites + Poisson
        likelihoods on the observed counts). The deterministic forward pass
        — spectral evaluation, instrument folding, background — lives on
        the forward model and is reused by ``fakeit`` and posterior-predictive
        checks.
        """
        inputs = self._sample_inputs()

        # Clone the forward_model per call so each evaluate sees a fresh tree
        # (the original module's Variables would otherwise accumulate tracers
        # across MCMC's repeated traces, surfacing as UnexpectedTracerError).
        # ``evaluate`` itself does NOT clone — that would break ``jax.vmap``.
        fresh_forward = nnx.clone(self.forward_model)
        predictions = fresh_forward.evaluate(inputs, missing_key_style="prior")

        fm = self.forward_model
        for obs_name, obs in fm.observations.items():
            source_flux = predictions[obs_name]["source"]
            bkg_rate = predictions[obs_name]["background"]
            bg = fm.background.get(obs_name)

            if bkg_rate is not None:
                if getattr(obs, "folded_background", None) is None:
                    raise ValueError(
                        "Trying to fit a background model but no background is "
                        "linked to this observation"
                    )
                bkg_in_obs = bkg_rate * obs.folded_backratio.data
                total = source_flux + bkg_in_obs

                if bg.is_stochastic:
                    with numpyro.plate(
                        f"observed_background_plate.{obs_name}", len(obs.folded_background)
                    ):
                        numpyro.sample(
                            f"observed_background.{obs_name}",
                            Poisson(bkg_rate),
                            obs=obs.folded_background.data if observed else None,
                        )
                else:
                    numpyro.deterministic(f"observed_background.{obs_name}", bkg_rate)
            else:
                total = source_flux

            with numpyro.plate(f"observed_plate.{obs_name}", len(obs.folded_counts)):
                numpyro.sample(
                    f"observed.{obs_name}",
                    Poisson(total),
                    obs=obs.folded_counts.data if observed else None,
                )

    # ----- Prior sampling -----

    def _sample_inputs(self) -> dict[str, Any]:
        """Sample the (effective) prior into the leaf-path inputs dict that
        :meth:`ForwardModel.evaluate` consumes.

        Creates the per-leaf numpyro sample sites along the way. Thin wrapper
        around :func:`~jaxspec.fit._prior_resolution.sample_prior` that
        provides the prefix → applicable-obs table built from
        :meth:`_applicable_obs`.
        """
        applicable = {prefix: self._applicable_obs(prefix) for prefix in _KNOWN_PREFIXES}
        return sample_prior(self.forward_model, self._effective_prior, applicable)

    def _spectrum_is_shared(self) -> bool:
        """Whether every spectral prior entry is shared across obs.

        Used at construction time to set :attr:`ForwardModel.settings`'s
        ``"spectrum_shared"`` flag, which lets :meth:`ForwardModel.evaluate`
        evaluate the spectrum **once** when a user energy grid is set
        (otherwise each obs's per-obs replica must be evaluated separately,
        e.g. when any spectral param has a ``[*]`` / ``[obs]`` scope).

        Conservative: callable priors return ``False`` since we cannot
        statically inspect them.
        """
        if not isinstance(self._effective_prior, dict):
            return False
        for raw_key in self._effective_prior:
            path, scope = parse_prior_key(raw_key)
            if path.startswith("spectrum.") and scope is not None:
                return False
        return True

    # ----- Cached properties for fitter machinery -----

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
        """Build the log likelihood function for each bin in each observation."""

        @jax.jit
        def log_likelihood_per_obs(constrained_params):
            log_likelihood = numpyro.infer.util.log_likelihood(
                model=self.numpyro_model, posterior_samples=constrained_params
            )
            return jax.tree.map(lambda x: jnp.where(jnp.isnan(x), -jnp.inf, x), log_likelihood)

        return log_likelihood_per_obs

    @cached_property
    def log_likelihood(self) -> Callable:
        """Build the total log likelihood function."""

        @jax.jit
        def log_likelihood(constrained_params):
            log_likelihood = self.log_likelihood_per_obs(constrained_params)
            return jax.tree.reduce(operator.add, jax.tree.map(jnp.sum, log_likelihood))

        return log_likelihood

    @cached_property
    def log_posterior_prob(self) -> Callable:
        """Build the posterior probability function.

        Enables distribution validation only during this trace so that
        out-of-support parameter values produce ``-inf`` log-probabilities
        (instead of silently bogus finite values), letting external samplers
        such as AIES / ESS / MH correctly reject them.

        We don't use ``numpyro.validation_enabled()`` because that context
        manager has an upstream bug: it saves ``_VALIDATION_ENABLED`` (a
        module-global) but restores via ``enable_validation`` which writes
        to *both* that global *and* ``Distribution._validate_args`` — and
        those two can be out of sync (they are at fresh import). The
        manual save/restore below targets ``Distribution._validate_args``
        directly, which is the attribute every ``Distribution`` instance
        actually reads at construction time.
        """

        @jax.jit
        def log_posterior_prob(constrained_params):
            with numpyro.validation_enabled(True):
                log_posterior_prob, _ = log_density(
                    self.numpyro_model, (), dict(observed=True), constrained_params
                )

            return jnp.where(jnp.isnan(log_posterior_prob), -jnp.inf, log_posterior_prob)

        return log_posterior_prob

    @cached_property
    def parameter_names(self) -> list[str]:
        """List of parameter names for the model."""
        relations = get_model_relations(self.numpyro_model)
        all_sites = relations["sample_sample"].keys()
        observed_sites = relations["observed"]
        return sorted(site for site in all_sites if site not in observed_sites)

    @cached_property
    def observation_names(self) -> list[str]:
        """List of the observations."""
        relations = get_model_relations(self.numpyro_model)
        all_sites = relations["sample_sample"].keys()
        observed_sites = relations["observed"]
        return sorted(site for site in all_sites if site in observed_sites)

    def array_to_dict(self, theta):
        return {name: theta[i] for i, name in enumerate(self.parameter_names)}

    def dict_to_array(self, dict_of_params):
        theta = jnp.zeros(len(self.parameter_names))
        for index, parameter_key in enumerate(self.parameter_names):
            theta = theta.at[index].set(dict_of_params[parameter_key])
        return theta

    def prior_samples(self, key: Array = rng_key(0), num_samples: int = 100):
        """Sample from the prior distribution."""

        @jax.jit
        def prior_sample(key):
            return Predictive(
                self.numpyro_model, return_sites=self.parameter_names, num_samples=num_samples
            )(key, observed=False)

        return prior_sample(key)

    def mock_observations(self, parameters, key: Array = rng_key(0)):
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
        key: Array = rng_key(0),
        num_samples: int = 1000,
        min_counts: int | None = None,
        grouping: int | None = None,
    ):
        """Check if the prior distribution includes the observed data."""
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

            bin_ids = _compute_bin_ids(observed, min_counts, grouping)

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
