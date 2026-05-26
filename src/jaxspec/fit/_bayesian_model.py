from __future__ import annotations

import inspect
import operator
import re

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
from ..model.abc import SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel
from ._forward_model import ForwardModel
from ._parameter import TiedParameter

#: First dotted segment of every prior key. Used to validate that user keys
#: target one of the three model layers in the forward model.
_KNOWN_PREFIXES = ("spectrum", "instrument", "background")

#: Default fold used by observations with no per-obs ``InstrumentModel``.
#: With ``gain=None`` and ``shift=None`` it has zero ``nnx.Param`` leaves and
#: zero state, so it never appears in the forward tree's parameter walk —
#: ``_applicable_obs("instrument")`` stays restricted to the user-instrumented
#: obs and no extra numpyro sites get registered.
_IDENTITY_INSTRUMENT = InstrumentModel()


def _bind_priors(nn_module, prior):
    """Sample every nnx.Param leaf of ``nn_module`` from ``prior`` and return a
    bound copy of the module with the sampled values.

    Per-obs numpyro sites are registered under the literal prefix ``"forward."``
    (e.g. ``"forward.spectrum.MOS1.powerlaw_1.norm"``). The prefix is applied
    manually rather than via ``numpyro.handlers.scope`` — the handler's
    site-rename interacts badly with NSFitter's UniformReparam under jaxns
    scoring (substitution coverage breaks for handler-added sites).
    """
    graph_def, params_state, other_state = nnx.split(nn_module, nnx.Param, nnx.Not(nnx.Param))
    params_pure = nnx.to_pure_dict(params_state)

    _sample_leaves(params_pure, prior, prefix="", site_prefix="forward.")

    nnx.replace_by_pure_dict(params_state, params_pure)
    return nnx.merge(graph_def, params_state, other_state, copy=True)


def _sample_leaves(
    params: dict,
    prior,
    *,
    prefix: str,
    site_prefix: str = "",
) -> None:
    """Walk the pure-dict nnx params tree, binding each leaf via the prior callable.

    Writes back into ``params`` in-place; safe because we read each leaf into
    ``item`` before assigning the new value.

    The prior callable receives ``(flatten_name, shape)`` and returns one of:

    * a :class:`numpyro.distributions.Distribution` — registered as a numpyro
      sample site under ``site_prefix + flatten_name`` and assigned to the leaf.
    * any array-like value — written directly to the leaf with no numpyro site.
      Used by the dict-form prior, where :meth:`_resolve_prior` samples each
      entry once and the adapter routes pre-sampled values to leaves. Avoids
      spawning redundant sites for the per-obs duplicates of shared params,
      which keeps the trace clean for nested-sampling backends.
    """
    for name, item in params.items():
        flatten_name = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            _sample_leaves(item, prior, prefix=flatten_name, site_prefix=site_prefix)
            continue
        shape = jnp.shape(item)
        result = prior(flatten_name, shape) if callable(prior) else prior[flatten_name]
        if isinstance(result, dist.Distribution):
            event_dim = getattr(result, "event_dim", 0)
            batch_shape = shape[: len(shape) - event_dim]
            params[name] = numpyro.sample(
                f"{site_prefix}{flatten_name}", result.expand(batch_shape).to_event()
            )
        else:
            params[name] = jnp.asarray(result)


class BayesianModel:
    """
    Bayesian spectral model that composes a deterministic forward model with
    numpyro priors and Poisson likelihoods.

    Parameters:
        model: The spectral model to fit (cloned per observation inside the
            internal :class:`~jaxspec.fit._forward_model.ForwardModel`).
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
        model: SpectralModel,
        prior: dict | Callable,
        observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
        background_model: BackgroundModel | dict[str, BackgroundModel | None] | None = None,
        instrument_model: dict[str, InstrumentModel | None] | None = None,
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
        self._user_prior = prior
        self._effective_prior = self._build_prior_dict()
        self._validate_prior_dict()

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
        """Merge per-obs background defaults into the user prior (user wins).
        For callable priors, pass through unchanged."""
        if not isinstance(self._user_prior, dict):
            return self._user_prior

        prior = dict(self._user_prior)
        for obs_name, bg in self.forward_model.background.items():
            obs = self.forward_model.observations[obs_name]
            for key, value in bg.default_prior(obs, obs_name).items():
                prior.setdefault(key, value)
        return prior

    def _applicable_obs(self, prefix: str) -> set[str]:
        fm = self.forward_model
        if prefix == "spectrum":
            return set(fm.observations.keys())
        if prefix == "instrument":
            return set(fm.instrument.keys())
        if prefix == "background":
            return set(fm.background.keys())
        return set()

    def _validate_prior_dict(self) -> None:
        """Validate the (effective) prior dict against the forward model.

        Callable priors are not validated structurally — the user is on the hook
        for their callable's correctness.
        """
        if not isinstance(self._effective_prior, dict):
            return
        for raw_key, value in self._effective_prior.items():
            self._validate_prior_entry(raw_key, value)

    def _validate_prior_entry(self, raw_key, value) -> None:
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

        applicable = self._applicable_obs(prefix)
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
        if isinstance(value, dist.Distribution | TiedParameter):
            return
        try:
            jnp.asarray(value)
        except Exception as exc:
            raise TypeError(f"Invalid fixed prior value for {raw_key!r}: {value!r}") from exc

    # ----- numpyro model wiring -----

    def numpyro_model(self, observed: bool = True):
        """Build the full numpyro model: bind priors, compute counts, register likelihoods."""
        prior_fn = self._resolve_prior()

        # Clone the forward_model per call so each ``_bind_priors`` invocation
        # sees a fresh tree (the original module's Variables would otherwise
        # accumulate tracers across MCMC's repeated traces, surfacing as
        # UnexpectedTracerError).
        fresh_forward = nnx.clone(self.forward_model)

        bound_forward = _bind_priors(fresh_forward, prior=prior_fn)

        fm = self.forward_model
        n_points = fm.settings["n_points"]

        for obs_name, obs in fm.observations.items():
            spec = bound_forward.spectrum[obs_name]
            cache = fm.obs_caches[obs_name]

            inst = bound_forward.instrument.get(obs_name, _IDENTITY_INSTRUMENT)
            source = inst.fold(obs, cache, spec, n_points=n_points)

            bg = bound_forward.background.get(obs_name)
            if bg is not None:
                if getattr(obs, "folded_background", None) is None:
                    raise ValueError(
                        "Trying to fit a background model but no background is "
                        "linked to this observation"
                    )
                bkg_rate = bg(obs)
                bkg_in_obs = bkg_rate * obs.folded_backratio.data
                total = source + bkg_in_obs

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
                total = source

            with numpyro.plate(f"observed_plate.{obs_name}", len(obs.folded_counts)):
                numpyro.sample(
                    f"observed.{obs_name}",
                    Poisson(total),
                    obs=obs.folded_counts.data if observed else None,
                )

    # ----- Prior resolution helpers -----

    def _resolve_prior(self) -> Callable:
        """Normalise the user prior into a leaf callable.

        For the dict form we pre-sample every entry up-front: shared paths get
        sampled once under their bare name (``"spectrum.powerlaw_1.alpha"``);
        per-obs paths under their full prefixed name
        (``"forward.spectrum.MOS1.powerlaw_1.norm"``). The returned adapter
        routes each nnx leaf path to its already-sampled raw value — when
        ``_bind_priors`` walks the nnx tree, every leaf is covered by a
        pre-sampled array (no numpyro.sample inside the walk). This makes
        ``TiedParameter`` resolution straightforward (sources are always
        available in the samples dict, regardless of obs scope) and keeps the
        trace clean.
        """
        prior = self._effective_prior

        if callable(prior):
            return _normalise_callable_prior(prior)

        # Dict form: sample every entry up-front, then build a lookup adapter.
        samples: dict[tuple[str, str | None], Any] = {}
        deferred_ties: list[tuple[str, str | None, TiedParameter]] = []

        # Pass 1: sample non-tied entries with appropriate site names.
        for raw_key, value in prior.items():
            path, scope = parse_prior_key(raw_key)
            if isinstance(value, TiedParameter):
                deferred_ties.append((path, scope, value))
                continue
            for sample_obs, site_name in self._expand_scope_for_sampling(path, scope):
                if isinstance(value, dist.Distribution):
                    samples[(path, sample_obs)] = numpyro.sample(site_name, value)
                else:
                    samples[(path, sample_obs)] = jnp.asarray(value)

        # Pass 2: resolve ties (sources are now all in `samples`). Each
        # resolved tie is also registered as ``numpyro.deterministic`` so it
        # appears in the trace for posterior inspection.
        for path, scope, tied in deferred_ties:
            src_path, src_scope = parse_prior_key(tied.tied_to)
            for dest_obs, dest_site in self._expand_scope_for_sampling(path, scope):
                key = _tied_source_key(src_scope, src_path, dest_obs)
                source_value = _lookup_tied_source(samples, key, path, tied.tied_to)
                value = tied.func(source_value)
                samples[(path, dest_obs)] = value
                numpyro.deterministic(dest_site, value)

        return self._build_dict_adapter(samples)

    def _expand_scope_for_sampling(
        self, path: str, scope: str | None
    ) -> list[tuple[str | None, str]]:
        """Return ``[(sample_obs, site_name), ...]`` for a prior entry.

        ``sample_obs=None`` denotes a shared sample; otherwise the specific
        observation the sample belongs to. ``site_name`` is the numpyro site
        name to register the sample under.
        """
        if scope is None:
            return [(None, path)]
        prefix, rest = path.split(".", 1)
        if scope == "*":
            applicable = sorted(self._applicable_obs(prefix))
            return [(obs, f"forward.{prefix}.{obs}.{rest}") for obs in applicable]
        # Specific obs
        return [(scope, f"forward.{prefix}.{scope}.{rest}")]

    def _build_dict_adapter(self, samples: dict[tuple[str, str | None], Any]) -> Callable:
        """Return the per-leaf prior callable consumed by ``_bind_priors``.

        Every per-leaf invocation returns the pre-sampled raw array — all
        sampling happened in :meth:`_resolve_prior`. The adapter routes each
        nnx leaf path to its matching value; ``_sample_leaves`` writes it
        straight to the leaf with no numpyro site.

        Resolution order for each leaf path ``"<prefix>.<obs>.<rest>"``:
            1. ``samples[(base, obs)]``  (specific per-obs entry; covers ``[obs]`` and ``[*]``)
            2. ``samples[(base, None)]``  (shared)
            3. fall back to SpectralModelBackground's ``user_path`` re-mapping
            4. raise KeyError              (loud failure on missing prior)
        """

        background_models = self.forward_model.background

        def adapter(leaf_path: str, shape):
            base, obs = _split_nnx_leaf(leaf_path)
            candidates = _candidate_user_paths(base, obs, background_models)

            for cand in candidates:
                if (cand, obs) in samples:
                    return samples[(cand, obs)]
                if (cand, None) in samples:
                    return samples[(cand, None)]

            raise KeyError(
                f"No prior provided for parameter {leaf_path!r}. "
                f"Tried user-facing paths: {candidates}. "
                f"Add an entry like {candidates[-1]!r} (shared), "
                f"{candidates[-1] + '[*]'!r} (split), or "
                f"{candidates[-1] + f'[{obs}]'!r} (specific) to the prior dict."
            )

        return adapter

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


# ---- Module-level helpers used by the dict→callable adapter ----------------


def _split_nnx_leaf(leaf_path: str) -> tuple[str, str]:
    """Split ``"spectrum.MOS1.powerlaw_1.alpha"`` into ``("spectrum.powerlaw_1.alpha", "MOS1")``.

    The forward_model's nnx tree has shape ``{<prefix>: {<obs>: <Module>}}``
    where ``<prefix>`` is one of ``spectrum`` / ``instrument`` / ``background``.
    Raises ``ValueError`` on paths with fewer than three segments;
    :func:`dict_prior` relies on this to return ``None`` on miss.
    """
    parts = leaf_path.split(".")
    if len(parts) < 3:
        raise ValueError(f"Unexpected nnx leaf path: {leaf_path!r}")
    prefix, obs, *rest = parts
    return f"{prefix}.{'.'.join(rest)}", obs


_PRIOR_KEY_RE = re.compile(r"^(?P<path>[^\[\]]+?)(?:\[(?P<scope>[^\[\]]+)\])?$")


def parse_prior_key(key: str) -> tuple[str, str | None]:
    """Split a prior dict key into ``(path, scope)``.

    ``scope`` is ``None`` for a bare key (shared across applicable obs),
    ``"*"`` for the wildcard (split across all applicable obs), or a specific
    observation name.
    """
    match = _PRIOR_KEY_RE.match(key)
    if match is None:
        raise ValueError(f"Malformed prior key: {key!r}")
    return match.group("path"), match.group("scope")


def _normalise_callable_prior(prior: Callable) -> Callable:
    """Resolve a callable prior to its 2-arg leaf-callable form.

    Two callable shapes are auto-detected by argument count:
      * 2 args → leaf callable ``(path, shape) -> Distribution``; used as-is.
      * 0 args → factory ``() -> leaf_callable``; invoked inside the trace
        so it can sample shared/hierarchical params before returning
        the leaf callable.
    """
    n_params = len(inspect.signature(prior).parameters)
    if n_params == 2:
        return prior
    if n_params == 0:
        return prior()
    raise TypeError(
        f"Callable prior must take either 0 args (factory `() -> leaf_callable`) "
        f"or 2 args (leaf callable `(path, shape) -> Distribution`); got {n_params}."
    )


def _tied_source_key(
    src_scope: str | None, src_path: str, dest_obs: str | None
) -> tuple[str, str | None]:
    """Map a tied source's scope to the (path, obs) key under which it was sampled.

    Three cases:
      * source scope unscoped → ``(src_path, None)``
      * source scope ``"*"``  → ``(src_path, dest_obs)`` (element-wise tie)
      * specific source scope → ``(src_path, src_scope)`` (same value for all dests)
    """
    if src_scope is None:
        return (src_path, None)
    if src_scope == "*":
        return (src_path, dest_obs)
    return (src_path, src_scope)


def _lookup_tied_source(
    samples: dict[tuple[str, str | None], Any],
    key: tuple[str, str | None],
    dest_path: str,
    tied_to: str,
):
    """Fetch the source sample for a tied parameter; raise with a rich message on miss."""
    if key not in samples:
        raise ValueError(
            f"TiedParameter {dest_path!r} references unknown source "
            f"{tied_to!r} (resolved key={key})."
        )
    return samples[key]


def _candidate_user_paths(base: str, obs: str, background_models) -> list[str]:
    """Resolve user-facing prior keys for an nnx leaf base.

    Background models with wrapper segments (``SpectralModelBackground`` nests
    under ``spectral_model.``) expose a flattened user-facing path. When the
    base is a background path, append the flattened candidate so the adapter
    can match the user's dict key form.
    """
    candidates = [base]
    if base.startswith("background."):
        bg = background_models.get(obs)
        if bg is not None:
            inner = base[len("background.") :]
            user_inner = bg.user_path(inner)
            if user_inner != inner:
                candidates.append(f"background.{user_inner}")
    return candidates
