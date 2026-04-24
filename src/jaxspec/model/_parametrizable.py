from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro

from jax.typing import ArrayLike
from numpyro.distributions import Distribution

if TYPE_CHECKING:
    import arviz as az


class ParametrizableMixin(ABC):
    """Mixin providing standard prior-registration logic for models that accept
    prior distributions.

    Any model class that accepts a prior dict and registers numpyro sample
    sites can inherit from this mixin to reuse the shared sampling engine
    (:meth:`register_priors`), posterior-sample reconstruction
    (:meth:`extract_posterior_samples`), and the :meth:`default_prior` hook
    for data-dependent defaults.

    Subclasses declare the routing prefix they own via the class attribute
    :attr:`prior_prefix` (e.g. ``"spectrum."``, ``"instrument."``,
    ``"background."``). Only keys in the unified prior dict starting with that
    prefix are consumed by this model.
    """

    @property
    @abstractmethod
    def prior_prefix(self) -> str:
        """
        Routing prefix in the unified prior dict (e.g. `"spectrum."`). Must be set by concrete subclasses.
        """
        pass

    def _default_skip_observation(self) -> str | None:
        """Return the observation to exclude by default from prior registration and
        posterior reconstruction.

        Subclasses override this when they always want to skip a specific observation
        (e.g. the reference in :class:`~jaxspec.model.instrument.InstrumentModel`). The
        default returns ``None`` (keep every observation). Callers of
        :meth:`register_priors` / :meth:`extract_posterior_samples` that pass an
        explicit ``skip_observation`` override this hook.
        """
        return None

    def register_priors(
        self,
        prior_distributions: dict,
        observation_names: list[str],
        *,
        skip_observation: str | None = None,
    ) -> dict[str, dict[str, ArrayLike]]:
        """Register this model's prior entries as numpyro sites.

        The unified prior dict is filtered by :attr:`prior_prefix`, then each
        matching entry is converted to a ``{observation_name: value}`` mapping.
        Shared priors are sampled once and reused for every kept observation,
        :class:`~jaxspec.fit.PerObs` priors produce one value per observation,
        fixed values are converted with :func:`jax.numpy.asarray`, and
        :class:`~jaxspec.fit.TiedParameter` entries are resolved after their
        source parameter has been materialized.

        Parameters:
            prior_distributions: Unified prior dict containing every model's
                priors, keyed by fully-qualified dotted parameter names.
            observation_names: Observation names in the order used by the
                forward model.
            skip_observation: Optional observation to exclude from sampling and
                from the returned nested dict.

        Returns:
            Nested parameter mapping ``{site_name: {observation_name: value}}``
            for the entries owned by this model.
        """
        if skip_observation is None:
            skip_observation = self._default_skip_observation()

        parameters: dict[str, dict[str, ArrayLike]] = {}
        tied_entries: list[tuple[str, object]] = []
        kept_obs = [obs for obs in observation_names if obs != skip_observation]

        for site_name, kind, payload in _iter_prior_entries(
            prior_distributions,
            self.prior_prefix,
            observation_names,
            skip_observation=skip_observation,
        ):
            if kind == "per_obs":
                obs_values: dict[str, ArrayLike] = {}
                for obs_name, sub_site, inner in payload:
                    if isinstance(inner, Distribution):
                        obs_values[obs_name] = numpyro.sample(sub_site, inner)
                    else:
                        obs_values[obs_name] = jnp.asarray(inner)
                parameters[site_name] = obs_values

            elif kind == "shared":
                if isinstance(payload, Distribution):
                    sample = numpyro.sample(site_name, payload)
                else:
                    sample = jnp.asarray(payload)
                parameters[site_name] = {obs: sample for obs in kept_obs}

            else:  # "tied"
                tied_entries.append((site_name, payload))

        for site_name, tied in tied_entries:
            if tied.tied_to not in parameters:
                raise ValueError(
                    f"TiedParameter {site_name!r} references unknown source {tied.tied_to!r}"
                )
            src = parameters[tied.tied_to]
            parameters[site_name] = {obs: tied.func(src[obs]) for obs in kept_obs}

        return parameters

    def extract_posterior_samples(
        self,
        inference_data: az.InferenceData,
        prior_distributions: dict,
        observation_names: list[str],
        *,
        skip_observation: str | None = None,
    ) -> dict[str, ArrayLike]:
        """Rebuild this model's posterior parameter arrays from inference output.

        The unified prior dict is filtered by ``prior_prefix`` using the same routing rules as ``register_priors``.
        Sampled shared parameters are read from the posterior and broadcast across a trailing observation axis, fixed
        shared values are broadcast to ``(chain, draw, *site_shape, n_obs)``, and per-observation parameters are
        stacked on that trailing axis when all observation leaves share the same shape. When per-observation leaves
        have incompatible shapes, the result is kept as ``{observation_name: array}``.

        Parameters:
            inference_data: Posterior samples stored in an ``arviz.InferenceData``.
            prior_distributions: Unified prior dict containing every model's
                priors, keyed by dotted parameter names.
            observation_names: Observation names in the order used by the forward model.
            skip_observation: Optional observation to exclude from the rebuilt output.

        Returns:
            Mapping ``{site_name: array}`` for this model's parameters. Arrays
            use a trailing observation axis when the per-observation leaves can
            be stacked; otherwise the site maps to ``{observation_name: array}``.
        """
        import arviz as az

        if skip_observation is None:
            skip_observation = self._default_skip_observation()

        posterior = az.extract(inference_data, combined=False)
        chain_draw = (posterior.sizes["chain"], posterior.sizes["draw"])
        data_vars = {
            name: jnp.asarray(data_array.data) for name, data_array in posterior.data_vars.items()
        }
        n_obs = sum(obs != skip_observation for obs in observation_names)

        out: dict[str, ArrayLike] = {}
        tied_entries: list[tuple[str, object]] = []

        for site_name, kind, payload in _iter_prior_entries(
            prior_distributions,
            self.prior_prefix,
            observation_names,
            skip_observation=skip_observation,
        ):
            if kind == "per_obs":
                leaves = [
                    (
                        obs_name,
                        _extract_posterior_leaf(
                            inner,
                            data_vars=data_vars,
                            site_name=sub_site,
                            chain_draw=chain_draw,
                        ),
                    )
                    for obs_name, sub_site, inner in payload
                ]
                out[site_name] = _stack_observation_leaves(leaves)

            elif kind == "shared":
                if site_name in data_vars:
                    out[site_name] = _append_obs_axis(data_vars[site_name], n_obs)
                else:
                    out[site_name] = _broadcast_fixed(payload, chain_draw=chain_draw, n_obs=n_obs)

            else:  # "tied"
                tied_entries.append((site_name, payload))

        for site_name, tied in tied_entries:
            if tied.tied_to not in out:
                raise ValueError(
                    f"TiedParameter {site_name!r} references unknown source {tied.tied_to!r}"
                )
            source = out[tied.tied_to]
            if isinstance(source, dict):
                out[site_name] = {obs: tied.func(val) for obs, val in source.items()}
            else:
                out[site_name] = tied.func(source)

        return out

    def default_prior(self, observations: dict) -> dict:
        """Return default priors, possibly data-dependent.

        Subclasses override this to provide defaults (e.g. ``BackgroundWithError`` constructs Gamma priors from
        observed background counts). The returned dict is merged with user-provided priors, with user entries
        taking precedence.

        Parameters:
            observations: ``{name: ObsConfiguration}`` dict.

        Returns:
            A prior dict with fully-qualified keys (e.g. ``"background.countrate"``).
        """
        return {}


def _iter_prior_entries(
    prior_dict: dict,
    prefix: str,
    observation_names: list[str],
    *,
    skip_observation: str | None = None,
):
    """Walk the prior dict and yield ``(site_name, kind, payload)`` triples.

    Kinds and payloads:

    - ``"shared"``: ``payload`` is a ``numpyro.Distribution`` or a fixed value.
      One numpyro site named ``site_name``, shared across all observations.
    - ``"per_obs"``: ``payload`` is a list of ``(obs_name, sub_site, inner)``
      tuples. One numpyro site per observation named ``f"{site_name}.{obs}"``,
      where ``inner`` is either a ``numpyro.Distribution`` or a fixed value.
    - ``"tied"``: ``payload`` is a ``TiedParameter``. Resolved by the
      caller in a second pass after every other entry has been processed.

    If ``skip_observation`` is given, that observation name is omitted from
    ``PerObs`` expansion — no leaf is yielded for it.

    The prior dict is expected to have already been validated by ``BayesianModel``.
    """
    from ..fit._parameter import PerObs, TiedParameter

    for site_name, prior in prior_dict.items():
        # Skip parameters that don't match the prefix
        if not site_name.startswith(prefix):
            continue

        if isinstance(prior, PerObs):
            if prior.is_homogeneous:
                leaves = [
                    (obs, f"{site_name}.{obs}", prior.value)
                    for obs in observation_names
                    if obs != skip_observation
                ]
            else:
                leaves = [
                    (obs, f"{site_name}.{obs}", prior.value[obs])
                    for obs in observation_names
                    if obs != skip_observation
                ]
            yield site_name, "per_obs", leaves

        elif isinstance(prior, TiedParameter):
            yield site_name, "tied", prior

        else:
            yield site_name, "shared", prior


def _append_obs_axis(value, n_obs: int):
    a = jnp.asarray(value)
    return jnp.broadcast_to(a[..., None], (*a.shape, n_obs))


def _broadcast_fixed(value, *, chain_draw: tuple[int, int], n_obs: int):
    a = jnp.asarray(value)
    expanded = a[..., None] if a.ndim >= 1 else a
    return jnp.broadcast_to(expanded, (*chain_draw, *a.shape, n_obs))


def _extract_posterior_leaf(
    value,
    *,
    data_vars: dict[str, ArrayLike],
    site_name: str,
    chain_draw: tuple[int, int],
):
    if isinstance(value, Distribution):
        return data_vars[site_name]
    return jnp.broadcast_to(jnp.asarray(value), (*chain_draw, *jnp.shape(value)))


def _stack_observation_leaves(leaves: list[tuple[str, ArrayLike]]):
    arrays = [leaf for _, leaf in leaves]
    if all(array.shape == arrays[0].shape for array in arrays[1:]):
        return jnp.stack(arrays, axis=-1)

    return {obs_name: leaf for obs_name, leaf in leaves}
