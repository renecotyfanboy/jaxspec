"""Prior resolution machinery for the dict / callable forms.

This module owns the pure functions that walk a forward model's nnx tree,
match user prior entries to nnx leaves, and produce the leaf-callable adapter
consumed by :func:`_bind_priors`. The orchestrating
:class:`~jaxspec.fit.BayesianModel` keeps validation (which needs ``self``)
but defers the actual resolution to :func:`resolve_prior` here.
"""

from __future__ import annotations

import inspect
import re

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from flax import nnx

from ._parameter import TiedParameter

#: First dotted segment of every prior key. Used to validate that user keys
#: target one of the three model layers in the forward model.
_KNOWN_PREFIXES = ("spectrum", "instrument", "background")


def _bind_priors(nn_module, prior):
    """Sample every nnx.Param leaf of ``nn_module`` from ``prior`` and return a
    bound copy of the module with the sampled values.

    Per-obs numpyro sites are registered under the literal prefix ``"forward."``
    (e.g. ``"forward.spectrum.MOS1.powerlaw_1.norm"``). The prefix is applied
    manually rather than via ``numpyro.handlers.scope``: when the model is
    wrapped with ``numpyro.handlers.scope("forward", divider=".")`` and passed
    through :class:`~jaxspec.fit.NSFitter`, jaxns crashes during the
    ``parse_joint`` phase with ``unexpected PRNG key type <class 'NoneType'>``
    inside its internal UniformReparam — the scope handler renames sites at
    trace time but jaxns has pre-built substitution tables under the un-scoped
    names. Verified empirically on jaxns >=2.6.9 / numpyro >0.20.1.
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
      Used by the dict-form prior, where :func:`resolve_prior` samples each
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


def _split_nnx_leaf(leaf_path: str) -> tuple[str, str]:
    """Split ``"spectrum.MOS1.powerlaw_1.alpha"`` into ``("spectrum.powerlaw_1.alpha", "MOS1")``.

    The forward_model's nnx tree has shape ``{<prefix>: {<obs>: <Module>}}``
    where ``<prefix>`` is one of ``spectrum`` / ``instrument`` / ``background``.
    Raises ``ValueError`` on paths with fewer than three segments;
    :func:`~jaxspec.fit._parameter.dict_prior` relies on this to return
    ``None`` on miss.
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


def _enumerate_leaves(forward_model) -> dict[str, dict[str, str]]:
    """Return ``{user_facing_path: {obs_name: nnx_leaf_path}}``.

    Walks every parametric submodule under ``forward_model.spectrum``,
    ``forward_model.instrument`` and ``forward_model.background``. For
    background leaves the inner nnx path is remapped via :meth:`bg.user_path`
    so user prior keys like ``"background.powerlaw_1.alpha"`` match without
    the verbose ``spectral_model.`` wrapper segment.

    The returned table is the source of truth for *which* leaves a user prior
    entry targets: see :func:`_resolve_targets`.
    """
    leaves: dict[str, dict[str, str]] = {}
    for prefix in _KNOWN_PREFIXES:
        modules = getattr(forward_model, prefix)
        for obs_name, module in modules.items():
            _, params_state, _ = nnx.split(module, nnx.Param, nnx.Not(nnx.Param))
            for inner_path in _iter_pure_dict_paths(nnx.to_pure_dict(params_state)):
                leaf_path = f"{prefix}.{obs_name}.{inner_path}"
                user_inner = module.user_path(inner_path) if prefix == "background" else inner_path
                user_path = f"{prefix}.{user_inner}"
                leaves.setdefault(user_path, {})[obs_name] = leaf_path
    return leaves


def _iter_pure_dict_paths(d: dict, prefix: str = "") -> Any:
    """Yield dotted leaf paths from a nested pure-dict (``nnx.to_pure_dict``)."""
    for name, item in d.items():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            yield from _iter_pure_dict_paths(item, full)
        else:
            yield full


def _resolve_targets(
    path: str,
    scope: str | None,
    leaves: dict[str, dict[str, str]],
    applicable: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Return ``[(obs, leaf_path), ...]`` for a prior key ``(path, scope)``.

    * ``scope is None`` (shared) — every applicable obs that owns ``path``.
    * ``scope == "*"``           — every applicable obs that owns ``path``.
    * specific obs               — that obs only (silently skipped if no leaf).

    ``applicable`` is the prefix → set-of-obs table built by the caller;
    validation has already ensured the requested obs is in that set, so a
    miss here means "no leaf exists for this obs" (e.g. a background path
    that exists on some obs but not others).
    """
    by_obs = leaves.get(path, {})
    if not by_obs:
        return []
    prefix = path.split(".", 1)[0]
    applicable_for_prefix = applicable.get(prefix, set())
    if scope is None or scope == "*":
        return sorted((obs, leaf) for obs, leaf in by_obs.items() if obs in applicable_for_prefix)
    if scope in by_obs:
        return [(scope, by_obs[scope])]
    return []


def _sample_entry(
    path: str,
    scope: str | None,
    value: Any,
    leaves: dict[str, dict[str, str]],
    applicable: dict[str, set[str]],
    samples: dict[str, Any],
) -> None:
    """Sample a direct (non-tied) prior entry and write into ``samples``.

    Shared entries (scope=None) emit a *single* numpyro site under the bare
    path name and broadcast the same sample by identity to every targeted
    leaf. ``[*]`` and ``[obs]`` entries emit one site per leaf under the
    ``"forward.<prefix>.<obs>.<rest>"`` convention.
    """
    targets = _resolve_targets(path, scope, leaves, applicable)
    if not targets:
        return
    if scope is None:
        if isinstance(value, dist.Distribution):
            sample = numpyro.sample(path, value)
        else:
            sample = jnp.asarray(value)
        for _obs, leaf in targets:
            samples[leaf] = sample
        return
    for obs, leaf in targets:
        site = _per_obs_site_name(path, obs)
        if isinstance(value, dist.Distribution):
            samples[leaf] = numpyro.sample(site, value)
        else:
            samples[leaf] = jnp.asarray(value)


def _resolve_tied_entry(
    path: str,
    scope: str | None,
    tied: TiedParameter,
    leaves: dict[str, dict[str, str]],
    applicable: dict[str, set[str]],
    samples: dict[str, Any],
) -> None:
    """Apply a ``TiedParameter`` to every destination leaf, registering each as deterministic."""
    src_path, src_scope = parse_prior_key(tied.tied_to)
    src_targets = _resolve_targets(src_path, src_scope, leaves, applicable)
    if not src_targets:
        raise ValueError(
            f"TiedParameter {path!r} references unknown source {tied.tied_to!r}: "
            f"no leaves match. Check the source path / scope."
        )

    source_for_obs = _source_lookup_for_tie(src_scope, src_targets, samples)
    dest_targets = _resolve_targets(path, scope, leaves, applicable)

    if scope is None:
        # Shared dest: compute the derived value once, broadcast to every leaf,
        # register one deterministic site under the bare path.
        first_obs = dest_targets[0][0] if dest_targets else next(iter(src_targets))[0]
        value = tied.func(source_for_obs(first_obs))
        numpyro.deterministic(path, value)
        for _obs, dest_leaf in dest_targets:
            samples[dest_leaf] = value
        return

    for obs, dest_leaf in dest_targets:
        src_value = source_for_obs(obs)
        if src_value is None:
            raise ValueError(
                f"TiedParameter {path!r}[{obs!r}] cannot match a source value: "
                f"tied_to={tied.tied_to!r} resolved to {[o for o, _ in src_targets]!r}."
            )
        value = tied.func(src_value)
        samples[dest_leaf] = value
        numpyro.deterministic(_per_obs_site_name(path, obs), value)


def _source_lookup_for_tie(
    src_scope: str | None,
    src_targets: list[tuple[str, str]],
    samples: dict[str, Any],
) -> Callable[[str], Any]:
    """Return a ``dest_obs -> source_value`` lookup matching the source's scope.

    * src shared (None) or specific obs → one value, same for every dest obs.
    * src ``"*"``                       → element-wise pairing; returns ``None``
      if the dest obs has no matching source leaf.
    """
    if src_scope == "*":
        by_obs = {obs: samples[leaf] for obs, leaf in src_targets}
        return by_obs.get
    # Shared or specific: a single value applies to all dests.
    _obs, leaf = src_targets[0]
    value = samples[leaf]
    return lambda _dest_obs: value


def _per_obs_site_name(path: str, obs: str) -> str:
    """Compose the canonical per-obs site name ``"forward.<prefix>.<obs>.<rest>"``."""
    prefix, rest = path.split(".", 1)
    return f"forward.{prefix}.{obs}.{rest}"


def _missing_prior_message(leaf_path: str) -> str:
    """Build the rich error message for a leaf with no matching prior entry."""
    parts = leaf_path.split(".")
    if len(parts) < 3:
        return f"No prior provided for parameter {leaf_path!r}."
    prefix, obs, *rest = parts
    rest_dotted = ".".join(rest)
    return (
        f"No prior provided for parameter {leaf_path!r}. Add an entry like "
        f"'{prefix}.{rest_dotted}' (shared), "
        f"'{prefix}.{rest_dotted}[*]' (split), or "
        f"'{prefix}.{rest_dotted}[{obs}]' (specific) to the prior dict."
    )


def resolve_prior(
    forward_model,
    prior: dict | Callable,
    applicable: dict[str, set[str]],
) -> Callable:
    """Normalise a user prior into the leaf callable consumed by :func:`_bind_priors`.

    * Callable forms (2-arg leaf or 0-arg factory) are normalised by
      :func:`_normalise_callable_prior`.
    * Dict forms walk the forward-model nnx tree once via :func:`_enumerate_leaves`,
      sample each entry into ``samples[leaf_path]``, resolve ties, and return a
      ``dict.get``-style adapter.

    ``applicable`` is the prefix → set-of-obs table the caller built from
    :meth:`BayesianModel._applicable_obs`.
    """
    if callable(prior):
        return _normalise_callable_prior(prior)

    leaves = _enumerate_leaves(forward_model)
    samples: dict[str, Any] = {}
    deferred_ties: list[tuple[str, str | None, TiedParameter]] = []

    for raw_key, value in prior.items():
        path, scope = parse_prior_key(raw_key)
        if isinstance(value, TiedParameter):
            deferred_ties.append((path, scope, value))
            continue
        _sample_entry(path, scope, value, leaves, applicable, samples)

    for path, scope, tied in deferred_ties:
        _resolve_tied_entry(path, scope, tied, leaves, applicable, samples)

    def adapter(leaf_path: str, _shape):
        if leaf_path not in samples:
            raise KeyError(_missing_prior_message(leaf_path))
        return samples[leaf_path]

    return adapter
