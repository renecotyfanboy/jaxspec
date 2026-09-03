"""Resolve priors and bind parameter values to forward models.

Priors may be supplied as mappings or callables and may contain fixed values,
NumPyro distributions, and :class:`TiedParameter` instances. This module resolves
those specifications to flat ``{leaf_path: value}`` dictionaries and binds such
dictionaries to the corresponding ``nnx.Param`` leaves of a forward model.
"""

from __future__ import annotations

import difflib
import inspect
import re

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from flax import nnx

from ._parameter import TiedParameter

_KNOWN_PREFIXES = ("spectrum", "instrument", "background")

# Namespace for observation-specific NumPyro sites.
_SITE_PREFIX = "forward."

# Namespace for quantities a model derives from its parameters. It prefixes the
# grammar above unchanged: a user path when shared, a ``_SITE_PREFIX`` name otherwise.
_DERIVED_PREFIX = "derived."


def bind_inputs(forward_model, inputs, *, missing_key_style: str = "inputs"):
    """Return a forward model whose parameter leaves are populated from ``inputs``.

    ``inputs`` must map every fully resolved parameter path to a value. The function
    updates a pure parameter state and merges it with the model's graph and non-parameter
    state; it does not clone the model or create NumPyro sites.

    Args:
        forward_model: Forward model containing the ``nnx.Param`` leaves to populate.
        inputs: Mapping from fully resolved leaf paths to parameter values.
        missing_key_style: Error-message style for missing values. ``"inputs"`` refers
            to resolved input paths, while ``"prior"`` suggests valid prior-dict keys.

    Raises:
        KeyError: If an ``nnx.Param`` leaf has no corresponding value in ``inputs``.
    """
    graph_def, params_state, other_state = nnx.split(forward_model, nnx.Param, nnx.Not(nnx.Param))
    params_pure = nnx.to_pure_dict(params_state)

    def _lookup(leaf_path, _shape):
        try:
            return inputs[leaf_path]
        except KeyError:
            raise KeyError(_missing_prior_message(leaf_path, style=missing_key_style)) from None

    _sample_leaves(params_pure, _lookup, prefix="", site_prefix=_SITE_PREFIX)

    nnx.replace_by_pure_dict(params_state, params_pure)
    return nnx.merge(graph_def, params_state, other_state)


def _sample_leaves(
    params: dict,
    prior: Callable,
    *,
    prefix: str,
    site_prefix: str = "",
) -> None:
    """Replace every leaf in ``params`` with the value returned by ``prior``.

    ``prior`` is called with each dotted leaf path and its shape. Distribution
    results are sampled at ``site_prefix + leaf_path``; other results are converted
    to JAX arrays. The nested ``params`` mapping is modified in place.
    """
    for name, item in params.items():
        flatten_name = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            _sample_leaves(item, prior, prefix=flatten_name, site_prefix=site_prefix)
            continue
        shape = jnp.shape(item)
        result = prior(flatten_name, shape)
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

    Args:
        leaf_path: Fully resolved path whose second segment is an observation name.

    Returns:
        The observation-independent parameter path and the observation name.

    Raises:
        ValueError: If ``leaf_path`` contains fewer than three segments.
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

    The mapping includes every parameter below the spectrum, instrument, and
    background namespaces. Background modules translate their internal parameter
    paths through ``user_path`` so the keys match the public prior syntax.
    """
    leaves: dict[str, dict[str, str]] = {}
    for prefix in _KNOWN_PREFIXES:
        modules = getattr(forward_model, prefix)
        for obs_name, module in modules.items():
            _, params_state, _ = nnx.split(module, nnx.Param, nnx.Not(nnx.Param))
            for inner_path, _ in _iter_pure_dict_items(nnx.to_pure_dict(params_state)):
                leaf_path = f"{prefix}.{obs_name}.{inner_path}"
                user_inner = module.user_path(inner_path) if prefix == "background" else inner_path
                user_path = f"{prefix}.{user_inner}"
                leaves.setdefault(user_path, {})[obs_name] = leaf_path
    return leaves


def _iter_pure_dict_items(d: dict, prefix: str = "") -> Any:
    """Yield ``(dotted_path, value)`` pairs from a nested pure-dict (``nnx.to_pure_dict``)."""
    for name, item in d.items():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            yield from _iter_pure_dict_items(item, full)
        else:
            yield full, item


def _resolve_targets(
    path: str,
    scope: str | None,
    leaves: dict[str, dict[str, str]],
    applicable: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Return ``[(obs, leaf_path), ...]`` for a prior key ``(path, scope)``.

    A shared or wildcard scope selects every applicable observation that owns the
    path. A named scope selects only that observation. Returns an empty list when
    the path or requested observation has no matching leaf.
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
        raise KeyError(_unmatched_key_message(path, scope, leaves))

    def draw(site: str):
        """A distribution becomes a site under ``site``; a fixed value becomes an array."""
        if isinstance(value, dist.Distribution):
            return numpyro.sample(site, value)
        return jnp.asarray(value)

    if scope is None:
        sample = draw(path)
        for _obs, leaf in targets:
            samples[leaf] = sample
        return

    for obs, leaf in targets:
        samples[leaf] = draw(_per_obs_site_name(path, obs))


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
    if not dest_targets:
        raise KeyError(_unmatched_key_message(path, scope, leaves))

    if scope is None:
        if src_scope == "*":
            raise ValueError(
                f"Cannot resolve shared TiedParameter {path!r}: source {tied.tied_to!r} "
                f"provides a different value for each observation, but the unscoped "
                f"destination requires one value shared by all observations. Use "
                f"'{path}[*]' to pair each destination with its same-observation source, "
                f"or tie the shared destination to an unscoped or specific-observation "
                f"source."
            )
        first_obs = dest_targets[0][0]
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
    _obs, leaf = src_targets[0]
    value = samples[leaf]
    return lambda _dest_obs: value


def _per_obs_site_name(path: str, obs: str) -> str:
    """Compose the canonical per-obs site name ``"forward.<prefix>.<obs>.<rest>"``."""
    prefix, rest = path.split(".", 1)
    return f"{_SITE_PREFIX}{prefix}.{obs}.{rest}"


def _parse_per_obs_site_name(site: str) -> tuple[str, str] | None:
    """Inverse of :func:`_per_obs_site_name`: ``site -> (user_path, obs)``.

    Returns ``None`` for anything that is not a per-obs site (shared entries, post-fit
    ``derived.<kind>_<band>`` names, observed data), so callers can branch on the result
    instead of hand-rolling ``split(".")`` unpacking. Strip ``_DERIVED_PREFIX`` first to
    parse a component's per-obs derived site.
    """
    if not site.startswith(_SITE_PREFIX):
        return None

    parts = site[len(_SITE_PREFIX) :].split(".")
    if len(parts) < 3:
        return None

    prefix, obs, *rest = parts
    return f"{prefix}.{'.'.join(rest)}", obs


def _prefix_to_obs_names(forward_model) -> dict[str, list[str]]:
    """Map each known prefix to the obs names it applies to, in the forward
    model's insertion order: ``spectrum`` → every observation,
    ``instrument`` / ``background`` → the obs that own a model."""
    return {
        "spectrum": list(forward_model.observations.keys()),
        "instrument": list(forward_model.instrument.keys()),
        "background": list(forward_model.background.keys()),
    }


def _unmatched_key_message(path: str, scope: str | None, leaves: dict[str, dict[str, str]]) -> str:
    """Build the error message for a prior key that resolves to zero leaves."""
    key = path if scope is None else f"{path}[{scope}]"
    if path in leaves:
        owners = sorted(leaves[path])
        return (
            f"Prior key {key!r} matches no parameter: {path!r} only exists on "
            f"observation(s) {owners}."
        )
    close = difflib.get_close_matches(path, leaves, n=3, cutoff=0.6)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return f"Prior key {key!r} does not match any model parameter.{hint}"


def _validate_no_conflicting_keys(
    prior: dict,
    leaves: dict[str, dict[str, str]],
    applicable: dict[str, set[str]],
) -> None:
    """Raise if two prior keys resolve to the same model parameter.

    Shared, wildcard, and observation-specific entries must target disjoint sets of
    leaves. Overlapping entries are rejected because prior dictionaries do not define
    a precedence rule between them.
    """
    owner: dict[str, str] = {}
    for raw_key in prior:
        path, scope = parse_prior_key(raw_key)
        for obs, leaf in _resolve_targets(path, scope, leaves, applicable):
            previous = owner.get(leaf)
            if previous is not None and previous != raw_key:
                raise ValueError(
                    f"Prior keys {previous!r} and {raw_key!r} both set the same "
                    f"parameter ({path!r} on observation {obs!r}). jaxspec does not "
                    f"apply a precedence rule between overlapping keys — one draw "
                    f"would silently be discarded. Use disjoint keys: either one "
                    f"shared entry {path!r} for every observation, or one explicit "
                    f"'{path}[<obs>]' entry per observation."
                )
            owner[leaf] = raw_key


def _missing_prior_message(leaf_path: str, *, style: str = "inputs") -> str:
    """Build the rich error message for a leaf with no matching value.

    ``style`` tailors the advice to the calling flow:

    * ``"inputs"`` — the direct :meth:`ForwardModel.evaluate` / ``fakeit`` path,
      whose ``inputs`` dict is keyed by fully-resolved leaf paths. Points the user
      at the resolved key verbatim and notes the bracketed prior-dict syntax does
      not apply here.
    * ``"prior"`` — the fitter prior-dict path, where the miss is an omitted prior
      entry. Suggests the shared / ``[*]`` / ``[obs]`` key forms.
    """
    parts = leaf_path.split(".")
    if style == "prior":
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
    if len(parts) < 3:
        return f"No value provided for parameter {leaf_path!r}."
    prefix, _obs, *rest = parts
    rest_dotted = ".".join(rest)
    return (
        f"No value provided for parameter {leaf_path!r}. evaluate() takes a flat "
        f"inputs dict keyed by fully-resolved leaf paths — add {leaf_path!r} to it. "
        f"The bracketed prior-dict syntax ('{prefix}.{rest_dotted}[*]', etc.) is only "
        f"for the fitter prior dict, not evaluate()."
    )


def sample_prior(
    forward_model,
    prior: dict | Callable,
    applicable: dict[str, set[str]],
    leaves: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Sample ``prior`` and return values keyed by resolved parameter paths.

    Mapping priors support fixed values, distributions, and tied parameters. A bare
    key creates one shared value for all targeted observations; ``[*]`` and ``[obs]``
    keys create observation-specific values. Callable priors are evaluated for every
    parameter leaf and create sites named ``"forward.<leaf_path>"`` when they return
    distributions.

    Args:
        forward_model: Model whose parameter leaves define the required paths.
        prior: Prior mapping, leaf callable, or zero-argument callable factory.
        applicable: Observation names applicable to each parameter namespace. This is
            used only for mapping priors.
        leaves: Optional precomputed result from :func:`_enumerate_leaves`.

    Returns:
        A flat mapping from fully resolved parameter paths to sampled or fixed values.

    Raises:
        KeyError: If a mapping entry does not target any parameter leaf.
        ValueError: If a tied parameter cannot be resolved to a source value.
    """
    if callable(prior):
        return _sample_callable_prior(forward_model, _normalise_callable_prior(prior))

    if leaves is None:
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

    return samples


def _sample_callable_prior(forward_model, leaf_callable: Callable) -> dict[str, Any]:
    """Apply ``leaf_callable`` to every parameter leaf and return the resolved values.

    Distribution results create ``"forward.<leaf_path>"`` sample sites. Other
    results are converted to JAX arrays without creating sites.
    """
    _, params_state, _ = nnx.split(forward_model, nnx.Param, nnx.Not(nnx.Param))
    params_pure = nnx.to_pure_dict(params_state)
    _sample_leaves(params_pure, leaf_callable, prefix="", site_prefix=_SITE_PREFIX)
    return dict(_iter_pure_dict_items(params_pure))
