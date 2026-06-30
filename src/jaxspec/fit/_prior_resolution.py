"""Prior sampling + leaf binding for the unified inputs-dict interface.

The :class:`~jaxspec.fit._forward_model.ForwardModel` evaluation contract is a
flat ``{leaf_path: value}`` dict ("inputs"). This module owns two operations
that bridge that contract with numpyro:

* :func:`sample_prior` — turn the user's prior (dict-of-distributions form,
  callable form, or any mix of fixed values / :class:`TiedParameter` /
  :class:`~numpyro.distributions.Distribution`) into the inputs dict, creating
  the numpyro sample sites along the way.
* :func:`bind_inputs` — bind an already-sampled inputs dict onto the forward
  model's nnx tree, deterministically. No numpyro sites, no ``nnx.clone`` so
  it stays ``jax.vmap``-safe.

The split lets non-sampling callers (``fakeit``, posterior-predictive checks)
reuse :meth:`~jaxspec.fit._forward_model.ForwardModel.evaluate` without
spinning a numpyro trace.
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

#: First dotted segment of every prior key. Used to validate that user keys
#: target one of the three model layers in the forward model.
_KNOWN_PREFIXES = ("spectrum", "instrument", "background")


def bind_inputs(forward_model, inputs, *, missing_key_style: str = "inputs"):
    """Bind a leaf-path → value ``inputs`` dict onto a bound copy of
    ``forward_model``'s nnx tree and return it.

    This is the deterministic half of prior binding: it creates **no** numpyro
    sites and does **no** ``nnx.clone``, mirroring
    :meth:`~jaxspec.model.abc.SpectralModel._with_params` so it stays
    ``jax.vmap``-safe. The per-trace isolation that MCMC needs (to avoid
    cross-trace ``UnexpectedTracerError``) is the caller's responsibility — it
    clones the forward model once per trace before calling
    :meth:`~jaxspec.fit._forward_model.ForwardModel.evaluate`.

    Every ``nnx.Param`` leaf must have a matching key in ``inputs``; a miss
    raises the rich :func:`_missing_prior_message` so a forgotten prior surfaces
    loudly (same strict contract the old adapter enforced at bind time).
    ``missing_key_style`` picks that message's wording: ``"inputs"`` (default,
    for direct ``evaluate`` / ``fakeit`` callers) points at the resolved leaf
    path, while ``"prior"`` (the fitter flow) suggests the prior-dict key forms.
    """
    graph_def, params_state, other_state = nnx.split(forward_model, nnx.Param, nnx.Not(nnx.Param))
    params_pure = nnx.to_pure_dict(params_state)

    def _lookup(leaf_path, _shape):
        try:
            return inputs[leaf_path]
        except KeyError:
            raise KeyError(_missing_prior_message(leaf_path, style=missing_key_style)) from None

    _sample_leaves(params_pure, _lookup, prefix="", site_prefix="forward.")

    nnx.replace_by_pure_dict(params_state, params_pure)
    return nnx.merge(graph_def, params_state, other_state)


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
      Used by :func:`bind_inputs` to route pre-sampled values from a
      ``{leaf_path: value}`` inputs dict to the nnx tree (no extra sites), and
      by dict-form :func:`sample_prior` to broadcast a single shared draw to
      every per-obs leaf without spawning redundant per-obs sites.
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
    * specific obs               — that obs only.

    ``applicable`` is the prefix → set-of-obs table built by the caller;
    validation has already ensured the requested obs is in that set, so a
    miss here means "no leaf exists for this obs" (e.g. a background path
    that exists on some obs but not others). Returns an empty list on a
    miss — callers raise :func:`_unmatched_key_message` so a typo'd key
    surfaces loudly instead of being silently dropped.
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
    if not dest_targets:
        raise KeyError(_unmatched_key_message(path, scope, leaves))

    if scope is None:
        # Shared dest: compute the derived value once, broadcast to every leaf,
        # register one deterministic site under the bare path.
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
    # Shared or specific: a single value applies to all dests.
    _obs, leaf = src_targets[0]
    value = samples[leaf]
    return lambda _dest_obs: value


def _per_obs_site_name(path: str, obs: str) -> str:
    """Compose the canonical per-obs site name ``"forward.<prefix>.<obs>.<rest>"``."""
    prefix, rest = path.split(".", 1)
    return f"forward.{prefix}.{obs}.{rest}"


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
) -> dict[str, Any]:
    """Produce a ``{leaf_path: value}`` inputs dict by sampling ``prior``.

    This is the sampling half of the old ``resolve_prior``: it creates the
    numpyro sample sites and returns the inputs dict that
    :meth:`~jaxspec.fit._forward_model.ForwardModel.evaluate` will bind onto
    the tree (via :func:`bind_inputs`).

    * Dict form — walk the tree once via :func:`_enumerate_leaves`, sample
      each entry into ``samples[leaf_path]`` (one shared site broadcast to
      every targeted leaf, or one site per leaf for ``[*]`` / ``[obs]``
      scopes), then resolve :class:`TiedParameter` entries.
    * Callable form — walk the tree's leaves via :func:`_sample_leaves` so
      each ``nnx.Param`` leaf gets one ``"forward.<leaf>"`` site.

    Strict coverage is enforced *later* by :func:`bind_inputs`: a leaf with
    no matching inputs entry raises :func:`_missing_prior_message`.

    ``applicable`` is the prefix → set-of-obs table the caller built from
    :meth:`BayesianModel._applicable_obs` (unused in the callable form).
    """
    if callable(prior):
        return _sample_callable_prior(forward_model, _normalise_callable_prior(prior))

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
    """Walk the tree's nnx.Param leaves and sample each one via ``leaf_callable``.

    Mirrors the dict-form output: returns ``{leaf_path: value}``. Each leaf
    that hits a :class:`~numpyro.distributions.Distribution` becomes a
    ``"forward.<leaf>"`` site (same naming the old adapter produced); leaves
    that resolve to plain values are written through unchanged.
    """
    _, params_state, _ = nnx.split(forward_model, nnx.Param, nnx.Not(nnx.Param))
    params_pure = nnx.to_pure_dict(params_state)
    _sample_leaves(params_pure, leaf_callable, prefix="", site_prefix="forward.")
    return dict(_iter_pure_dict_values(params_pure))


def _iter_pure_dict_values(d: dict, prefix: str = "") -> Any:
    """Yield ``(dotted_path, value)`` pairs from a nested pure-dict."""
    for name, item in d.items():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            yield from _iter_pure_dict_values(item, full)
        else:
            yield full, item
