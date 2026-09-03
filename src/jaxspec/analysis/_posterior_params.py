"""Translate posterior arrays into user-facing prior-key mappings.

The sampler records sites under the naming grammar owned by
:mod:`jaxspec.fit._prior_resolution` — a bare path for a shared entry, and
``"forward.<prefix>.<obs>.<rest>"`` for a per-observation one. Users think in prior-dict
keys instead. This module is the inverse map: it reads an
:class:`~arviz.InferenceData` posterior and reassembles it into
``{prior_key: array}``, applying the same scope and tie semantics the sampler used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import jax.numpy as jnp
import numpyro.distributions as dist

from jax.typing import ArrayLike

from ..fit._parameter import TiedParameter
from ..fit._prior_resolution import (
    _enumerate_leaves,
    _per_obs_site_name,
    _prefix_to_obs_names,
    _resolve_targets,
    parse_prior_key,
)

if TYPE_CHECKING:
    from ..fit import BayesianModel


def leaf_inputs(forward_model, input_parameters: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
    """Convert user-path ``input_parameters`` into the flat leaf-path inputs dict that
    :meth:`~jaxspec.fit._forward_model.ForwardModel.evaluate` consumes.

    Inverts the broadcasting :func:`build_input_parameters` applies: shared params
    (broadcast to ``(..., n_obs)``) get sliced per obs, ``[*]`` stacks (same shape) get
    sliced per obs by index, and ragged ``{obs: array}`` entries get looked up by name.
    The result is keyed by nnx leaf paths (``"<prefix>.<obs>.<rest>"``) and every
    ``nnx.Param`` leaf of the forward model is covered — missing keys would surface as a
    :func:`~jaxspec.fit._prior_resolution.bind_inputs` ``KeyError``.
    """
    leaves = _enumerate_leaves(forward_model)
    prefix_to_obs = _prefix_to_obs_names(forward_model)

    inputs: dict[str, ArrayLike] = {}
    for user_path, by_obs in leaves.items():
        value = input_parameters.get(user_path)
        if value is None:
            continue
        prefix = user_path.split(".", 1)[0]
        obs_order = prefix_to_obs[prefix]
        for obs_name, leaf_path in by_obs.items():
            if isinstance(value, dict):
                if obs_name in value:
                    inputs[leaf_path] = value[obs_name]
            else:
                obs_idx = obs_order.index(obs_name)
                inputs[leaf_path] = value[..., obs_idx]
    return inputs


def build_input_parameters(
    bayesian_fitter: BayesianModel, inference_data: az.InferenceData
) -> dict[str, ArrayLike]:
    """Reassemble the posterior into ``{prior_key: array}``.

    Backs :attr:`~jaxspec.analysis.results.FitResult.input_parameters`; see there for the
    user-facing contract.
    """
    fm = bayesian_fitter.forward_model
    effective_prior = bayesian_fitter._effective_prior

    posterior = az.extract(inference_data, combined=False)
    chain_draw = (posterior.sizes["chain"], posterior.sizes["draw"])

    prefix_to_obs = _prefix_to_obs_names(fm)
    leaves = _enumerate_leaves(fm)
    needed = _needed_posterior_names(effective_prior, prefix_to_obs, leaves)
    data_vars = {name: jnp.asarray(posterior[name].data) for name in needed}

    by_base: dict[str, dict[str | None, Any]] = {}
    for raw_key, value in effective_prior.items():
        base, scope = parse_prior_key(raw_key)
        by_base.setdefault(base, {})[scope] = value

    out: dict[str, ArrayLike] = {}
    deferred_ties: list[tuple[str, str | None, TiedParameter, list[str]]] = []

    for base, scopes in by_base.items():
        prefix = base.split(".", 1)[0]
        obs_axis = prefix_to_obs.get(prefix, [])

        if None in scopes:
            value = _resolve_shared_entry(
                scopes[None], base, obs_axis, data_vars, chain_draw, deferred_ties
            )
            if value is not None:
                out[base] = value
        else:
            value = _resolve_per_obs_entry(
                scopes,
                base,
                obs_axis,
                data_vars,
                chain_draw,
                deferred_ties,
                owning_obs=set(leaves.get(base, {})),
            )
            if value is not None:
                out[base] = value

    _apply_tied_resolutions(out, deferred_ties, prefix_to_obs)
    return out


def _needed_posterior_names(effective_prior, prefix_to_obs, leaves) -> set[str]:
    """Return the posterior site names required to reconstruct input parameters.

    Shared distribution entries contribute their bare path. Scoped distribution entries
    contribute one per-observation site for each matching leaf. Fixed values and tied
    parameters do not contribute sample sites. Wildcard scopes include only observations
    that own the corresponding leaf.
    """
    needed: set[str] = set()
    applicable = {prefix: set(obs_names) for prefix, obs_names in prefix_to_obs.items()}

    for raw_key, value in effective_prior.items():
        if not isinstance(value, dist.Distribution):
            continue
        base, scope = parse_prior_key(raw_key)
        if scope is None:
            needed.add(base)
            continue
        for obs, _leaf in _resolve_targets(base, scope, leaves, applicable):
            needed.add(_per_obs_site_name(base, obs))

    return needed


def _resolve_shared_entry(
    value, base, obs_axis, data_vars, chain_draw, deferred_ties
) -> ArrayLike | None:
    """Materialise a shared (unscoped) prior entry as an obs-broadcast array.

    Returns ``None`` and appends to ``deferred_ties`` when the entry is a
    ``TiedParameter`` — the caller should skip this base for now.
    """
    if isinstance(value, TiedParameter):
        deferred_ties.append((base, None, value, obs_axis))
        return None
    if isinstance(value, dist.Distribution):
        arr = data_vars[base]
    else:
        fixed = jnp.asarray(value)
        arr = jnp.broadcast_to(fixed, (*chain_draw, *fixed.shape))
    return jnp.broadcast_to(arr[..., None], (*arr.shape, len(obs_axis)))


def _resolve_per_obs_entry(
    scopes, base, obs_axis, data_vars, chain_draw, deferred_ties, owning_obs=None
) -> ArrayLike | dict | None:
    """Materialise scoped entries, stacking complete uniform observation sets.

    Ties are deferred until their sources are available. ``owning_obs`` restricts a
    wildcard to observations that contain the requested leaf. Partial or ragged sets
    remain name-keyed dictionaries.
    """
    per_obs: dict[str, ArrayLike] = {}
    has_ties = False
    for obs in obs_axis:
        value = scopes.get(obs)
        if value is None and (owning_obs is None or obs in owning_obs):
            value = scopes.get("*")
        if value is None:
            continue
        if isinstance(value, TiedParameter):
            deferred_ties.append((base, obs, value, obs_axis))
            has_ties = True
            continue
        if isinstance(value, dist.Distribution):
            per_obs[obs] = data_vars[_per_obs_site_name(base, obs)]
        else:
            fixed = jnp.asarray(value)
            per_obs[obs] = jnp.broadcast_to(fixed, (*chain_draw, *fixed.shape))

    if has_ties:
        return per_obs
    if not per_obs:
        return None
    return _compact_per_obs(per_obs, obs_axis)


def _compact_per_obs(per_obs: dict, obs_axis: list[str]) -> ArrayLike | dict:
    """Stack complete uniform observation mappings; keep partial or ragged mappings."""
    shapes = {arr.shape for arr in per_obs.values()}
    if len(shapes) == 1 and len(per_obs) == len(obs_axis):
        return jnp.stack([per_obs[obs] for obs in obs_axis], axis=-1)
    return per_obs


def _apply_tied_resolutions(out, deferred_ties, prefix_to_obs) -> None:
    """Resolve every deferred TiedParameter once all direct entries are in ``out``.

    Mirrors the sampling-time semantics of
    :func:`~jaxspec.fit._prior_resolution._resolve_tied_entry`: a bare or
    ``[obs]``-scoped source provides one value for every destination, a
    ``[*]`` source pairs each destination obs with its same-obs draw. Per-obs
    tied values are merged into the base's ``{obs: array}`` staging dict (next
    to its direct entries) and compacted afterwards.
    """
    touched: set[str] = set()
    for dest_base, dest_obs, tied, obs_axis in deferred_ties:
        src_base, src_scope = parse_prior_key(tied.tied_to)
        entry = out.get(src_base)
        if entry is None:
            raise ValueError(
                f"TiedParameter {dest_base!r} references unknown source {tied.tied_to!r}"
            )
        src_axis = prefix_to_obs.get(src_base.split(".", 1)[0], [])

        def pick(obs, entry=entry, src_axis=src_axis):
            if isinstance(entry, dict):
                value = entry.get(obs)
            elif obs in src_axis:
                value = entry[..., src_axis.index(obs)]
            else:
                value = None
            if value is None:
                raise ValueError(
                    f"TiedParameter {dest_base!r} cannot match a source value for "
                    f"observation {obs!r}: tied_to={tied.tied_to!r}."
                )
            return value

        if dest_obs is None:
            if src_scope is None:
                if isinstance(entry, dict):
                    out[dest_base] = {obs: tied.func(v) for obs, v in entry.items()}
                else:
                    out[dest_base] = tied.func(entry)
            else:
                anchor = src_scope if src_scope != "*" else sorted(obs_axis)[0]
                value = tied.func(pick(anchor))
                out[dest_base] = jnp.broadcast_to(value[..., None], (*value.shape, len(obs_axis)))
            continue

        if src_scope == "*":
            source = pick(dest_obs)
        elif src_scope is None:
            source = pick(dest_obs) if isinstance(entry, dict) else entry[..., 0]
        else:
            source = pick(src_scope)
        staged = out.setdefault(dest_base, {})
        staged[dest_obs] = tied.func(source)
        touched.add(dest_base)

    for base in touched:
        value = out[base]
        if isinstance(value, dict):
            obs_axis = prefix_to_obs.get(base.split(".", 1)[0], [])
            out[base] = _compact_per_obs(value, obs_axis)
