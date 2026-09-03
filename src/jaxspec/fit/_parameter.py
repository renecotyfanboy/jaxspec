from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax.typing import ArrayLike


class TiedParameter:
    """Define a parameter as a deterministic transformation of another parameter.

    Args:
        tied_to: Dotted prior key for the source parameter. An optional ``[obs]``
            or ``[*]`` suffix selects an observation-specific source.
        func: Transformation applied to the resolved source value.

    Example::

        prior = {
            "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
            "spectrum.powerlaw_2.alpha": TiedParameter(
                "spectrum.powerlaw_1.alpha", lambda x: 0.5 * x
            ),
        }
    """

    def __init__(self, tied_to: str, func):
        self.tied_to = tied_to
        self.func = func


def joint_prior_factory(
    components: Sequence[str],
    joint_dist: dist.Distribution,
    *,
    name: str | None = None,
) -> Callable[[str], ArrayLike | None]:
    """Draw a joint sample and return a lookup for its parameter components.

    Call this function inside a callable-prior factory. The returned lookup accepts
    a parameter leaf path and returns its component of the joint sample, or ``None``
    when the path is not listed in ``components``.

    Args:
        components: Ordered user-facing parameter paths. Component ``k`` receives
            ``joint_dist`` sample ``[..., k]``.
        joint_dist: Distribution whose final event dimension has one entry per component.
        name: NumPyro site name. Defaults to the component paths joined by ``"+"``.

    Returns:
        A ``lookup(leaf_path)`` callable for the sampled components.

    Example::

        def prior_factory():
            joint = joint_prior_factory(
                ("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
                joint_dist,
            )

            def prior(path, shape):
                value = joint(path)
                return value if value is not None else default_prior(path, shape)

            return prior

    Check explicitly for ``None``: matched values may be traced JAX arrays and cannot
    be used as booleans.
    """
    site_name = name or "+".join(components)
    sample = numpyro.sample(site_name, joint_dist)

    component_to_idx = {comp: i for i, comp in enumerate(components)}

    def lookup(leaf_path: str) -> ArrayLike | None:
        parts = leaf_path.split(".")
        if len(parts) >= 3:
            stripped = f"{parts[0]}.{'.'.join(parts[2:])}"
            if stripped in component_to_idx:
                return sample[..., component_to_idx[stripped]]
        if leaf_path in component_to_idx:
            return sample[..., component_to_idx[leaf_path]]
        return None

    return lookup


def _materialise_prior_value(value):
    """Pass a Distribution through unchanged; otherwise convert to a jnp array."""
    if isinstance(value, dist.Distribution):
        return value
    return jnp.asarray(value)


def dict_prior(
    prior_dict: dict,
) -> Callable[[str, tuple], dist.Distribution | ArrayLike | None]:
    """Return a leaf-prior callable backed by ``prior_dict``.

    Call this function inside a callable-prior factory. Shared distributions are
    sampled immediately and reused across observations. Observation-specific and
    wildcard entries return their distribution or fixed value when looked up. Missing
    paths return ``None``.

    Args:
        prior_dict: Prior values keyed by bare, ``[*]``, or ``[obs]`` parameter paths.

    Returns:
        A ``lookup(leaf_path, shape)`` callable returning a distribution, sampled or
        fixed array value, or ``None`` when no entry matches.

    Example::

        def my_prior_factory():
            covered = dict_prior({
                "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
                "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
            })

            def prior(path, shape):
                value = covered(path, shape)
                return value if value is not None else default_prior(path, shape)

            return prior

    Keep scopes disjoint. This helper does not validate overlaps and resolves them in
    ``[obs]``, ``[*]``, then bare-key order. Also check results explicitly for ``None``;
    matched values may be traced JAX arrays and cannot be used as booleans.
    """
    from ._prior_resolution import _split_nnx_leaf, parse_prior_key

    shared: dict[str, object] = {}
    for raw_key, value in prior_dict.items():
        path, scope = parse_prior_key(raw_key)
        if scope is None:
            if isinstance(value, dist.Distribution):
                shared[path] = numpyro.sample(path, value)
            else:
                shared[path] = jnp.asarray(value)

    def lookup(leaf_path: str, shape):
        try:
            base, obs = _split_nnx_leaf(leaf_path)
        except ValueError:
            return None
        for key in (f"{base}[{obs}]", f"{base}[*]"):
            if key in prior_dict:
                return _materialise_prior_value(prior_dict[key])
        if base in shared:
            return shared[base]
        return None

    return lookup
