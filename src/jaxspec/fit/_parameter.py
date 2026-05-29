from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax.typing import ArrayLike


class TiedParameter:
    """Declare that a parameter is deterministically derived from another.

    Parameters:
        tied_to: Full dotted-path key of the source parameter
            (e.g. ``"spectrum.powerlaw_1.alpha"``). May include an ``[obs]``
            suffix to reference a specific per-observation sample
            (e.g. ``"instrument.gain.factor[MOS1]"``).
        func: A callable ``f(source_value) -> derived_value``.

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
    """Sample a single multivariate site and return a per-leaf value lookup.

    Use from inside a factory-callable prior to give several parameters a
    correlated joint draw (the dict form is strictly per-leaf, so joint
    sampling has to drop into the callable form). The returned lookup
    function maps a leaf path to the pre-sampled component value when the
    leaf matches one of ``components``, and ``None`` otherwise — letting the
    caller chain it with structural defaults via ``or``.

    Parameters:
        components: Ordered tuple of user-facing parameter keys (without
            ``[obs]`` suffix). The k-th component of ``joint_dist``'s sample
            is bound to ``components[k]``.
        joint_dist: A multivariate distribution whose ``event_shape[-1]``
            equals ``len(components)``.
        name: Optional numpyro site name; defaults to ``"+"``-joined components.

    Example::

        def my_prior_factory():
            alpha_norm = joint_prior_factory(
                components=("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
                joint_dist=dist.MultivariateNormal(
                    loc=jnp.array([2.0, 1e-4]),
                    covariance_matrix=jnp.array([[0.5, 1e-5], [1e-5, 1e-8]]),
                ),
            )

            def prior(path, shape):
                return alpha_norm(path) or _structural_defaults(path, shape)
            return prior

        fitter = MCMCFitter(spectral_model, observations, prior=my_prior_factory, ...)

    The implementation samples ``joint_dist`` exactly once (under the supplied
    ``name``), then returns the per-component scalar slice on match. The leaf
    callable contract in :func:`~jaxspec.fit._bayesian_model._sample_leaves`
    treats array-like returns as pre-sampled values: they are written straight
    to the leaf with no extra numpyro.sample site.
    """
    site_name = name or "+".join(components)
    sample = numpyro.sample(site_name, joint_dist)

    component_to_idx = {comp: i for i, comp in enumerate(components)}

    def lookup(leaf_path: str) -> ArrayLike | None:
        # The callable form of ``sample_prior`` calls the prior callable with
        # paths like ``"spectrum.<obs>.powerlaw_1.alpha"`` (the obs segment is
        # part of the nnx tree shape). Strip it to match user-facing paths.
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
    """Wrap a dict-form prior as a leaf callable.

    Useful for hybrid setups: cover the common cases with dict-style entries
    and the special ones with custom callable logic. The returned function
    returns ``None`` on miss (so callers can chain with ``or``) — unlike the
    framework's internal dict adapter which raises ``KeyError`` on miss.

    Per the leaf-callable contract: the return value is either a
    :class:`numpyro.distributions.Distribution` (registered as a numpyro
    sample site by :func:`~jaxspec.fit._bayesian_model._sample_leaves`), a
    pre-sampled array (written straight to the leaf, no site), or ``None``
    on miss.

    Must be called from INSIDE the numpyro trace (typically from inside a
    factory-callable prior) since shared entries get sampled here.

    Example::

        def my_prior_factory():
            covered = dict_prior({
                "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
                "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
            })

            def prior(path, shape):
                return covered(path, shape) or _structural_defaults(path, shape)
            return prior

    The dict resolution order is the same as the framework's: explicit
    ``[obs]`` > ``[*]`` > shared > miss.
    """
    from ._prior_resolution import _split_nnx_leaf, parse_prior_key

    # Pre-sample shared entries once.
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
