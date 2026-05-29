from collections.abc import Callable

import numpyro.distributions as dist

from jax.typing import ArrayLike

from ..fit._parameter import TiedParameter

PriorValueType = dist.Distribution | ArrayLike | float | TiedParameter
PriorDictType = dict[str, PriorValueType]

"""
Leaf-form prior callable: receives ``(nnx_leaf_path, shape)`` and returns
either a :class:`numpyro.distributions.Distribution` (registered as a
numpyro sample site) or any array-like value (written directly to the leaf
with no numpyro site, useful for pre-sampled / deterministic values).
Consumed by :func:`~jaxspec.fit._prior_resolution.sample_prior` (callable
form).
"""
LeafCallable = Callable[[str, tuple[int, ...]], dist.Distribution | ArrayLike]

"""
Factory-form prior callable: a zero-arg factory that, when invoked inside
``numpyro_model``'s trace context, returns a :data:`LeafCallable`. Use this
shape when the leaf callable needs to first sample shared / hierarchical
hyperparameters via :func:`numpyro.sample`.
"""
FactoryCallable = Callable[[], LeafCallable]

PriorType = PriorDictType | LeafCallable | FactoryCallable
