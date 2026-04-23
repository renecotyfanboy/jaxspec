import numpyro.distributions as dist

from jax.typing import ArrayLike

# TODO Put this at top level so every subpackage can use it ?

PriorValueType = dist.Distribution | ArrayLike | float
PriorDictType = dict[str, PriorValueType]
