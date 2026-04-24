import numpyro.distributions as dist

from jax.typing import ArrayLike

from ..fit._parameter import PerObs, TiedParameter

# TODO Put this at top level so every subpackage can use it ?

PriorValueType = dist.Distribution | ArrayLike | float | PerObs | TiedParameter
PriorDictType = dict[str, PriorValueType]
