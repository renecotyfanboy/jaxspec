import numpyro.distributions as dist

from jax.typing import ArrayLike

PriorDictType = dict[str, dict[str, dist.Distribution | ArrayLike]]
