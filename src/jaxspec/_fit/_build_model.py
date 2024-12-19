from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro

from jax.experimental.sparse import BCOO
from jax.typing import ArrayLike
from numpyro.distributions import Distribution

if TYPE_CHECKING:
    from ..data import ObsConfiguration
    from ..model.abc import SpectralModel
    from ..util.typing import PriorDictType


def forward_model(
    model: "SpectralModel",
    parameters,
    obs_configuration: "ObsConfiguration",
    sparse=False,
):
    energies = np.asarray(obs_configuration.in_energies)

    if sparse:
        # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
        transfer_matrix = BCOO.from_scipy_sparse(
            obs_configuration.transfer_matrix.data.to_scipy_sparse().tocsr()
        )

    else:
        transfer_matrix = np.asarray(obs_configuration.transfer_matrix.data.todense())

    expected_counts = transfer_matrix @ model.photon_flux(parameters, *energies)

    # The result is clipped at 1e-6 to avoid 0 round-off and diverging likelihoods
    return jnp.clip(expected_counts, a_min=1e-6)


def build_prior(prior: "PriorDictType", expand_shape: tuple = (), prefix=""):
    """
    Transform a dictionary of prior distributions into a dictionary of parameters sampled from the prior.
    Must be used within a numpyro model.
    """
    parameters = {}

    for key, value in prior.items():
        # Split the key to extract the module name and parameter name
        module_name, param_name = key.rsplit("_", 1)
        if isinstance(value, Distribution):
            parameters[key] = jnp.ones(expand_shape) * numpyro.sample(
                f"{prefix}{module_name}_{param_name}", value
            )

        elif isinstance(value, ArrayLike):
            parameters[key] = jnp.ones(expand_shape) * value

        else:
            raise ValueError(
                f"Invalid prior type {type(value)} for parameter {prefix}{module_name}_{param_name} : {value}"
            )

    return parameters
