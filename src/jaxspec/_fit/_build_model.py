from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
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
    gain: Callable | None = None,
    shift: Callable | None = None,
    split_branches: bool = False,
):
    energies = np.asarray(obs_configuration.in_energies)

    if sparse:
        # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
        transfer_matrix = BCOO.from_scipy_sparse(
            obs_configuration.transfer_matrix.data.to_scipy_sparse().tocsr()
        )

    else:
        transfer_matrix = np.asarray(obs_configuration.transfer_matrix.data.todense())

    energies = shift(energies) if shift is not None else energies
    energies = jnp.clip(energies, min=1e-6)  # Ensure shifted energies remain positive
    factor = gain(energies) if gain is not None else 1.0
    factor = jnp.clip(factor, min=0.0)  # Ensure the gain is positive to avoid NaNs

    if not split_branches:
        expected_counts = transfer_matrix @ (model.photon_flux(parameters, *energies) * factor)
        return jnp.clip(expected_counts, min=1e-6)  # Ensure the expected counts are positive

    else:
        model_flux = model.photon_flux(parameters, *energies, split_branches=True)
        return jax.tree.map(
            lambda f: jnp.clip(transfer_matrix @ (f * factor), min=1e-6), model_flux
        )


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
