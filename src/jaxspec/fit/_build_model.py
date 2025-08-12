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
    from jaxspec.data import ObsConfiguration
    from jaxspec.model.abc import SpectralModel
    from jaxspec.util.typing import PriorDictType


class TiedParameter:
    def __init__(self, tied_to, func):
        self.tied_to = tied_to
        self.func = func


def forward_model(
    model: "SpectralModel",
    parameters,
    obs_configuration: "ObsConfiguration",
    sparse=False,
    gain: Callable | None = None,
    shift: Callable | None = None,
    split_branches: bool = False,
    n_points: int | None = 2,
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
        expected_counts = transfer_matrix @ (
            model.photon_flux(parameters, *energies, n_points=n_points) * factor
        )
        return jnp.clip(expected_counts, min=1e-6)  # Ensure the expected counts are positive

    else:
        model_flux = model.photon_flux(
            parameters, *energies, split_branches=True, n_points=n_points
        )
        return jax.tree.map(
            lambda f: jnp.clip(transfer_matrix @ (f * factor), min=1e-6), model_flux
        )


def build_prior(prior: "PriorDictType", expand_shape: tuple = (), prefix=""):
    """
    Transform a dictionary of prior distributions into a dictionary of parameters sampled from the prior.
    Must be used within a numpyro model.
    """
    parameters = {}
    params_to_tie = {}

    for key, value in prior.items():
        # Split the key to extract the module name and parameter name
        module_name, param_name = key.rsplit("_", 1)

        if isinstance(value, Distribution):
            parameters[key] = jnp.ones(expand_shape) * numpyro.sample(
                f"{prefix}{module_name}_{param_name}", value
            )

        elif isinstance(value, TiedParameter):
            params_to_tie[key] = value

        elif isinstance(value, ArrayLike):
            parameters[key] = jnp.ones(expand_shape) * value

        else:
            raise ValueError(
                f"Invalid prior type {type(value)} for parameter {prefix}{module_name}_{param_name} : {value}"
            )

    for key, value in params_to_tie.items():
        func_to_apply = value.func
        tied_to = value.tied_to
        parameters[key] = func_to_apply(parameters[tied_to])

    return parameters
