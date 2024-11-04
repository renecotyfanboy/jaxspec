from collections.abc import Callable
from typing import TYPE_CHECKING

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from jax.experimental.sparse import BCOO
from jax.typing import ArrayLike
from numpyro.distributions import Distribution, Poisson

if TYPE_CHECKING:
    from ..data import ObsConfiguration
    from ..model.abc import SpectralModel
    from ..util.typing import PriorDictType


class CountForwardModel(hk.Module):
    """
    A haiku module which allows to build the function that simulates the measured counts
    """

    # TODO: It has no point of being a haiku module, it should be a simple function

    def __init__(self, model: "SpectralModel", folding: "ObsConfiguration", sparse=False):
        super().__init__()
        self.model = model
        self.energies = jnp.asarray(folding.in_energies)

        if (
            sparse
        ):  # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
            self.transfer_matrix = BCOO.from_scipy_sparse(
                folding.transfer_matrix.data.to_scipy_sparse().tocsr()
            )

        else:
            self.transfer_matrix = jnp.asarray(folding.transfer_matrix.data.todense())

    def __call__(self, parameters):
        """
        Compute the count functions for a given observation.
        """

        expected_counts = self.transfer_matrix @ self.model.photon_flux(parameters, *self.energies)

        return jnp.clip(expected_counts, a_min=1e-6)


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


def build_numpyro_model_for_single_obs(
    obs,
    model,
    background_model,
    name: str = "",
    sparse: bool = False,
) -> Callable:
    """
    Build a numpyro model for a given observation and spectral model.
    """

    def numpyro_model(prior_params, observed=True):
        # Return the expected countrate for a set of parameters
        obs_model = jax.jit(lambda par: forward_model(model, par, obs, sparse=sparse))
        countrate = obs_model(prior_params)

        # Handle the background model
        if (getattr(obs, "folded_background", None) is not None) and (background_model is not None):
            bkg_countrate = background_model.numpyro_model(
                obs, model, name="bkg_" + name, observed=observed
            )

        elif (getattr(obs, "folded_background", None) is None) and (background_model is not None):
            raise ValueError(
                "Trying to fit a background model but no background is linked to this observation"
            )

        else:
            bkg_countrate = 0.0

        # Register the observed value
        # This is the case where we fit a model to a TOTAL spectrum as defined in OGIP standard
        with numpyro.plate("obs_plate_" + name, len(obs.folded_counts)):
            numpyro.sample(
                "obs_" + name,
                Poisson(countrate + bkg_countrate / obs.folded_backratio.data),
                obs=obs.folded_counts.data if observed else None,
            )

    return numpyro_model


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
