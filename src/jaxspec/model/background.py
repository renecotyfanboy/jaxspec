from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist

from flax import nnx

from ._parametrizable import ParametrizableMixin

if TYPE_CHECKING:
    from ..data import ObsConfiguration


class BackgroundModel(ParametrizableMixin, nnx.Module):
    """Base class for background models.

    A background model predicts a count rate in the *background* space and
    carries enough information to be Poisson-fitted against an observation's
    ``folded_background``. The orchestrator
    (:class:`~jaxspec.fit.BayesianModel`) scales the predicted rate by
    ``folded_backratio`` before adding it to the source-space likelihood.

    Subclasses must implement :meth:`__call__` (the deterministic forward
    prediction) and declare their parameters via the unified prior dict under
    the ``"background."`` prefix.
    """

    prior_prefix: str = "background."

    #: Whether the background contributes a Poisson likelihood term on the
    #: observed background spectrum. ``False`` means the background is treated
    #: as a fixed deterministic quantity (e.g. :class:`SubtractedBackground`).
    is_stochastic: bool = True

    @abstractmethod
    def __call__(self, observation: ObsConfiguration, *, name: str, params: dict | None = None):
        """Return the predicted background rate in background space (no backratio).

        Parameters:
            observation: The observation configuration for this pointing.
            name: The observation name (used to namespace parameter keys).
            params: Optional parameter dict keyed by full dotted-path site
                names (e.g. ``"background.countrate"``). Values are scalar
                (pre-sliced for this observation by the orchestrator).
        """


class SubtractedBackground(BackgroundModel):
    """
    Return the observed background unchanged.

    !!! danger
        This is not a good way to model the background, as it does not account for the
        fact that the measured background is a Poisson realization of the true
        background's countrate. Prefer :class:`BackgroundWithError`.
    """

    is_stochastic: bool = False

    def __call__(self, observation, *, name: str, params: dict | None = None):
        return jnp.asarray(observation.folded_background.data)


class BackgroundWithError(BackgroundModel):
    """Fit an independent countrate per background bin using a Gamma prior.

    For each bin, the default prior is ``Gamma(observed_counts + 1, rate=1)``,
    which is conjugate to the Poisson likelihood and peaks near the observed
    value. This default can be overridden by providing a ``"background.countrate"``
    key in the unified prior dict.
    """

    def default_prior(self, observations: dict) -> dict:
        """Construct per-observation Gamma priors from observed background counts."""
        from ..fit._parameter import PerObs

        return {
            "background.countrate": PerObs(
                {
                    name: dist.Gamma(jnp.asarray(obs.folded_background.data) + 1.0, rate=1.0)
                    for name, obs in observations.items()
                }
            )
        }

    def __call__(self, observation, *, name: str, params: dict | None = None):
        params: dict = params or {}
        countrate = params.get("background.countrate")

        if countrate is None:
            raise ValueError("No countrate prior provided for BackgroundWithError")
        return countrate


class SpectralModelBackground(BackgroundModel):
    """Model the background as a spectral model folded through the instrument response.

    The inner ``spectral_model`` is evaluated against the observation's
    ``folded_background`` spectrum, using the same transfer matrix and energy
    grid as the source. Prior keys use dotted paths relative to the inner
    spectral model (e.g. ``"background.powerlaw_1.alpha"``), provided in the
    unified prior dict.

    Parameters:
        spectral_model: The spectral model describing the background shape.
        sparse: Whether to use sparse transfer matrices for the background
            convolution.
    """

    def __init__(
        self,
        spectral_model,
        sparse: bool = False,
    ):
        self.spectral_model = spectral_model
        self.sparse = sparse

    def __call__(self, observation, *, name: str, params: dict | None = None):
        import numpy as np

        from jax.experimental.sparse import BCOO

        energies = np.asarray(observation.in_energies)
        if self.sparse:
            transfer_matrix = BCOO.from_scipy_sparse(
                observation.transfer_matrix.data.to_scipy_sparse().tocsr()
            )
        else:
            transfer_matrix = np.asarray(observation.transfer_matrix.data.todense())
        flux = self.spectral_model.photon_flux(*energies, params=params)
        return jnp.clip(transfer_matrix @ flux, min=1e-6)
