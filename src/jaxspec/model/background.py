from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist

from flax import nnx

from ..data.obsconf import to_jax_matrix

if TYPE_CHECKING:
    from ..data import ObsConfiguration


class BackgroundModel(nnx.Module):
    """Base class for background models.

    A background model is created *per observation*. It predicts a count rate
    in the *background* energy space; the orchestrator scales it by
    ``folded_backratio`` before adding to the source-space likelihood.

    Subclasses implement :meth:`__call__` (the deterministic forward
    prediction) and optionally :meth:`default_prior` for data-dependent
    defaults that get merged into the unified prior dict.
    """

    #: Whether the background contributes a Poisson likelihood term on the
    #: observed background spectrum. ``False`` means the background is treated
    #: as a fixed deterministic quantity (e.g. :class:`SubtractedBackground`).
    is_stochastic: bool = True

    @abstractmethod
    def __call__(self, observation: ObsConfiguration):
        """Return the predicted background rate in background space (no backratio)."""

    def _set_obs_cache(self, observation: ObsConfiguration, *, sparse: bool) -> None:
        """Pre-build any JAX-typed caches the model needs for this observation.

        Default: no-op. Subclasses that fold a spectral model through the
        response (e.g. :class:`SpectralModelBackground`) override this to
        populate caches eagerly, before any JAX trace runs over their
        :meth:`__call__`.
        """

    def default_prior(self, observation: ObsConfiguration, obs_name: str) -> dict:
        """Return data-dependent default priors scoped to this obs.

        Subclasses override to inject defaults (e.g.
        :class:`BackgroundWithError`'s observed-counts Gamma prior). Keys must
        be ``[obs_name]``-scoped (e.g. ``"background.countrate[PN]"``). User
        prior entries override these defaults.
        """
        return {}

    def user_path(self, nnx_path: str) -> str:
        """Map an internal nnx leaf path to the user-facing prior-key path.

        Default identity. Subclasses that wrap inner modules (e.g.
        :class:`SpectralModelBackground`) override this to hide internal
        wrapper segments from the user-facing key.
        """
        return nnx_path


class SubtractedBackground(BackgroundModel):
    """
    Return the observed background unchanged.

    !!! danger
        This is not a good way to model the background, as it does not account for the
        fact that the measured background is a Poisson realization of the true
        background's countrate. Prefer :class:`BackgroundWithError`.
    """

    is_stochastic: bool = False

    def __call__(self, observation):
        return jnp.asarray(observation.folded_background.data)


class BackgroundWithError(BackgroundModel):
    """Fit an independent countrate per background bin using a Gamma prior.

    For each bin, the default prior is ``Gamma(observed_counts + 1, rate=1)``,
    which is conjugate to the Poisson likelihood and peaks near the observed
    value. This default can be overridden by providing a
    ``"background.countrate[obs_name]"`` entry in the unified prior dict.
    """

    def __init__(self):
        self.countrate = nnx.Param(jnp.asarray(0.0))

    def _set_obs_cache(self, observation, *, sparse: bool) -> None:
        # Initialise countrate to the observed background counts so the nnx
        # default matches the Gamma prior's mode if a user forgets to provide one.
        self.countrate = nnx.Param(jnp.asarray(observation.folded_background.data) + 1.0)

    def default_prior(self, observation, obs_name: str) -> dict:
        return {
            f"background.countrate[{obs_name}]": dist.Gamma(
                jnp.asarray(observation.folded_background.data) + 1.0, rate=1.0
            )
        }

    def __call__(self, observation):
        return self.countrate[...]


class SpectralModelBackground(BackgroundModel):
    """Model the background as a spectral model folded through the instrument response.

    The inner ``spectral_model`` is evaluated against the observation's
    ``folded_background`` spectrum, using the same transfer matrix and energy
    grid as the source. Prior keys use dotted paths relative to the inner
    spectral model (e.g. ``"background.powerlaw_1.alpha"``), provided in the
    unified prior dict.

    Each per-obs instance carries its own transfer-matrix cache, populated by
    :meth:`_set_obs_cache` at :class:`~jaxspec.fit._forward_model.ForwardModel`
    construction time.

    Parameters:
        spectral_model: The spectral model describing the background shape.
        sparse: Whether to use sparse transfer matrices for the background
            convolution.
    """

    def __init__(self, spectral_model, sparse: bool = False):
        self.spectral_model = spectral_model
        self.sparse = sparse
        self._tm = nnx.data(jnp.zeros((1, 1)))

    def user_path(self, nnx_path: str) -> str:
        """Strip the internal ``spectral_model.`` wrapper segment so user prior
        keys can be ``"background.powerlaw_1.alpha"`` instead of the verbose
        ``"background.spectral_model.powerlaw_1.alpha"``."""
        prefix = "spectral_model."
        return nnx_path[len(prefix) :] if nnx_path.startswith(prefix) else nnx_path

    def _set_obs_cache(self, observation, *, sparse: bool) -> None:
        """Build this instance's transfer-matrix cache for ``observation``.

        The bundled ``sparse`` flag from :class:`~jaxspec.fit._forward_model.ForwardModel`
        takes precedence over the constructor flag so the background folding
        matches the source folding's storage choice.
        """
        self._tm = nnx.data(to_jax_matrix(observation.transfer_matrix.data, sparse=sparse))

    def __call__(self, observation):
        import numpy as np

        energies = np.asarray(observation.in_energies)
        flux = self.spectral_model.photon_flux(*energies)
        return jnp.clip(self._tm @ flux, min=1e-6)
