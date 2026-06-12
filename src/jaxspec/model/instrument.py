from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from flax import nnx
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ..data import ObsConfiguration


def redistribute(integrated_spectrum, old_e_low, old_e_high, e_low, e_high):
    # Suppose old_e_high[i] == old_e_low[i+1]
    edges = jnp.concatenate([old_e_low[:1], old_e_high])
    cumflux = jnp.concatenate([jnp.zeros(1), jnp.cumsum(integrated_spectrum)])
    return jnp.interp(e_high, edges, cumflux) - jnp.interp(e_low, edges, cumflux)


class GainModel(nnx.Module):
    """Generic gain model. ``__call__(energies)`` returns the per-energy gain factor."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantGain(GainModel):
    """A scalar gain factor, independent of energy.

    The factor lives as :attr:`factor` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.gain.factor"``
    (shared across instrumented obs) or ``"instrument.gain.factor[*]"`` /
    ``"instrument.gain.factor[obs_name]"`` (per-obs).
    """

    def __init__(self):
        self.factor = nnx.Param(jnp.asarray(1.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return self.factor[...]


class ShiftModel(nnx.Module):
    """Generic shift model. ``__call__(energies)`` returns shifted energies."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantShift(ShiftModel):
    """An additive energy shift, constant across the spectrum.

    The offset lives as :attr:`offset` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.shift.offset"``
    (shared) or ``"instrument.shift.offset[*]"`` / ``"instrument.shift.offset[obs_name]"``
    (per-obs).
    """

    def __init__(self):
        self.offset = nnx.Param(jnp.asarray(0.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return energies + self.offset[...]


class InstrumentModel(nnx.Module):
    """Per-observation instrument response.

    Pass as a dict to :class:`~jaxspec.fit.BayesianModel`::

        BayesianModel(
            spectral_model, prior, observations,
            instrument_model={
                "PN": None, # explicit reference
                "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
                "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            },
        )

    ``None`` entries (or simply omitting an observation) apply the identity
    fold (``transfer_matrix @ flux``) — useful for the reference instrument.

    Parameters:
        gain: Optional :class:`GainModel` (e.g. :class:`ConstantGain`). When
            ``None``, no flux scaling is applied.
        shift: Optional :class:`ShiftModel` (e.g. :class:`ConstantShift`). When
            ``None``, the input energies pass through unchanged.
    """

    #: When ``True``, :class:`~jaxspec.fit._forward_model.ForwardModel` builds
    #: the un-merged response components (``redistribution``, ``grouping``,
    #: ``area``, ``exposure``) into the per-observation cache passed to
    #: :meth:`fold`. Subclasses set this to ``True`` when their math needs the
    #: components separately (e.g. pileup, RMF calibration).
    requires_components = False

    def __init__(self, gain: GainModel | None = None, shift: ShiftModel | None = None):
        self.gain = gain
        self.shift = shift

    def default_prior(self, observation: ObsConfiguration, obs_name: str) -> dict:
        """Return data-dependent default priors scoped to this obs.

        Mirrors :meth:`~jaxspec.model.background.BackgroundModel.default_prior`:
        subclasses (e.g. pileup models with per-obs dead-time / grade-fraction
        parameters) override this to inject ``[obs_name]``-scoped defaults.
        User prior entries override these defaults.
        """
        return {}

    def apply_shift(self, energies: ArrayLike) -> ArrayLike:
        """Apply :attr:`shift` to ``energies`` and clip away non-positive values."""
        if self.shift is None:
            return energies
        return jnp.clip(self.shift(energies), min=1e-6)

    def apply_gain(self, flux, energies: ArrayLike):
        """Multiply ``flux`` (or each branch in a pytree) by :attr:`gain`'s factor."""
        if self.gain is None:
            return flux
        factor = jnp.clip(self.gain(energies), min=0.0)
        return jax.tree.map(lambda f: f * factor, flux)

    def fold(
        self,
        spectrum: ArrayLike,
        cache: dict,
        eval_energies: ArrayLike | None = None,
    ):
        """
        Fold the input spectrum (or branches of a pytree) into the instrument using the pre-computed transfer matrix.
        """

        # Current contract: shift is applied here if a grid is provided, else in
        # the forward model. The spectrum was integrated on the *unshifted*
        # ``eval_energies``, so the shift must go on the native grid: bin i
        # collects the flux from its shifted band, matching the no-grid path
        # (which evaluates flux_func on the shifted native energies directly).
        if eval_energies is not None:
            target_energies = self.apply_shift(cache["in_energies"])
            spectrum = jax.tree.map(
                lambda s: redistribute(s, *eval_energies, *target_energies), spectrum
            )

        spectrum = jax.tree.map(lambda s: self.apply_gain(s, cache["in_energies"]), spectrum)

        return jax.tree.map(lambda s: jnp.clip(cache["transfer_matrix"] @ s, min=1e-6), spectrum)
