from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jax.typing import ArrayLike


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

    def _apply_shift(self, energies: ArrayLike) -> ArrayLike:
        """Apply :attr:`shift` to ``energies`` and clip away non-positive values."""
        if self.shift is None:
            return energies
        return jnp.clip(self.shift(energies), min=1e-6)

    def _apply_gain(self, flux, energies: ArrayLike):
        """Multiply ``flux`` (or each branch in a pytree) by :attr:`gain`'s factor."""
        if self.gain is None:
            return flux
        factor = jnp.clip(self.gain(energies), min=0.0)
        return jax.tree.map(lambda f: f * factor, flux)

    def fold(
        self,
        observation,
        cache: dict,
        spectral_model,
        *,
        n_points: int = 2,
        split_branches: bool = False,
    ):
        """Return expected counts in folded space (or a per-branch pytree).

        Parameters:
            observation: The :class:`~jaxspec.data.ObsConfiguration` for this
                pointing (used for energy-grid metadata).
            cache: Per-observation JAX-typed views built by
                :class:`~jaxspec.fit._forward_model.ForwardModel`. Always
                contains ``"transfer_matrix"``; also contains
                ``"redistribution"``, ``"grouping"``, ``"area"``, ``"exposure"``
                when :attr:`requires_components` is ``True``.
            spectral_model: The per-obs spectral model replica (already
                parameter-bound).
            n_points: Quadrature points per energy bin for flux integration.
            split_branches: If ``True``, return a pytree with one folded counts
                array per additive branch of the spectral model.
        """
        energies = self._apply_shift(np.asarray(observation.in_energies))

        flux = spectral_model.flux_func(
            *energies, n_points=n_points, return_branches=split_branches
        )
        flux = self._apply_gain(flux, energies)

        return jax.tree.map(lambda f: jnp.clip(cache["transfer_matrix"] @ f, min=1e-6), flux)
