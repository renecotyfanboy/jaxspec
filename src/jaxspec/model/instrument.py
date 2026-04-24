from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import jax.numpy as jnp

from flax import nnx
from jax.typing import ArrayLike

from ._parametrizable import ParametrizableMixin


class GainModel(nnx.Module):
    """Generic gain model."""

    @abstractmethod
    def __call__(self, observation_name: str, *, params: dict | None = None) -> Callable: ...


class ConstantGain(GainModel):
    """A constant gain model.

    The gain factor prior is provided via the unified prior dict under the
    key ``"instrument.gain.factor"``, which may be a ``Distribution`` or a
    :class:`~jaxspec.fit.PerObs` wrapper.
    """

    def __call__(self, observation_name: str, *, params: dict | None = None) -> Callable:
        key = f"instrument.gain.factor.{observation_name}"
        factor = params.get(key, jnp.asarray(1.0)) if params else jnp.asarray(1.0)
        return lambda energy: factor


class ShiftModel(nnx.Module):
    """Generic shift model."""

    @abstractmethod
    def __call__(self, observation_name: str, *, params: dict | None = None) -> Callable: ...


class ConstantShift(ShiftModel):
    """A constant shift model.

    The shift offset prior is provided via the unified prior dict under the
    key ``"instrument.shift.offset"``, which may be a ``Distribution`` or a
    :class:`~jaxspec.fit.PerObs` wrapper.
    """

    def __call__(self, observation_name: str, *, params: dict | None = None) -> Callable:
        key = f"instrument.shift.offset.{observation_name}"
        offset = params.get(key, jnp.asarray(0.0)) if params else jnp.asarray(0.0)
        return lambda energy: energy + offset


class InstrumentModel(ParametrizableMixin, nnx.Module):
    """Encapsulate an instrument model, built as a combination of a shift and gain model.

    Parameters:
        reference_observation_name: The observation to use as a reference.
        gain_model: The gain model.
        shift_model: The shift model.
    """

    prior_prefix: str = "instrument."

    def __init__(
        self,
        reference_observation_name: str,
        gain_model: GainModel | None = None,
        shift_model: ShiftModel | None = None,
    ):
        self.reference = reference_observation_name
        self.gain_model = gain_model
        self.shift_model = shift_model

    def _default_skip_observation(self) -> str | None:
        return self.reference

    def __call__(
        self,
        observation_names: list[str],
        *,
        params: dict | None = None,
    ) -> dict[str, tuple[Callable | None, Callable | None]]:
        """Return per-observation ``(gain_fn, shift_fn)`` tuples.

        Parameters:
            observation_names: All observation names (including the reference).
            params: Flat dict of sampled instrument params, with per-obs values
                keyed as ``instrument.{param}.{obs_name}`` (from
                :meth:`register_priors`).

        Returns:
            ``{obs_name: (gain_fn | None, shift_fn | None)}`` for every
            observation. The reference observation gets ``(None, None)``.
        """
        # Unstack (n_non_ref,) arrays into per-obs keys for subcomponents
        non_ref = [n for n in observation_names if n != self.reference]
        unstacked: dict[str, ArrayLike] = {}
        if params is not None:
            for key, value in params.items():
                for obs_name in non_ref:
                    unstacked[f"{key}.{obs_name}"] = value[obs_name]

        out: dict[str, tuple[Callable | None, Callable | None]] = {}
        for name in observation_names:
            if name == self.reference:
                out[name] = (None, None)
                continue
            gain_fn = (
                self.gain_model(name, params=unstacked) if self.gain_model is not None else None
            )
            shift_fn = (
                self.shift_model(name, params=unstacked) if self.shift_model is not None else None
            )
            out[name] = (gain_fn, shift_fn)
        return out
