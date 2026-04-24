from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jax.experimental.sparse import BCOO
from jax.typing import ArrayLike

from ..data import ObsConfiguration
from ..model.abc import HideUnderscoreMixin, SpectralModel

if TYPE_CHECKING:
    from ..model.background import BackgroundModel
    from ..model.instrument import InstrumentModel


def _normalise_observations(
    observations: ObsConfiguration | list[ObsConfiguration] | dict[str, ObsConfiguration],
) -> dict[str, ObsConfiguration]:
    """Return a name-keyed dict of observations regardless of user input shape."""
    if isinstance(observations, dict):
        return observations
    if isinstance(observations, list):
        return {f"data_{i}": obs for i, obs in enumerate(observations)}
    if isinstance(observations, ObsConfiguration):
        return {"data": observations}
    raise ValueError(f"Invalid type for observations : {type(observations)}")


def _build_transfer_matrix(obs: ObsConfiguration, sparse: bool = False):
    if sparse:
        return BCOO.from_scipy_sparse(obs.transfer_matrix.data.to_scipy_sparse().tocsr())
    return jnp.asarray(obs.transfer_matrix.data.todense())


class ForwardModel(HideUnderscoreMixin, nnx.Module):
    """Deterministic forward model.

    Owns a :class:`~jaxspec.model.abc.SpectralModel`, one or more
    :class:`~jaxspec.data.ObsConfiguration` objects, pre-built transfer
    matrices, and — optionally — a
    :class:`~jaxspec.model.background.BackgroundModel` and a
    :class:`~jaxspec.model.instrument.InstrumentModel`. Given parameter dicts,
    it produces expected counts (source, background, total) per observation.

    Parameters:
        spectral_model: The spectral model whose photon flux is folded through
            the instrument response.
        observations: One or more observation configurations. Accepts a single
            :class:`~jaxspec.data.ObsConfiguration`, a list, or a
            ``{name: obs}`` dict.
        background_model: Optional background model used to predict the
            background rate per observation.
        instrument_model: Optional instrument calibration model providing
            per-observation gain / shift callables.
        sparsify_matrix: Whether to store transfer matrices as sparse BCOO.
        n_points: Number of quadrature points per energy bin for the flux
            integration.
    """

    settings: dict[str, Any]

    def __init__(
        self,
        spectral_model: SpectralModel,
        observations: ObsConfiguration | list | dict,
        background_model: BackgroundModel | None = None,
        instrument_model: InstrumentModel | None = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
    ):
        self.spectrum = spectral_model
        self.observations = nnx.data(_normalise_observations(observations))
        self.background_model = background_model
        self.instrument_model = instrument_model
        self.settings = {"sparse": sparsify_matrix, "n_points": n_points}
        self._transfer_matrices = nnx.data(
            {
                name: _build_transfer_matrix(obs, sparse=sparsify_matrix)
                for name, obs in self.observations.items()
            }
        )

    @cached_property
    def parameter_names(self) -> list[str]:
        """Sorted list of dotted paths to every leaf ``nnx.Param`` owned by this
        module (e.g. ``"spectrum.powerlaw_1.alpha"``)."""
        _, param_state, _ = nnx.split(self, nnx.Param, ...)
        pure = nnx.to_pure_dict(param_state)

        def _walk(prefix: str, d: dict) -> list[str]:
            out: list[str] = []
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out.extend(_walk(path, v))
                else:
                    out.append(path)
            return out

        return sorted(_walk("", pure))

    def _build_gain_shift(
        self, instrument_params: dict | None
    ) -> dict[str, tuple[Callable | None, Callable | None]]:
        """Return ``{obs_name: (gain_fn, shift_fn)}`` for every observation."""
        obs_names = list(self.observations.keys())
        if self.instrument_model is None:
            return {name: (None, None) for name in obs_names}
        return self.instrument_model(obs_names, params=instrument_params)

    @staticmethod
    def _simulate_one(
        spectral_model: SpectralModel,
        obs: ObsConfiguration,
        transfer_matrix,
        gain: Callable | None,
        shift: Callable | None,
        n_points: int,
        split_branches: bool,
    ):
        energies = np.asarray(obs.in_energies)
        energies = shift(energies) if shift is not None else energies
        energies = jnp.clip(energies, min=1e-6)
        factor = gain(energies) if gain is not None else 1.0
        factor = jnp.clip(factor, min=0.0)

        if not split_branches:
            expected_counts = transfer_matrix @ (
                spectral_model.turbo_flux(*energies, n_points=n_points) * factor
            )
            return jnp.clip(expected_counts, min=1e-6)

        model_flux = spectral_model.turbo_flux(*energies, n_points=n_points, return_branches=True)
        return jax.tree.map(
            lambda f: jnp.clip(transfer_matrix @ (f * factor), min=1e-6), model_flux
        )

    def __call__(
        self,
        params: dict | None = None,
        *,
        instrument_params: dict | None = None,
        split_branches: bool = False,
    ) -> dict[str, ArrayLike]:
        """Compute expected source counts for every observation.

        Parameters:
            params: Optional dotted-path parameter dict for the spectral model.
                Values may be scalar (shared across observations) or have a
                leading axis of length ``n_obs`` (per-observation). If
                ``None``, the parameters currently stored on the module are
                used.
            instrument_params: Optional instrument-calibration parameter dict
                (as produced by :meth:`InstrumentModel.register_priors`). If
                the forward model owns an :attr:`instrument_model`, these are
                converted internally into per-observation gain / shift
                callables.
            split_branches: If ``True``, return per-branch folded counts
                (one entry per additive branch of the spectral model).

        Returns:
            ``{obs_name: expected_source_counts}`` dict.
        """
        counts: dict[str, ArrayLike] = {}
        n_points = self.settings["n_points"]
        gain_shift = self._build_gain_shift(instrument_params)

        for name, obs in self.observations.items():
            params_i = (
                {key: value[name] for key, value in params.items()} if params is not None else None
            )
            spectral_model = self.spectrum._with_params(params_i)
            gain, shift = gain_shift[name]
            counts[name] = self._simulate_one(
                spectral_model,
                obs,
                self._transfer_matrices[name],
                gain,
                shift,
                n_points,
                split_branches,
            )
        return counts

    def expected_counts(
        self,
        source_params: dict | None = None,
        instrument_params: dict | None = None,
        background_params: dict | None = None,
    ) -> dict[str, dict[str, ArrayLike]]:
        """Compose source, instrument and background into per-observation counts.

        Parameters:
            source_params: Optional spectral-model parameter dict (same
                convention as :meth:`__call__`).
            instrument_params: Optional instrument-calibration parameter dict.
            background_params: Optional background-model parameter dict (as
                produced by :meth:`BackgroundModel.register_priors`).

        Returns:
            ``{obs_name: {"source", "background_rate",
            "background_in_obs_space", "total"}}``. ``background_rate`` is in
            background energy space (pre-``folded_backratio``);
            ``background_in_obs_space`` is ``background_rate *
            obs.folded_backratio.data``; ``total`` is ``source +
            background_in_obs_space``. When no background model is attached,
            ``background_rate`` and ``background_in_obs_space`` are ``0.0``.
        """
        source_counts = self(params=source_params, instrument_params=instrument_params)

        out: dict[str, dict[str, ArrayLike]] = {}
        for name, obs in self.observations.items():
            source = source_counts[name]
            if self.background_model is None:
                bkg_rate: ArrayLike = jnp.asarray(0.0)
                bkg_in_obs: ArrayLike = jnp.asarray(0.0)
            else:
                if background_params is not None:
                    bkg_params_i = {key: value[name] for key, value in background_params.items()}
                else:
                    bkg_params_i = None
                bkg_rate = self.background_model(obs, name=name, params=bkg_params_i)
                bkg_in_obs = bkg_rate * obs.folded_backratio.data
            out[name] = {
                "source": source,
                "background_rate": bkg_rate,
                "background_in_obs_space": bkg_in_obs,
                "total": source + bkg_in_obs,
            }
        return out
