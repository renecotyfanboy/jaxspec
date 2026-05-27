from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from flax import nnx
from jax.typing import ArrayLike

from ..data import ObsConfiguration
from ..data.obsconf import to_jax_matrix
from ..model.abc import HideUnderscoreMixin, SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel

if TYPE_CHECKING:
    pass


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


def _build_obs_cache(
    obs: ObsConfiguration, instrument: InstrumentModel | None, *, sparse: bool
) -> dict[str, Any]:
    """Pre-build per-observation JAX-typed response views for one observation.

    Always builds ``"transfer_matrix"``. Additionally builds the un-merged
    components (``redistribution``, ``grouping``, ``area``, ``exposure``) when
    the per-obs instrument declares :attr:`InstrumentModel.requires_components`.
    """
    cache: dict[str, Any] = {
        "transfer_matrix": to_jax_matrix(obs.transfer_matrix.data, sparse=sparse),
        "in_energies": jnp.asarray(obs.in_energies),
    }

    if instrument is not None and instrument.requires_components:
        cache["redistribution"] = to_jax_matrix(obs.redistribution.data, sparse=sparse)
        cache["grouping"] = to_jax_matrix(obs.grouping.data, sparse=sparse)
        cache["area"] = jnp.asarray(obs.area.data)
        cache["exposure"] = jnp.asarray(obs.exposure.data)
    return cache


def _normalise_background(
    background_model: BackgroundModel | dict[str, BackgroundModel | None] | None,
    obs_names: list[str],
) -> dict[str, BackgroundModel]:
    """Singleton → cloned per-obs dict; dict → as-is (drop None entries); None → empty."""
    if background_model is None:
        return {}
    if isinstance(background_model, BackgroundModel):
        return {name: nnx.clone(background_model) for name in obs_names}
    return {name: bg for name, bg in background_model.items() if bg is not None}


def _normalise_instrument(
    instrument_model: dict[str, InstrumentModel | None] | None,
) -> dict[str, InstrumentModel]:
    """Drop ``None`` entries — those observations get the identity fold."""
    if instrument_model is None:
        return {}
    return {name: m for name, m in instrument_model.items() if m is not None}


def _validate_obs_keys(user_dict: dict, obs_names: list[str], *, model_kind: str) -> None:
    """Raise if ``user_dict`` has keys that don't match any observation name.

    Catches typos before the silent-drop in ``_normalise_*`` would discard the
    user's configuration. Only applied when the user passes a dict (singleton
    ``BackgroundModel`` / ``None`` skip this entirely).
    """
    unknown = [k for k in user_dict if k not in obs_names]
    if unknown:
        raise ValueError(
            f"{model_kind} contains keys {unknown!r} that are not in the "
            f"observation set {obs_names!r}. Keys must match observation names "
            f"(auto-generated as 'data_0', 'data_1', ... for list inputs; the "
            f"dict key for dict inputs; or 'data' for a single ObsConfiguration)."
        )


class ForwardModel(HideUnderscoreMixin, nnx.Module):
    """Pure parametric nnx tree consumed by :func:`~jaxspec.fit._bayesian_model._bind_priors`.

    Only parameters and parametric submodules live here. Non-parametric state
    (xarray observations, response caches, settings) is held off the nnx tree
    on :attr:`_aux` — these Python objects aren't pytree-friendly and don't
    belong in nnx's Variable tracking.

    Parameters live as ``nnx.Param`` leaves under three dict-of-modules attributes:

    - :attr:`spectrum`: ``{obs_name: SpectralModel}`` — one cloned replica per
      observation, so per-obs spectral params become natural nnx leaves at
      ``spectrum.<obs>.<path>``.
    - :attr:`instrument`: ``{obs_name: InstrumentModel}`` — only observations
      with a non-``None`` entry in the user's ``instrument_model`` arg.
    - :attr:`background`: ``{obs_name: BackgroundModel}`` — singleton expanded
      to per-obs clones, or the per-obs dict as supplied; ``None`` entries
      dropped.

    Parameters:
        spectral_model: The spectral model template; cloned per observation.
        observations: One or more observation configurations. Accepts a single
            :class:`~jaxspec.data.ObsConfiguration`, a list (auto-named
            ``data_0``, ``data_1``, ...), or a ``{name: obs}`` dict.
        background_model: ``None``, a singleton ``BackgroundModel`` (applied to
            every observation as a clone), or a ``{obs_name: BackgroundModel | None}``
            dict for per-obs heterogeneous backgrounds.
        instrument_model: ``None``, or a ``{obs_name: InstrumentModel | None}``
            dict. ``None`` entries (and observations missing from the dict)
            apply the identity fold.
        sparsify_matrix: Whether to store transfer matrices as sparse BCOO.
        n_points: Number of quadrature points per energy bin for the flux
            integration.
    """

    def __init__(
        self,
        spectral_model: SpectralModel,
        observations: ObsConfiguration | list | dict,
        background_model: BackgroundModel | dict[str, BackgroundModel | None] | None = None,
        instrument_model: dict[str, InstrumentModel | None] | None = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
        energy_grid: ArrayLike | None = None,
    ):
        obs_dict = _normalise_observations(observations)
        obs_names = list(obs_dict)

        # Catch typos in user-supplied per-obs dicts before normalisation
        # silently drops the misspelled entries.
        if isinstance(instrument_model, dict):
            _validate_obs_keys(instrument_model, obs_names, model_kind="instrument_model")
        if isinstance(background_model, dict):
            _validate_obs_keys(background_model, obs_names, model_kind="background_model")

        instrument_dict = _normalise_instrument(instrument_model)
        background_dict = _normalise_background(background_model, obs_names)

        self.spectrum = nnx.data({name: nnx.clone(spectral_model) for name in obs_dict})
        self.instrument = nnx.data(instrument_dict)
        self.background = nnx.data(background_dict)

        # Non-parametric state lives OFF the nnx tree (plain attributes on
        # ForwardModel itself would still be tracked; stash them on the orchestrator).
        # These are exposed to the BayesianModel via the public accessors below.
        self._aux = _ForwardModelAux(
            observations=obs_dict,
            caches={
                name: _build_obs_cache(obs, self.instrument.get(name), sparse=sparsify_matrix)
                for name, obs in obs_dict.items()
            },
            settings={"sparse": sparsify_matrix, "n_points": n_points, "energy_grid": energy_grid},
        )

        # Background models with caches (e.g. SpectralModelBackground transfer matrix,
        # BackgroundWithError per-bin shape) need their per-obs cache before any
        # JAX trace runs over their __call__.
        for name, bg in self.background.items():
            bg._set_obs_cache(obs_dict[name], sparse=sparsify_matrix)

    # ----- Non-parametric state accessors (read-through to self._aux) -----

    @property
    def observations(self) -> dict[str, ObsConfiguration]:
        return self._aux.observations

    @property
    def settings(self) -> dict[str, Any]:
        return self._aux.settings

    @property
    def obs_caches(self) -> dict[str, dict[str, Any]]:
        return self._aux.caches


class _ForwardModelAux:
    """Non-pytree container for the per-observation Python objects that don't
    belong on the nnx tree (xarray datasets, pre-built caches, plain dicts).

    Stashing them here instead of as direct ``ForwardModel`` attributes keeps
    them out of nnx's Variable tracking.
    """

    __slots__ = ("observations", "caches", "settings")

    def __init__(self, observations, caches, settings):
        self.observations = observations
        self.caches = caches
        self.settings = settings
