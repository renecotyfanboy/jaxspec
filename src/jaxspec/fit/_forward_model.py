from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from flax import nnx
from jax.typing import ArrayLike

from ..data import ObsConfiguration
from ..data.obsconf import to_jax_matrix
from ..model.abc import HideUnderscoreMixin, ModelComponent, SpectralModel
from ..model.background import BackgroundModel
from ..model.instrument import InstrumentModel

if TYPE_CHECKING:
    pass


#: Reused per-obs identity instrument for observations the user didn't
#: configure. ``InstrumentModel()`` with ``gain=None`` and ``shift=None`` has
#: no ``nnx.Param`` leaves, so ``apply_shift`` / ``apply_gain`` are no-ops and
#: ``fold(...)`` reduces to ``transfer_matrix @ spectrum``. Module-level
#: singleton to keep allocations out of the JIT-traced obs loop.
_IDENTITY_INSTRUMENT = InstrumentModel()


def _validate_energy_grid(energy_grid: ArrayLike) -> jnp.ndarray:
    """Validate a user-supplied energy grid and return it as a ``jnp.ndarray``.

    The grid is the edges over which the spectral model gets evaluated, then
    redistributed onto each instrument's native grid by
    :meth:`~jaxspec.model.instrument.InstrumentModel.fold`. We require:

    * 1-D with at least 2 edges (the model integrates over ``[edge_i, edge_{i+1}]``);
    * strictly increasing (the redistribution path uses
      :func:`jnp.interp` which assumes a monotonic ``xp``);
    * strictly positive (energies are in keV).
    """
    arr = np.asarray(energy_grid)
    if arr.ndim != 1:
        raise ValueError(f"energy_grid must be 1-D, got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(f"energy_grid must have at least 2 points, got {arr.size}.")
    if not bool((arr[1:] > arr[:-1]).all()):
        raise ValueError("energy_grid must be strictly increasing.")
    if not bool((arr > 0).all()):
        raise ValueError("energy_grid values must be strictly positive.")
    return jnp.asarray(arr)


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
    """Parametric nnx tree + per-obs caches + a deterministic ``evaluate``.

    Only parameters and parametric submodules live on the nnx tree.
    Non-parametric state (xarray observations, response caches, settings) is
    held off the nnx tree on ``_aux`` — these Python objects aren't
    pytree-friendly and don't belong in nnx's Variable tracking.

    Parameters live as ``nnx.Param`` leaves under three dict-of-modules attributes:

    - ``spectrum``: ``{obs_name: SpectralModel}`` — one cloned replica per
      observation, so per-obs spectral params become natural nnx leaves at
      ``spectrum.<obs>.<path>``.
    - ``instrument``: ``{obs_name: InstrumentModel}`` — only observations
      with a non-``None`` entry in the user's ``instrument_model`` arg.
    - ``background``: ``{obs_name: BackgroundModel}`` — singleton expanded
      to per-obs clones, or the per-obs dict as supplied; ``None`` entries
      dropped.

    Parameters:
        spectral_model: The spectral model template; cloned per observation. A
            single bare component (e.g. ``Powerlaw()``) is accepted too and is
            auto-wrapped via
            [`SpectralModel.from_component`][jaxspec.model.abc.SpectralModel.from_component].
        observations: One or more observation configurations. Accepts a single
            [`ObsConfiguration`][jaxspec.data.obsconf.ObsConfiguration], a list
            (auto-named ``data_0``, ``data_1``, ...), or a ``{name: obs}`` dict.
        background_model: ``None``, a singleton ``BackgroundModel`` (applied to
            every observation as a clone), or a ``{obs_name: BackgroundModel | None}``
            dict for per-obs heterogeneous backgrounds.
        instrument_model: ``None``, or a ``{obs_name: InstrumentModel | None}``
            dict. ``None`` entries (and observations missing from the dict)
            apply the identity fold.
        sparsify_matrix: Whether to store transfer matrices as sparse BCOO.
        n_points: Number of quadrature points per energy bin for the flux
            integration.
        energy_grid: Optional shared 1-D array of energy bin edges (keV,
            strictly increasing) on which to evaluate the spectral model before
            redistributing onto each observation's native grid via
            [`InstrumentModel.fold`][jaxspec.model.instrument.InstrumentModel.fold].
            When ``None`` (default) each observation evaluates the spectrum on
            its own native grid.
    """

    def __init__(
        self,
        spectral_model: SpectralModel | ModelComponent,
        observations: ObsConfiguration | list | dict,
        background_model: BackgroundModel | dict[str, BackgroundModel | None] | None = None,
        instrument_model: dict[str, InstrumentModel | None] | None = None,
        sparsify_matrix: bool = False,
        n_points: int = 2,
        energy_grid: ArrayLike | None = None,
    ):
        # Accept a bare component (e.g. ``Powerlaw()``) where a SpectralModel is
        # expected — wrap it so it gains flux_func / branch topology and its params
        # nest under the conventional ``<component>_1`` name (e.g. ``powerlaw_1``).
        if isinstance(spectral_model, ModelComponent):
            spectral_model = SpectralModel.from_component(spectral_model)

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

        validated_grid = _validate_energy_grid(energy_grid) if energy_grid is not None else None

        # Non-parametric state lives OFF the nnx tree (plain attributes on
        # ForwardModel itself would still be tracked; stash them on the orchestrator).
        # These are exposed to the BayesianModel via the public accessors below.
        #
        # ``spectrum_shared`` is a hint set by :class:`BayesianModel` after it
        # inspects the prior dict; ``evaluate`` reads it to decide whether a
        # user-supplied ``energy_grid`` enables the eval-once-then-fold fast
        # path. Defaults to ``False`` so direct ForwardModel use is correct
        # by default (no fast path; one spectral eval per obs).
        self._aux = _ForwardModelAux(
            observations=obs_dict,
            caches={
                name: _build_obs_cache(obs, self.instrument.get(name), sparse=sparsify_matrix)
                for name, obs in obs_dict.items()
            },
            settings={
                "sparse": sparsify_matrix,
                "n_points": n_points,
                "energy_grid": validated_grid,
                "spectrum_shared": False,
            },
        )

        # Background models with caches (e.g. SpectralModelBackground transfer matrix,
        # BackgroundWithError per-bin shape) need their per-obs cache before any
        # JAX trace runs over their __call__.
        for name, bg in self.background.items():
            bg._set_obs_cache(obs_dict[name], sparse=sparsify_matrix)

    # ----- Non-parametric state accessors (read-through to self._aux) -----

    @property
    def observations(self) -> dict[str, ObsConfiguration]:
        """The name-keyed observation configurations, ``{obs_name: ObsConfiguration}``."""
        return self._aux.observations

    @property
    def settings(self) -> dict[str, Any]:
        """Evaluation settings (``sparse``, ``n_points``, ``energy_grid``, ``spectrum_shared``)."""
        return self._aux.settings

    @property
    def obs_caches(self) -> dict[str, dict[str, Any]]:
        """Per-observation JAX-typed response caches (transfer matrix and components)."""
        return self._aux.caches

    # ----- Unified evaluation entry point -----

    def evaluate(
        self,
        inputs: dict[str, ArrayLike],
        *,
        split_branches: bool = False,
        with_background: bool = True,
        missing_key_style: str = "inputs",
    ) -> dict[str, dict[str, Any]]:
        """Bind ``inputs`` and run the per-observation forward pass.

        ``inputs`` is a flat ``{leaf_path: value}`` dict keyed by nnx leaf
        paths (e.g. ``"spectrum.PN.powerlaw_1.norm"``,
        ``"instrument.MOS1.gain.factor"``, ``"background.PN.countrate"``).
        Every ``nnx.Param`` leaf of the tree must be covered; a miss raises a
        ``KeyError`` with the rich ``_missing_prior_message``.

        The method is **deterministic** — no numpyro sites are created here.
        Callers that need numpyro sampling build the inputs dict via
        ``sample_prior`` first (which is what
        [`BayesianModel.numpyro_model`][jaxspec.fit.BayesianModel.numpyro_model]
        does), then call ``evaluate``. Non-sampling callers (``fakeit``,
        posterior-predictive checks) build the inputs dict from concrete values
        and call ``evaluate`` directly, vmapping over batch dimensions.

        When ``settings["energy_grid"]`` is set, the spectral model is
        evaluated over that grid and redistributed onto each obs's native
        grid by [`InstrumentModel.fold`][jaxspec.model.instrument.InstrumentModel.fold].
        With ``settings["spectrum_shared"]`` additionally ``True`` the grid
        evaluation happens **once** and is broadcast to every obs (the
        BayesianModel sets this flag when no per-obs spectral prior is
        present). When ``energy_grid`` is ``None`` each obs evaluates the
        spectrum on its own (instrument-shifted) native grid.

        Parameters:
            inputs: Flat leaf-path → value dict covering every
                ``nnx.Param`` leaf of the tree.
            split_branches: If ``True``, the per-obs ``"source"`` entry is a
                ``{branch_name: folded_flux}`` pytree instead of a summed
                array. Used by posterior-predictive checks that overlay each
                component.
            with_background: If ``False``, skip the background evaluation and
                set each ``"background"`` entry to ``None``. Used by the
                source-only component overlay to avoid computing a rate it
                discards.
            missing_key_style: Internal selector for the "missing leaf" error
                wording. ``"inputs"`` (default) points direct callers at the
                resolved leaf path; the fitter passes ``"prior"`` to suggest the
                prior-dict key forms.

        Returns:
            ``{obs_name: {"source": folded_flux | {branch: folded_flux},
            "background": background_rate | None}}``. The background entry
            is ``None`` for obs without a background model.
        """
        from ._prior_resolution import bind_inputs

        bound = bind_inputs(self, inputs, missing_key_style=missing_key_style)
        settings = self.settings
        n_points = settings["n_points"]
        energy_grid = settings["energy_grid"]
        spectrum_shared = settings.get("spectrum_shared", False)

        # Slice the user grid once — it is identical for every obs. Fast path:
        # when the spectrum is also shared across every obs, evaluate the
        # spectral model once and reuse the result.
        shared_flux = None
        eval_energies = None
        if energy_grid is not None:
            e_low = energy_grid[:-1]
            e_high = energy_grid[1:]
            eval_energies = jnp.stack([e_low, e_high])
            if spectrum_shared:
                any_replica = next(iter(bound.spectrum.values()))
                shared_flux = any_replica.flux_func(
                    e_low, e_high, n_points=n_points, return_branches=split_branches
                )

        predictions: dict[str, dict[str, Any]] = {}
        for obs_name, obs in self.observations.items():
            cache = self.obs_caches[obs_name]
            inst = bound.instrument.get(obs_name, _IDENTITY_INSTRUMENT)

            if eval_energies is not None:
                if shared_flux is not None:
                    flux = shared_flux
                else:
                    spec = bound.spectrum[obs_name]
                    flux = spec.flux_func(
                        e_low, e_high, n_points=n_points, return_branches=split_branches
                    )
                # Shift is applied *inside* fold when eval_energies is provided
                # (see InstrumentModel.fold's contract).
                source_flux = inst.fold(flux, cache, eval_energies=eval_energies)
            else:
                spec = bound.spectrum[obs_name]
                shifted = inst.apply_shift(cache["in_energies"])
                flux = spec.flux_func(
                    shifted[0], shifted[1], n_points=n_points, return_branches=split_branches
                )
                source_flux = inst.fold(flux, cache)

            if with_background:
                bg = bound.background.get(obs_name)
                bg_rate = bg(obs) if bg is not None else None
            else:
                bg_rate = None
            predictions[obs_name] = {"source": source_flux, "background": bg_rate}

        return predictions


class _ForwardModelAux:
    """Non-pytree container for the per-observation Python objects that don't
    belong on the nnx tree (xarray datasets, pre-built caches, plain dicts).

    Stashing them here instead of as direct ``ForwardModel`` attributes keeps
    them out of nnx's Variable tracking.
    """

    __slots__ = ("caches", "observations", "settings")

    def __init__(self, observations, caches, settings):
        self.observations = observations
        self.caches = caches
        self.settings = settings
