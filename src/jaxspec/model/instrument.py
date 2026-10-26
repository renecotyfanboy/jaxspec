from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from jax.typing import ArrayLike

from ._mixin import HasDerivedQuantities

if TYPE_CHECKING:
    from ..data import ObsConfiguration


def redistribute(
    integrated_spectrum: ArrayLike,
    old_e_low: ArrayLike,
    old_e_high: ArrayLike,
    e_low: ArrayLike,
    e_high: ArrayLike,
) -> jax.Array:
    """Conservatively map integrated flux onto another set of energy bins.

    Assume constant photon-flux density within each source bin, zero in gaps,
    and zero outside the source coverage. A target bin receives only its overlap
    with source bins: flux outside the target coverage is not renormalized back
    into it. For example, a source bin ``[1, 3]`` with flux ``4`` contributes
    ``2`` to ``[2, 4]``. All energy bounds are in keV and the returned values
    retain the input integrated-flux units (normally photons/cm²/s).

    Source bins must be finite, sorted, non-overlapping and have positive width.
    Target bins may have gaps or arbitrary order; their widths must be
    nonnegative. These value constraints belong to the grid constructors, so
    this function can also accept traced bin edges. Empty inputs are supported.
    Search and cumulative sums avoid allocating a dense bin-overlap matrix.

    Parameters:
        integrated_spectrum: Integrated flux with source bins on the last axis.
        old_e_low: One-dimensional source lower bounds.
        old_e_high: One-dimensional source upper bounds.
        e_low: Target lower bounds, of any shape.
        e_high: Target upper bounds, of the same shape.

    Returns:
        jax.Array: Integrated flux with the spectrum's leading axes followed by
            the target shape. Differentiable in flux and, away from bin
            boundaries, bin edges.
    """
    spectrum = jnp.asarray(integrated_spectrum)
    old_e_low, old_e_high, e_low, e_high = (
        jnp.asarray(edges) for edges in (old_e_low, old_e_high, e_low, e_high)
    )
    if old_e_low.ndim != 1 or old_e_high.ndim != 1:
        raise ValueError("Source energy-bin bounds must be one-dimensional.")
    if old_e_low.shape != old_e_high.shape or e_low.shape != e_high.shape:
        raise ValueError("Lower and upper energy-bin bounds must have matching shapes.")
    if spectrum.ndim < 1 or spectrum.shape[-1] != old_e_low.size:
        raise ValueError("The spectrum's last axis must match the source energy bins.")

    dtype = jnp.result_type(spectrum, old_e_low, old_e_high, e_low, e_high, 1.0)
    if old_e_low.size == 0 or e_low.size == 0:
        return jnp.zeros((*spectrum.shape[:-1], *e_low.shape), dtype=dtype)

    bounds = jnp.stack((e_low, e_high))
    indices = jnp.minimum(jnp.searchsorted(old_e_high, bounds, side="right"), old_e_low.size - 1)
    fractions = jnp.clip(
        (bounds - old_e_low[indices]) / (old_e_high[indices] - old_e_low[indices]),
        0.0,
        1.0,
    )
    lower, upper = indices
    low_fraction, high_fraction = fractions
    cumulative = jnp.concatenate(
        (jnp.zeros((*spectrum.shape[:-1], 1), dtype=spectrum.dtype), jnp.cumsum(spectrum, axis=-1)),
        axis=-1,
    )
    # Keep endpoint pieces separate: subtracting two large cumulative totals
    # would otherwise erase faint bins after a bright one, even on an identity grid.
    interior = cumulative[..., upper] - cumulative[..., lower + 1]
    multi_bin = (
        spectrum[..., lower] * (1 - low_fraction) + interior + spectrum[..., upper] * high_fraction
    )
    same_bin = spectrum[..., lower] * (high_fraction - low_fraction)
    return jnp.where(lower == upper, same_bin, multi_bin)


class GainModel(HasDerivedQuantities, nnx.Module):
    """Generic gain model. ``__call__(energies)`` returns the per-energy gain factor."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantGain(GainModel):
    """A scalar gain factor, independent of energy.

    The factor lives as ``factor`` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.gain.factor"``
    (shared across instrumented obs) or ``"instrument.gain.factor[*]"`` /
    ``"instrument.gain.factor[obs_name]"`` (per-obs).
    """

    def __init__(self):
        self.factor = nnx.Param(jnp.asarray(1.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return self.factor[...]


class ShiftModel(HasDerivedQuantities, nnx.Module):
    """Generic shift model. ``__call__(energies)`` returns shifted energies."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantShift(ShiftModel):
    """An additive energy shift, constant across the spectrum.

    The offset lives as ``offset`` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.shift.offset"``
    (shared) or ``"instrument.shift.offset[*]"`` / ``"instrument.shift.offset[obs_name]"``
    (per-obs).
    """

    def __init__(self):
        self.offset = nnx.Param(jnp.asarray(0.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return energies + self.offset[...]


class InstrumentModel(HasDerivedQuantities, nnx.Module):
    """Per-observation instrument response.

    Pass as a dict to [`BayesianModel`][jaxspec.fit.BayesianModel]

    ```python
    BayesianModel(
        spectral_model, prior, observations,
        instrument_model={
            "PN": None,
            "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
        },
    )
    ```

    ``None`` entries and omitted observations apply the identity fold
    (``transfer_matrix @ flux``), allowing one instrument to serve as the reference.

    Parameters:
        gain: Optional [`GainModel`][jaxspec.model.instrument.GainModel] (e.g. [`ConstantGain`][jaxspec.model.instrument.ConstantGain]). When
            ``None``, no flux scaling is applied.
        shift: Optional [`ShiftModel`][jaxspec.model.instrument.ShiftModel] (e.g. [`ConstantShift`][jaxspec.model.instrument.ConstantShift]). When
            ``None``, the input energies pass through unchanged.
    """

    #: Whether ``fold`` requires separate response components in its cache.
    requires_components = False

    def __init__(self, gain: GainModel | None = None, shift: ShiftModel | None = None):
        self.gain = gain
        self.shift = shift

    def default_prior(self, observation: ObsConfiguration, obs_name: str) -> dict:
        """Return data-dependent default priors scoped to this obs.

        Mirrors [`BackgroundModel.default_prior`][jaxspec.model.background.BackgroundModel.default_prior]:
        subclasses (e.g. pileup models with per-obs dead-time / grade-fraction
        parameters) override this to inject ``[obs_name]``-scoped defaults.
        User prior entries override these defaults.
        """
        return {}

    def apply_shift(self, energies: ArrayLike) -> ArrayLike:
        """Apply ``shift`` to ``energies`` and clip away non-positive values."""
        if self.shift is None:
            return energies
        return jnp.clip(self.shift(energies), min=1e-6)

    def apply_gain(self, flux, energies: ArrayLike):
        """Multiply ``flux`` (or each branch in a pytree) by ``gain``'s factor."""
        if self.gain is None:
            return flux
        factor = jnp.clip(self.gain(energies), min=0.0)
        return jax.tree.map(lambda f: f * factor, flux)

    def _redistribute_to_native(self, spectrum, cache: dict, eval_energies: ArrayLike):
        """Resample ``spectrum`` from ``eval_energies`` onto the response's native grid.

        Shared by every subclass so the shift convention cannot diverge between folds.

        The shift goes on the **target** (native) grid, not the source grid. The spectrum
        was integrated on the *unshifted* ``eval_energies``, so bin ``i`` must collect the
        flux from its shifted band — which matches the no-grid path, where ``flux_func``
        is evaluated on the shifted native energies directly. Shifting the source grid
        instead is the same operation with the opposite sign, and a fit would then
        recover ``-offset``.
        """
        target_energies = self.apply_shift(cache["in_energies"])
        return jax.tree.map(lambda s: redistribute(s, *eval_energies, *target_energies), spectrum)

    def fold(
        self,
        spectrum: ArrayLike,
        cache: dict,
        eval_energies: ArrayLike | None = None,
    ):
        """
        Fold the input spectrum (or branches of a pytree) into the instrument using the pre-computed transfer matrix.
        """

        if eval_energies is not None:
            spectrum = self._redistribute_to_native(spectrum, cache, eval_energies)

        spectrum = jax.tree.map(lambda s: self.apply_gain(s, cache["in_energies"]), spectrum)

        return jax.tree.map(lambda s: jnp.clip(cache["transfer_matrix"] @ s, min=1e-6), spectrum)


class PileupModel(InstrumentModel):
    """Per-observation instrument response including CCD photon **pile-up**.

    The `jaxspec` counterpart of `XSPEC`
    [`pileup`](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPileup.html)
    convolution model ([Davis 2001](https://iopscience.iop.org/article/10.1086/323488/pdf)) which
    describes the mixing of photons that reach the same detector region within a
    single CCD frame, the dominant calibration effect for bright point sources on
    instruments such as Chandra/ACIS. [`PileupModel`][jaxspec.model.instrument.PileupModel] extends
    [`InstrumentModel`][jaxspec.model.instrument.InstrumentModel]: pile-up is
    applied to the effective-area–weighted (ARF) spectrum, *before* the
    redistribution matrix (RMF) and grouping, matching the ordering of the XSPEC
    `pileup` model.

    !!! warning
        [`PileupModel`][jaxspec.model.instrument.PileupModel] requires an explicit and linear evaluation
        grid (pass ``energy_grid=jnp.linspace(low, high, n_bins)`` to the fitter /
        [`BayesianModel`][jaxspec.fit.BayesianModel]).

    Pass it per observation just like a plain
    [`InstrumentModel`][jaxspec.model.instrument.InstrumentModel]:

    ```python
    from jaxspec.fit import MCMCFitter
    from jaxspec.model.instrument import ConstantGain, ConstantShift, PileupModel

    fitter = MCMCFitter(
        spectral_model,
        prior,
        observations,
        energy_grid=np.linspace(0.2, 11.0, 1_000),
        instrument_model={
            "ACIS": PileupModel(
                gain=ConstantGain(),
                shift=ConstantShift(),
                frame_time=3.2,  # CCD frame time / ACIS EXPTIME (s)
                frac_expo=1.0,  # ARF FRACEXPO keyword
            ),
        },
    )
    ```

    `alpha` and `psf_frac` are **fitted** parameters whose
    priors are supplied through the prior dictionary like any other instrument
    parameter, e.g. ``"instrument.alpha"`` and ``"instrument.psf_frac"`` (see
    [`MCMCFitter`][jaxspec.fit.MCMCFitter]). The remaining arguments are **fixed**
    configuration constants.

    !!! note
        Pile-up is **non-linear** in flux: as noted in the
        [XSPEC `pileup` documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPileup.html),
        increasing the source normalisation does not scale the predicted count
        rate linearly, and fluxes should be computed with the pile-up correction
        removed. The extraction region should be large enough to contain
        essentially all of the point-source PSF.

    Parameters:
        gain: Optional energy-independent flux scaling, as for
            [`InstrumentModel`][jaxspec.model.instrument.InstrumentModel] (e.g.
            [`ConstantGain`][jaxspec.model.instrument.ConstantGain]).
        shift: Optional energy shift, as for
            [`InstrumentModel`][jaxspec.model.instrument.InstrumentModel] (e.g.
            [`ConstantShift`][jaxspec.model.instrument.ConstantShift]).
        frame_time: CCD frame (readout) time in seconds (``EXPTIME`` keyword).
        frac_expo: Fractional exposure per frame in ``(0, 1]`` (``FRACEXPO`` keyword).
        g0: Grade correction for single-photon detection.
        npiled: Maximum number of photons piled up in a single frame
        num_regions: Number of regions over which the piled counts are distributed, `1` for a point source.

    Attributes:
        alpha: Grade-morphing parameter — the fraction of piled events that keep a good
            grade, with the good-grade fraction of an order-``p`` pile-up taken
            proportional to ``alpha ** (p - 1)``. Fitted; give it a prior under
            ``"instrument.alpha"``.
        psf_frac: PSF fraction — only this fraction of the extracted counts is treated for
            pile-up. Fitted; give it a prior under ``"instrument.psf_frac"``.
    """

    requires_components = True

    def __init__(
        self,
        gain: GainModel | None = None,
        shift: ShiftModel | None = None,
        frac_expo: float | None = None,
        frame_time: float | None = None,
        num_regions: float | None = 1.0,
        g0: float | None = 1.0,
        npiled: int | None = 5,
    ):
        super().__init__(gain=gain, shift=shift)

        if frac_expo is None or frame_time is None:
            raise ValueError(
                "PileupModel requires both `frac_expo` and `frame_time` keyword arguments "
                "(the 'FRACEXPO' and 'EXPTIME' header values)."
            )

        self.alpha = nnx.Param(jnp.asarray(0.5))
        self.psf_frac = nnx.Param(jnp.asarray(0.95))

        self._constants = {
            "frac_expo": frac_expo,
            "frame_time": frame_time,
            "num_regions": num_regions,
            "g0": g0,
            "npiled": int(np.rint(npiled)),
        }

    def fold(
        self,
        spectrum: ArrayLike,
        cache: dict,
        eval_energies: ArrayLike | None = None,
    ):
        if eval_energies is None:
            raise ValueError("Eval energies cannot be None : an energy grid must be provided")

        spectrum = self._redistribute_to_native(spectrum, cache, eval_energies)

        num_regions = self._constants["num_regions"]
        fracexpo = self._constants["frac_expo"]
        frame_time = self._constants["frame_time"]
        g0 = self._constants["g0"]
        npiled = self._constants["npiled"]

        in_energies = cache["in_energies"]
        # Align FFT indices when the uniform energy grid starts above zero.
        bin_width = in_energies[0, 1] - in_energies[0, 0]
        ioff = -jnp.array(in_energies[0, 0] // bin_width, jnp.int_)
        n_orig = in_energies.shape[1]
        shift_idx = jnp.arange(n_orig) + ioff

        def pileup_fold(s):
            arf_s = s * cache["area"]

            psf_frac = self.psf_frac / num_regions / fracexpo
            arf_s_tmp = arf_s * psf_frac
            integ_arf_s = jnp.sum(arf_s_tmp)

            results = arf_s * psf_frac  # term p=1 of the sum

            exp_factor = jnp.exp(-frame_time * integ_arf_s / g0)
            exp_factor = exp_factor * num_regions * fracexpo

            # Normalize to avoid overflow and perform FFT convolutions
            arf_s_tmp = arf_s_tmp / integ_arf_s
            integ_arf_s_n = integ_arf_s  # for renormalization after
            arf_s_fft = jnp.fft.rfft(arf_s_tmp)
            factor = 1

            # Compute FFT with offset
            tmpar = arf_s_tmp[shift_idx]
            arf_s_fft_2 = jnp.fft.rfft(tmpar)

            # Calculate higher order terms
            for i in range(2, npiled + 1):
                integ_arf_s_n = integ_arf_s_n * integ_arf_s  # renormalization factor

                # Convolution via FFT
                conv = jnp.fft.irfft(arf_s_fft * arf_s_fft_2 ** (i - 1), n=n_orig)
                conv = jnp.clip(conv, min=0)

                # Apply grade migration factor
                factor = factor * self.alpha * frame_time / i
                results = results + factor * integ_arf_s_n * conv

            # Apply final corrections
            results = results * exp_factor

            # Handle non-piled fraction
            remaining_frac = 1.0 - self.psf_frac
            results = results + arf_s * jnp.clip(remaining_frac, min=0)

            results = self.apply_gain(results, in_energies)

            return jnp.clip(
                cache["grouping"]
                @ (cache.get("areascal", 1.0) * (cache["redistribution"] @ results))
                * cache["exposure"],
                min=1e-10,
            )

        return jax.tree.map(pileup_fold, spectrum)
