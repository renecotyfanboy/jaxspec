"""Helpers for posterior-predictive plots and component overlays.

Fit objects are accepted through function arguments to avoid an import cycle between the
analysis and fitting packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import astropy.units as u
import jax
import matplotlib.pyplot as plt
import numpy as np

from astropy.units import Unit

from ._plot import (
    BACKGROUND_COLOR,
    BACKGROUND_DATA_COLOR,
    COLOR_CYCLE,
    SPECTRUM_COLOR,
    SPECTRUM_DATA_COLOR,
    _compute_bin_ids,
    _compute_effective_area,
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
    _rebin_xbins,
    rebin_counts,
)
from ._posterior_params import leaf_inputs

if TYPE_CHECKING:
    from .results import FitResult


_Y_UNITS_FOR_TYPE: dict[str, Any] = {
    "counts": u.ct,
    "countrate": u.ct / u.s,
    "photon_flux": u.ct / u.cm**2 / u.s,
}

_XLABEL_FOR_PHYSICAL_TYPE: dict[str, str] = {
    "length": "Wavelength",
    "energy": "Energy",
    "frequency": "Frequency",
}


def _validate_x_unit(x_unit: Unit) -> str:
    """Return the axis label for ``x_unit``, raising if its physical type is unsupported.

    Plot axes support length, energy, and frequency. The physical type is checked
    explicitly because spectral equivalencies also convert unsupported types such as
    wavenumber. Validation occurs before creating the figure.
    """
    physical_type = x_unit.physical_type
    label = next(
        (lbl for pt, lbl in _XLABEL_FOR_PHYSICAL_TYPE.items() if physical_type == pt), None
    )
    if label is None:
        raise ValueError(
            f"Unsupported physical type {str(physical_type)!r} for x_unit {x_unit}. "
            f"It must be homogeneous to a length, an energy or a frequency."
        )
    return label


_RESIDUAL_YLIM = 3.5
_RESIDUAL_SIGMA_GUIDE = 3
_YLIM_PAD_LOW, _YLIM_PAD_HIGH = 0.8, 1.2

_SCALE_TO_AXES: dict[str, tuple[str, str]] = {
    "linear": ("linear", "linear"),
    "semilogx": ("log", "linear"),
    "semilogy": ("linear", "log"),
    "loglog": ("log", "log"),
}


def folded_branches(result: FitResult, obs_id):
    """Per-branch posterior-predictive counts for ``obs_id``.

    Returns ``{branch_name: (n_chains, n_draws, n_bins)}`` count arrays
    suitable for component overlays (each branch is one
    ``additive * multiplicative*…`` path in the spectral model). Branches are
    evaluated through :meth:`ForwardModel.evaluate`, including the same response
    folding and per-observation gain or shift used during inference.
    """
    fm = result.bayesian_fitter.forward_model
    inputs = leaf_inputs(fm, result.input_parameters)
    if not inputs:
        raise ValueError(
            "Per-component PPC overlay is unavailable for callable priors "
            "(no static parameter set to enumerate)."
        )

    @jax.jit
    @jax.vmap
    @jax.vmap
    def evaluate_one(inp):
        return fm.evaluate(inp, split_branches=True, with_background=False)[obs_id]["source"]

    folded = evaluate_one(inputs)
    return jax.tree.map(lambda flux: np.random.poisson(np.asarray(flux)), folded)


def plot_ppc(
    result: FitResult,
    *,
    n_sigmas: int = 1,
    x_unit: str | u.Unit = "keV",
    y_type: Literal["counts", "countrate", "photon_flux", "photon_flux_density"] = (
        "photon_flux_density"
    ),
    plot_background: bool = True,
    plot_components: bool = False,
    scale: Literal["linear", "semilogx", "semilogy", "loglog"] = "loglog",
    alpha_envelope: tuple[float, float] = (0.15, 0.25),
    style: str | Any = "default",
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
    x_lims: tuple[float, float] | None = None,
    rescale_background: bool = False,
    min_counts: int | None = None,
    grouping: int | None = None,
) -> list[plt.Figure]:
    """Body of :meth:`~jaxspec.analysis.results.FitResult.plot_ppc` — see there for docs."""
    if min_counts is not None and grouping is not None:
        raise ValueError("min_counts and grouping are mutually exclusive")

    x_unit = u.Unit(x_unit)
    _validate_x_unit(x_unit)
    y_units = _resolve_y_units(y_type, x_unit)
    figure_list = []

    with plt.style.context(style):
        for obs_id, obsconf in result.obsconfs.items():
            fig, ax = plt.subplots(2, 1, figsize=figsize, sharex="col", height_ratios=[0.7, 0.3])

            count = az.extract(
                result.inference_data,
                var_names=f"observed.{obs_id}",
                group="posterior_predictive",
            ).values.T
            xbins, exposure, integrated_arf = _compute_effective_area(obsconf, x_unit)
            observed_counts = obsconf.folded_counts.data
            bin_ids = _compute_bin_ids(observed_counts, min_counts, grouping)

            count, observed_counts, xbins, integrated_arf = _apply_binning(
                bin_ids, count, observed_counts, xbins, integrated_arf
            )

            denominator = _compute_denominator(y_type, exposure, integrated_arf, xbins)
            y_samples = (count * u.ct / denominator).to(y_units)
            y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                observed_counts, denominator, y_units
            )

            model_plot = _plot_binned_samples_with_error(
                ax[0],
                xbins.value,
                y_samples.value,
                color=SPECTRUM_COLOR,
                n_sigmas=n_sigmas,
                alpha_envelope=alpha_envelope,
            )
            true_data_plot = _plot_poisson_data_with_error(
                ax[0],
                xbins.value,
                y_observed.value,
                y_observed_low.value,
                y_observed_high.value,
                color=SPECTRUM_DATA_COLOR,
                alpha=0.7,
            )

            legend_plots = [(true_data_plot,), *model_plot]
            legend_labels = ["Observed", "Model"]

            residual_samples = (observed_counts - count) / np.diff(
                np.percentile(count, [16, 84], axis=0), axis=0
            )
            _plot_binned_samples_with_error(
                ax[1],
                xbins.value,
                residual_samples,
                color=SPECTRUM_COLOR,
                n_sigmas=n_sigmas,
                alpha_envelope=alpha_envelope,
            )

            if plot_components:
                extra_plots, extra_labels = _plot_components_overlay(
                    result,
                    ax[0],
                    obs_id,
                    denominator,
                    y_units,
                    bin_ids,
                    xbins,
                    n_sigmas,
                    alpha_envelope,
                )
                legend_plots += extra_plots
                legend_labels += extra_labels

            if (
                result.bayesian_fitter.forward_model.background.get(obs_id) is not None
                and plot_background
            ):
                extra_plots, extra_labels = _plot_background_overlay(
                    result,
                    ax[0],
                    obsconf,
                    obs_id,
                    denominator,
                    y_units,
                    bin_ids,
                    xbins,
                    rescale_background,
                    n_sigmas,
                    alpha_envelope,
                )
                legend_plots += extra_plots
                legend_labels += extra_labels

            _style_axes(
                ax,
                x_unit,
                scale,
                x_lims,
                y_units,
                xbins,
                np.nanmin(y_observed),
                np.nanmax(y_observed),
                legend_plots,
                legend_labels,
            )

            fig.align_ylabels()
            plt.subplots_adjust(hspace=0.0)
            fig.suptitle(f"Posterior predictive - {obs_id}" if title is None else title)
            fig.tight_layout()
            figure_list.append(fig)

    plt.tight_layout()
    plt.show()

    return figure_list


def _plot_components_overlay(
    result,
    ax,
    obs_id,
    denominator,
    y_units,
    bin_ids,
    xbins,
    n_sigmas,
    alpha_envelope,
) -> tuple[list, list]:
    """Overlay per-component posterior bands; return legend entries to append."""
    extra_plots: list = []
    extra_labels: list = []
    for (component_name, comp_count), color in zip(
        folded_branches(result, obs_id).items(), COLOR_CYCLE
    ):
        comp_flat = comp_count.reshape((comp_count.shape[0] * comp_count.shape[1], -1))
        if bin_ids is not None:
            comp_flat = rebin_counts(comp_flat, bin_ids)
        y_samples = (comp_flat * u.ct / denominator).to(y_units)
        component_plot = _plot_binned_samples_with_error(
            ax,
            xbins.value,
            y_samples.value,
            color=color,
            linestyle="dashdot",
            n_sigmas=n_sigmas,
            alpha_envelope=alpha_envelope,
        )
        extra_plots += component_plot
        extra_labels.append(component_name.split("*")[-1])
    return extra_plots, extra_labels


def _plot_background_overlay(
    result,
    ax,
    obsconf,
    obs_id,
    denominator,
    y_units,
    bin_ids,
    xbins,
    rescale_background,
    n_sigmas,
    alpha_envelope,
) -> tuple[list, list]:
    """Overlay the background model/data; return legend entries to append."""
    bkg_count = az.extract(
        result.inference_data,
        var_names=f"observed_background.{obs_id}",
        group="posterior_predictive",
    ).values.T
    bkg_observed = obsconf.folded_background.data

    if bin_ids is not None:
        bkg_count = rebin_counts(bkg_count, bin_ids)
        bkg_observed = rebin_counts(bkg_observed, bin_ids)
        rescale_background_factor = (
            rebin_counts(obsconf.folded_backratio.data, bin_ids) / np.bincount(bin_ids)
            if rescale_background
            else 1.0
        )
    else:
        rescale_background_factor = obsconf.folded_backratio.data if rescale_background else 1.0

    y_samples_bkg = (bkg_count * u.ct / denominator).to(y_units)
    y_observed_bkg, y_observed_bkg_low, y_observed_bkg_high = _error_bars_for_observed_data(
        bkg_observed, denominator, y_units
    )
    model_bkg_plot = _plot_binned_samples_with_error(
        ax,
        xbins.value,
        y_samples_bkg.value * rescale_background_factor,
        color=BACKGROUND_COLOR,
        alpha_envelope=alpha_envelope,
        n_sigmas=n_sigmas,
    )
    true_bkg_plot = _plot_poisson_data_with_error(
        ax,
        xbins.value,
        y_observed_bkg.value * rescale_background_factor,
        y_observed_bkg_low.value * rescale_background_factor,
        y_observed_bkg_high.value * rescale_background_factor,
        color=BACKGROUND_DATA_COLOR,
        alpha=0.7,
    )
    return [(true_bkg_plot,), *model_bkg_plot], ["Observed (bkg)", "Model (bkg)"]


def _resolve_y_units(y_type, x_unit):
    if y_type == "photon_flux_density":
        return u.ct / u.cm**2 / u.s / x_unit
    units = _Y_UNITS_FOR_TYPE.get(y_type)
    if units is None:
        raise ValueError(
            f"Unknown y_type: {y_type}. Must be 'counts', 'countrate', 'photon_flux' "
            f"or 'photon_flux_density'"
        )
    return units


def _apply_binning(bin_ids, count, observed_counts, xbins, integrated_arf):
    if bin_ids is None:
        return count, observed_counts, xbins, integrated_arf
    count = rebin_counts(count, bin_ids)
    observed_counts = rebin_counts(observed_counts, bin_ids)
    xbins = _rebin_xbins(xbins, bin_ids)
    integrated_arf = rebin_counts(integrated_arf.value, bin_ids) * integrated_arf.unit
    return count, observed_counts, xbins, integrated_arf


def _compute_denominator(y_type, exposure, integrated_arf, xbins):
    if y_type == "counts":
        return 1
    if y_type == "countrate":
        return exposure
    if y_type == "photon_flux":
        return integrated_arf * exposure
    if y_type == "photon_flux_density":
        return (xbins[1] - xbins[0]) * integrated_arf * exposure
    raise ValueError(f"Unknown y_type: {y_type}")


def _style_axes(
    ax,
    x_unit,
    scale,
    x_lims,
    y_units,
    xbins,
    lowest_y,
    highest_y,
    legend_plots,
    legend_labels,
):
    ax[1].set_ylim(-_RESIDUAL_YLIM, _RESIDUAL_YLIM)
    ax[0].set_ylabel(f"Folded spectrum\n [{y_units:latex_inline}]")
    ax[1].set_ylabel("Residuals \n" + r"[$\sigma$]")

    ax[1].set_xlabel(f"{_validate_x_unit(x_unit)} \n[{x_unit:latex_inline}]")

    ax[1].axhline(0, color=SPECTRUM_DATA_COLOR, ls="--")
    for guide in (-_RESIDUAL_SIGMA_GUIDE, _RESIDUAL_SIGMA_GUIDE):
        ax[1].axhline(guide, color=SPECTRUM_DATA_COLOR, ls=":")
    ticks = [-_RESIDUAL_SIGMA_GUIDE, 0, _RESIDUAL_SIGMA_GUIDE]
    ax[1].set_yticks(ticks, labels=ticks)
    ax[1].set_yticks(range(-_RESIDUAL_SIGMA_GUIDE, _RESIDUAL_SIGMA_GUIDE + 1), minor=True)

    # Set scales first so logarithmic axes can clip non-positive limits.
    xscale, yscale = _SCALE_TO_AXES[scale]
    ax[0].set_xscale(xscale)
    ax[0].set_yscale(yscale)

    ax[0].set_xlim(xbins.value.min(), xbins.value.max())
    ax[0].set_ylim(lowest_y.value * _YLIM_PAD_LOW, highest_y.value * _YLIM_PAD_HIGH)
    ax[0].legend(legend_plots, legend_labels)

    if x_lims is not None:
        ax[0].set_xlim(*x_lims)
