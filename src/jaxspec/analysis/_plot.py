from __future__ import annotations

import catppuccin
import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from catppuccin.extras.matplotlib import load_color
from cycler import cycler
from jax.typing import ArrayLike
from scipy.integrate import trapezoid
from scipy.stats import nbinom, norm

from jaxspec.data import ObsConfiguration

PALETTE = catppuccin.PALETTE.latte

COLOR_CYCLE = [
    load_color(PALETTE.identifier, color)
    for color in ["sky", "teal", "green", "yellow", "peach", "maroon", "red", "pink", "mauve"][::-1]
]

LINESTYLE_CYCLE = ["dashed", "dotted", "dashdot", "solid"]

SPECS_CYCLE = cycler(linestyle=LINESTYLE_CYCLE) * cycler(color=COLOR_CYCLE)

SPECTRUM_COLOR = load_color(PALETTE.identifier, "blue")
SPECTRUM_DATA_COLOR = load_color(PALETTE.identifier, "overlay2")
BACKGROUND_COLOR = load_color(PALETTE.identifier, "sapphire")
BACKGROUND_DATA_COLOR = load_color(PALETTE.identifier, "overlay0")


def sigma_to_percentile_intervals(sigmas):
    intervals = []
    for sigma in sigmas:
        lower_bound = 100 * norm.cdf(-sigma)
        upper_bound = 100 * norm.cdf(sigma)
        intervals.append((lower_bound, upper_bound))
    return intervals


def _plot_poisson_data_with_error(
    ax: plt.Axes,
    x_bins: ArrayLike,
    y: ArrayLike,
    y_low: ArrayLike,
    y_high: ArrayLike,
    color=SPECTRUM_DATA_COLOR,
    linestyle="none",
    alpha=0.3,
):
    """
    Plot Poisson data with error bars. We extrapolate the intrinsic error of the observation assuming a prior rate
    distributed according to a Gamma RV.
    """

    ax_to_plot = ax.errorbar(
        np.sqrt(x_bins[0] * x_bins[1]),
        y,
        xerr=np.abs(x_bins - np.sqrt(x_bins[0] * x_bins[1])),
        yerr=[
            y - y_low,
            y_high - y,
        ],
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        capsize=2,
    )

    return ax_to_plot


def _plot_binned_samples_with_error(
    ax: plt.Axes,
    x_bins: ArrayLike,
    y_samples: ArrayLike,
    color=SPECTRUM_COLOR,
    alpha_median: float = 0.7,
    alpha_envelope: (float, float) = (0.15, 0.25),
    linestyle="solid",
    n_sigmas=3,
):
    """
    Helper function to plot the posterior predictive distribution of the model. The function
    computes the percentiles of the posterior predictive distribution and plot them as a shaded
    area. If the observed data is provided, it is also plotted as a step function.

    Parameters:
        x_bins: The bin edges of the data (2 x N).
        y_samples: The samples of the posterior predictive distribution (Samples X N).
        ax: The matplotlib axes object.
        color: The color of the posterior predictive distribution.
    """

    median = ax.stairs(
        list(np.median(y_samples, axis=0)),
        edges=[*list(x_bins[0]), x_bins[1][-1]],
        color=color,
        alpha=alpha_median,
        linestyle=linestyle,
    )

    # The legend cannot handle fill_between, so we pass a fill to get a fancy icon
    (envelope,) = ax.fill(np.nan, np.nan, alpha=alpha_envelope[-1], facecolor=color)

    if n_sigmas == 1:
        alpha_envelope = (alpha_envelope[1], alpha_envelope[0])

    for percentile, alpha in zip(
        sigma_to_percentile_intervals(list(range(n_sigmas, 0, -1))),
        np.linspace(*alpha_envelope, n_sigmas),
    ):
        percentiles = np.percentile(y_samples, percentile, axis=0)
        ax.stairs(
            percentiles[1],
            edges=[*list(x_bins[0]), x_bins[1][-1]],
            baseline=percentiles[0],
            alpha=alpha,
            fill=True,
            color=color,
        )

    return [(median, envelope)]


def _compute_effective_area(
    obsconf: ObsConfiguration,
    x_unit: str | u.Unit = "keV",
):
    """
    Helper function to compute the bins and effective area of an observational configuration

    Parameters:
        obsconf: The observational configuration.
        x_unit: The unit of the x-axis. It can be either a string (parsable by astropy.units) or an astropy unit. It must be homogeneous to either a length, a frequency or an energy.
    """

    # Note to Simon : do not change xbins[1] - xbins[0] to
    # np.diff, you already did this twice and forgot that it does not work since diff keeps the dimensions
    # and enable weird broadcasting that makes the plot fail

    xbins = obsconf.out_energies * u.keV
    xbins = xbins.to(x_unit, u.spectral())

    # This computes the total effective area within all bins
    # This is a bit weird since the following computation is equivalent to ignoring the RMF
    exposure = obsconf.exposure.data * u.s
    mid_bins_arf = obsconf.in_energies.mean(axis=0) * u.keV
    mid_bins_arf = mid_bins_arf.to(x_unit, u.spectral())
    e_grid = np.linspace(*xbins, 10)
    interpolated_arf = np.interp(e_grid, mid_bins_arf, obsconf.area)
    integrated_arf = (
        trapezoid(interpolated_arf, x=e_grid, axis=0)
        / (
            np.abs(
                xbins[1] - xbins[0]
            )  # Must fold in abs because some units reverse the ordering of the bins
        )
        * u.cm**2
    )

    return xbins, exposure, integrated_arf


def _error_bars_for_observed_data(observed_counts, denominator, units, sigma=1):
    r"""
    Compute the error bars for the observed data assuming a prior Gamma distribution

    Parameters:
        observed_counts: array of integer counts
        denominator: normalization factor (e.g. effective area)
        units: unit to convert to
        sigma: dispersion to use for quantiles computation

    Returns:
        y_observed: observed counts in the desired units
        y_observed_low: lower bound of the error bars
        y_observed_high: upper bound of the error bars
    """

    percentile = sigma_to_percentile_intervals([sigma])[0]

    y_observed = (observed_counts * u.ct / denominator).to(units)

    y_observed_low = (
        nbinom.ppf(percentile[0] / 100, observed_counts, 0.5) * u.ct / denominator
    ).to(units)

    y_observed_high = (
        nbinom.ppf(percentile[1] / 100, observed_counts, 0.5) * u.ct / denominator
    ).to(units)

    return y_observed, y_observed_low, y_observed_high
