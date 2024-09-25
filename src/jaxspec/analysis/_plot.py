import matplotlib.pyplot as plt
import numpy as np

from jax.typing import ArrayLike
from scipy.stats import nbinom


def _plot_poisson_data_with_error(
    ax: plt.Axes,
    x_bins: ArrayLike,
    y: ArrayLike,
    percentiles: tuple = (16, 84),
):
    """
    Plot Poisson data with error bars. We extrapolate the intrinsic error of the observation assuming a prior rate
    distributed according to a Gamma RV.
    """
    y_low = nbinom.ppf(percentiles[0] / 100, y, 0.5)
    y_high = nbinom.ppf(percentiles[1] / 100, y, 0.5)

    ax_to_plot = ax.errorbar(
        np.sqrt(x_bins[0] * x_bins[1]),
        y,
        xerr=np.abs(x_bins - np.sqrt(x_bins[0] * x_bins[1])),
        yerr=[
            y - y_low,
            y_high - y,
        ],
        color="black",
        linestyle="none",
        alpha=0.3,
        capsize=2,
    )

    return ax_to_plot
