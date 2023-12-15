import numpy as np
from .observation import Observation


def constant_rebin(obs: Observation, downfactor: int):
    """
    Create a grouping matrix for constant rebinning of the data.

    Parameters:
        obs: Observation object.
        downfactor: Downfactor for the rebinning.

    Returns:
        A grouping matrix.
    """
    n_channels = len(obs.pha.channel)
    n_bins = n_channels // downfactor + 1 * (n_channels % downfactor != 0)
    grouping = np.zeros((n_bins, n_channels), dtype=bool)

    for i in range(n_bins):
        grouping[i, i * downfactor : (i + 1) * downfactor] = True

    return grouping
