from typing import Dict
from .results import FitResult
from chainconsumer import ChainConsumer


def plot_corner_comparison(obs_dict: Dict[str, FitResult], **kwargs):
    """
    Plot the correlation plot of parameters from different fitted observations. Observations are passed in as a
    dictionary. Each observation is named according to its key. It shall be used to compare the same model independently
    fitted to different observations.

    Parameters:
        obs_dict: a dictionary containing the observations to plot. The keys are the names of the observations.
    """

    c = ChainConsumer()

    for name, obs in obs_dict.items():
        c.add_chain(obs.to_chain(name))

    return c.plotter.plot(**kwargs)
