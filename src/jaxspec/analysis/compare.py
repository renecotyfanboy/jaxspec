from chainconsumer import ChainConsumer

from .results import FitResult


def plot_corner_comparison(obs_dict: dict[str, FitResult], **kwargs):
    """Compare posterior correlations from several fitted observations.

    Parameters:
        obs_dict: Fit results keyed by the labels used in the comparison plot.
        **kwargs: Additional arguments passed to ``ChainConsumer.plotter.plot``.
    """

    c = ChainConsumer()

    for name, obs in obs_dict.items():
        c.add_chain(obs.to_chain(name))

    return c.plotter.plot(**kwargs)
