import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..data.observation import Observation
from ..model.abc import SpectralModel
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar, Tuple
from numpyro.infer import MCMC
from haiku.data_structures import traverse
from chainconsumer import ChainConsumer, Chain, PlotConfig

K = TypeVar("K")
V = TypeVar("V")


def format_parameters(parameter_name):
    if parameter_name == 'weight':
        # ChainConsumer add a weight column to the samples
        return parameter_name

    # Find second occurrence of the character '_'
    first_occurrence = parameter_name.find('_')
    second_occurrence = parameter_name.find('_', first_occurrence + 1)
    module = parameter_name[:second_occurrence]
    parameter = parameter_name[second_occurrence + 1:]

    name, number = module.split('_')
    module = rf'[{name.capitalize()} ({number})]'

    if parameter == "norm":
        return r'\text{Norm' + ' ' + module + r'}'

    else:

        return rf'{parameter}' + ' ' + r'\text{' + module + r'}'


class ResultContainer(ABC):
    """
    This class is a container for the results of a fit.

    TODO : Add flux, luminosity, etc.
    """

    model: SpectralModel

    def __init__(self, model: SpectralModel, structure: Mapping[K, V]):
        self.model = model
        self._structure = structure

    @abstractmethod
    def plot_ppc(self):
        ...

    @property
    @abstractmethod
    def table(self):
        ...


class ChainResult(ResultContainer):
    # TODO : Add docstring
    # TODO : Add type hints
    # TODO : Add proper separation between params and samples, cf from haiku and numpyro
    def __init__(self,
                 model: SpectralModel,
                 observations: list[Observation],
                 mcmc: MCMC,
                 structure: Mapping[K, V]):

        super().__init__(model, structure)

        self.inference_data = az.from_numpyro(mcmc)
        self.observations = observations
        self.samples = mcmc.get_samples()
        self.consumer = ChainConsumer()

        self.chain = Chain.from_numpyro(mcmc, "Model", kde=1)
        self.chain.samples.columns = [format_parameters(parameter) for parameter in self.chain.samples.columns]
        self._structure = structure

    @property
    def params(self):
        """
        Haiku-like structure for the parameters
        """

        params = {}

        for module, parameter, value in traverse(self._structure):
            if params.get(module, None) is None:
                params[module] = {}
            params[module][parameter] = self.samples[f'{module}_{parameter}']

        return params

    def plot_ppc(self, index: int, percentile: Tuple[int, int] = (14, 86)):

        from ..data.util import fakeit_for_multiple_parameters

        count = fakeit_for_multiple_parameters(self.observations[0], self.model, self.params)

        with plt.style.context('default'):
            fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, height_ratios=[0.8, 0.2])

            observation = self.observations[index]

            axs[0].step(observation.out_energies[0],
                        observation.observed_counts,
                        where="post",
                        label="data")

            axs[0].fill_between(observation.out_energies[0],
                                *np.percentile(count, percentile, axis=0),
                                alpha=0.3,
                                step='post',
                                label="posterior predictive")

            axs[0].set_ylabel('Counts')
            axs[0].loglog()

            residuals = np.percentile((observation.observed_counts - count)
                                      / np.diff(np.percentile(count, percentile, axis=0), axis=0), percentile, axis=0)

            max_residuals = np.max(np.abs(residuals))

            axs[1].fill_between(observation.out_energies[0],
                                *residuals,
                                alpha=0.3,
                                step='post',
                                label="posterior predictive")

            axs[1].set_ylim(-max_residuals, +max_residuals)
            axs[1].set_ylabel('Residuals')
            axs[1].set_xlabel('Energy [keV]')
            axs[1].axhline(0, color='k', ls='--')

            axs[0].set_xlim(observation.low_energy, observation.high_energy)
            plt.subplots_adjust(hspace=0.0)

    def table(self):

        return self.consumer.analysis.get_latex_table(caption="Results of the fit", label="tab:results")

    def plot_corner(self, **kwargs):

        consumer = ChainConsumer()
        consumer.set_plot_config(PlotConfig(usetex=False))
        consumer.add_chain(self.chain)

        # Context for default mpl style
        with plt.style.context('default'):
            consumer.plotter.plot(**kwargs)
