import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from ..data.observation import Observation
from ..model.abc import SpectralModel
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar, Tuple
from numpyro.infer import MCMC
from haiku.data_structures import traverse

K = TypeVar("K")
V = TypeVar("V")


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
    #TODO : Add docstring
    #TODO : Add type hints
    #TODO : Add proper separation between params and samples, cf from haiku and numpyro
    def __init__(self,
                 model: SpectralModel,
                 observations: list[Observation],
                 mcmc: MCMC,
                 structure: Mapping[K, V]):

        super().__init__(model, structure)

        self.inference_data = az.from_numpyro(mcmc)
        self.observations = observations
        self.samples = mcmc.get_samples()


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

    def plot_ppc(self, percentile: Tuple[int, int] = (14, 86)):

        from ..data.util import fakeit_for_multiple_parameters

        counts = fakeit_for_multiple_parameters(self.observations, self.model, self.params)
        n_graphs = len(self.observations)
        n_cols = int(np.ceil(np.sqrt(n_graphs)))
        n_rows = int(np.ceil(n_graphs / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        for ax in axs.flatten():
            ax.set_axis_off()

        for count, observation, ax in zip(counts, self.observations, axs.flatten()):
            ax.set_axis_on()
            ax.step(observation.out_energies[0], observation.observed_counts, where="pre", label="data")
            ax.fill_between(observation.out_energies[0], *np.percentile(count, percentile, axis=0), alpha=0.3, step='pre',
                            label="posterior predictive")
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Counts')
            ax.loglog()

    @property
    def table(self):

        for module, parameter, value in traverse(self.params):
            print(f'{module}_{parameter} : {value.mean()} +/- {value.std()}')
