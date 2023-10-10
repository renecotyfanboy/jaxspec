import arviz as az
from collections.abc import Mapping
from typing import TypeVar
from numpyro.infer import MCMC
from haiku.data_structures import traverse

K = TypeVar("K")
V = TypeVar("V")

class ResultContainer:

    pass


class ChainResult(ResultContainer):
    #TODO : Add docstring
    #TODO : Add type hints
    #TODO : Add proper separation between params and samples, cf from haiku and numpyro
    def __init__(self,
                 mcmc: MCMC,
                 structure: Mapping[K, V]):

        self.mcmc = mcmc
        self.samples = mcmc.get_samples()
        self.params = {}

        for module, parameter, value in traverse(structure):
            if self.params.get(module, None) is None:
                self.params[module] = {}
            self.params[module][parameter] = self.samples[f'{module}_{parameter}']

    @property
    def inference_data(self):

        return az.from_numpyro(self.mcmc)
