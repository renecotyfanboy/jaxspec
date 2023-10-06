import arviz as az
from numpyro.infer import MCMC


class ResultContainer:

    pass


class ChainResult(ResultContainer):

    def __init__(self, mcmc: MCMC):

        self.mcmc = mcmc
        self.samples = mcmc.get_samples()

    @property
    def inference_data(self):

        return az.from_numpyro(self.mcmc)
