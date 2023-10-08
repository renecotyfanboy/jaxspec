import arviz as az
from numpyro.infer import MCMC
from chainconsumer import ChainConsumer


class ResultContainer:

    pass


class ChainResult(ResultContainer):

    def __init__(self, mcmc: MCMC):

        self.mcmc = mcmc
        self.samples = mcmc.get_samples()

    @property
    def inference_data(self):

        return az.from_numpyro(self.mcmc)

    def plot_corner(self):

        consumer = ChainConsumer()
