import chex
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax.config import config
from unittest import TestCase
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs
from jaxspec.data.util import example_observations as obs_list
from jaxspec.fit import BayesianModel


chex.set_n_cpu_devices(n=4)

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


class TestResults(TestCase):
    model = Tbabs() * Powerlaw()
    forward = BayesianModel(model, list(obs_list.values()))
    prior = {"powerlaw_1": {"alpha": dist.Uniform(0, 10), "norm": dist.Exponential(1e4)}, "tbabs_1": {"N_H": dist.Uniform(0, 1)}}

    result = forward.fit(prior, num_samples=1000)

    def test_plot_ppc(self):
        """
        Test reading an AMF file using ref files
        """

        self.result.plot_ppc(0, percentile=(5, 95))
        plt.show()
