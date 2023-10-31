"""
Results for other notebooks
"""

import numpyro
import numpyro.distributions as dist
from jaxspec.data.util import example_observations as obs_list
from jaxspec.fit import BayesianModel
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs
from jax.config import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

model = Tbabs() * Powerlaw()
forward = BayesianModel(model, list(obs_list.values()))

prior = {"powerlaw_1": {"alpha": dist.Uniform(0, 10), "norm": dist.Exponential(1e4)}, "tbabs_1": {"N_H": dist.Uniform(0, 1)}}

result = forward.fit(prior, num_chains=4, num_samples=1000, mcmc_kwargs={"progress_bar": False})
