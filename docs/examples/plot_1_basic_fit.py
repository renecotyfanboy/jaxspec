"""
# JAXspec fitting speedrun

In this example, the basic spectral fitting workflow is illustrated on a XMM-Newton observation of the
pulsating ULX candidate from Quintin+2020.

"""
import numpyro
from jax.config import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


# %% New cell
# The first step consists in building your model using the various components available in JAXspec.
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model = Tbabs()*Powerlaw()

# %% New cell
# The second step consists in defining the data to be fitted.
from jaxspec.data.util import example_observations as obs_list
from jaxspec.fit import BayesianModel

forward = BayesianModel(model, list(obs_list.values()))

# %% New cell
# The third step consists in defining the priors for the model parameters.
import numpyro.distributions as dist

prior = {
    'powerlaw_1': {
        'alpha': dist.Uniform(0, 10),
        'norm': dist.Exponential(1e4)
    },
    'tbabs_1': {
        'N_H': dist.Uniform(0, 1)
    }
}

# %% New cell
# The fourth step consists in defining the likelihood for the model parameters.
result = forward.fit(prior,
                     num_chains=4,
                     num_samples=1000,
                     mcmc_kwargs={'progress_bar': False})

# %% New cell
# The fourth step consists in defining the likelihood for the model parameters.
print(result.table())

# %% New cell
# And that's it !