# Cookbook : how do I ...

## Fit observations with MCMC

This is the example we use in the `jaxspec` paper.

```python
import numpyro

numpyro.enable_x64()
numpyro.set_host_device_count(4)
numpyro.set_platform("cpu")

import numpyro.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxspec.data.util import load_example_obsconf
from jaxspec.fit import MCMCFitter
from jaxspec.model.additive import Powerlaw, Blackbodyrad
from jaxspec.model.multiplicative import Tbabs

spectral_model = Tbabs() * (Powerlaw() + Blackbodyrad())

prior = {
    'powerlaw_1': {
        'alpha': dist.Uniform(0. * jnp.ones((3,)), 5 * jnp.ones((3,))),
        'norm': dist.LogUniform(1e-6, 1e-3)
    },
    'blackbodyrad_1': {
        'kT': dist.Uniform(0.3, 3),
        'norm': dist.LogUniform(1e-2, 1e3)
    },
    'tbabs_1': {
        'N_H': 0.2
    }
}

ulx_observations = load_example_obsconf()
fitter = MCMCFitter(spectral_model, prior, ulx_observations)
result = fitter.fit(num_samples=1_000)
```

## Evaluate the true model

You should look at [`SpectralModel.photon_flux`][jaxspec.model.abc.SpectralModel.photon_flux] and
[`SpectralModel.energy_flux`][jaxspec.model.abc.SpectralModel.energy_flux] methods.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxspec.model.additive import Blackbodyrad
from jaxspec.model.multiplicative import Tbabs

spectral_model = Tbabs() * Blackbodyrad()

energies = jnp.geomspace(1, 50, 100)

params = {
    'blackbodyrad_1': {
        'kT': 1.,
        'norm': 1.
    },
    'tbabs': {
        'nH': 1.
    }
}

photon_flux = spectral_model.photon_flux(params, energies[:-1], energies[1:], n_points=30)
energy_flux = spectral_model.energy_flux(params, energies[:-1], energies[1:], n_points=30)
```

## Compute model photon flux, energy flux and luminosity

You should look at [`FitResult.photon_flux`][jaxspec.analysis.results.FitResult.photon_flux],
[`FitResult.energy_flux`][jaxspec.analysis.results.FitResult.energy_flux], and
[`FitResult.luminosity`][jaxspec.analysis.results.FitResult.luminosity]

## Save and load inference results

You can use the `dill` package to serialise and un-serialise such objects. First you should install it using `pip`

```
pip install dill
```

Then use the following lines to save and load the files:

```python
import dill

# Save the results
with open(r"result.pickle", "wb") as output_file:
    dill.dump(result, output_file)

# Load the results
with open(r"result.pickle", "rb") as input_file:
    result_pickled = dill.load(input_file)
```