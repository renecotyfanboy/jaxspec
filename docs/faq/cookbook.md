# Cookbook : how do I ...

## Evaluate the true model

You should look at [`SpectralModel.photon_flux`][jaxspec.model.abc.SpectralModel.photon_flux] and
[`SpectralModel.energy_flux`][jaxspec.model.abc.SpectralModel.energy_flux] methods.

``` python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxspec.model.additive import Blackbodyrad
from jaxspec.model.multiplicative import Tbabs

model = Tbabs() * Blackbodyrad()
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

photon_flux = model.photon_flux(model.params, energies[:-1], energies[1:], n_points=30)
energy_flux = model.energy_flux(model.params, energies[:-1], energies[1:], n_points=30)
```

## Fit a single observation with MCMC

## Fit multiple observations with MCMC

## Compute model photon flux, energy flux and luminosity