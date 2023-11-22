# Generate mock data

This tutorial illustrates how to make generate mock observed spectra using `fakeit` - like interface
as proposed by XSPEC.

``` python
import numpyro
from jax.config import config

config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
```

Let's build a model we want to fake and load an observation with the instrumental setup which should be applied

``` python
from jaxspec.model.additive import Powerlaw, Blackbodyrad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.data.observation import Observation

obs = Observation.pha_file('obs_1.pha')
model = Tbabs() * (Powerlaw() + Blackbodyrad())
```

Let's do fakeit for a bunch of parameters

``` python
from numpy.random import default_rng

rng = default_rng(42)

num_params = 10000
parameters = {
    "tbabs_1": {
        "N_H": rng.uniform(0.1, 0.4, size=num_params)
    },
    "powerlaw_1": {
        "alpha": rng.uniform(1, 3, size=num_params),
        "norm": rng.exponential(10 ** (-0.5), size=num_params)
    },
    "blackbodyrad_1": {
        "kT": rng.uniform(0.1, 3.0, size=num_params),
        "norm": rng.exponential(10 ** (-3), size=num_params)
    },
}
```

And now we can fakeit!

``` python
from jaxspec.data.util import fakeit_for_multiple_parameters

spectra = fakeit_for_multiple_parameters(obs, model, parameters)
```

Let's plot some of the resulting spectra

``` python
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))

for i in range(10):

    plt.step(
        obs.out_energies[0],
        spectra[i, :],
        where="post"
    )

plt.xlabel("Energy [keV]")
plt.ylabel("Counts")
plt.loglog()
```

![Some spectra](statics/fakeits.png)
