# JAXspec fitting speedrun

In this example, the basic spectral fitting workflow is illustrated on a XMM-Newton observation of the
pulsating ULX candidate from [Quintin & $al.$ (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5485Q/abstract).

``` python
import numpyro

numpyro.enable_x64()
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
```

## Define your model

The first step consists in building your model using the various components available in JAXspec.

``` python
from jaxspec.model.additive import Powerlaw, Blackbody
from jaxspec.model.multiplicative import Tbabs

model = Tbabs() * (Blackbody() + Powerlaw())
```

Which will produce the following model:

```mermaid
graph LR
    7af5aff7-db1a-4c0e-ad58-f5d306d01ac8("Tbabs (1)")
    daa4bc1a-03cd-4ac1-a706-d9fa85b36a34{x}
    28467034-307f-4b72-b2dd-eb68a54438e1("Powerlaw (1)")
    out("Output")
    7af5aff7-db1a-4c0e-ad58-f5d306d01ac8 --> daa4bc1a-03cd-4ac1-a706-d9fa85b36a34
    daa4bc1a-03cd-4ac1-a706-d9fa85b36a34 --> out
    28467034-307f-4b72-b2dd-eb68a54438e1 --> daa4bc1a-03cd-4ac1-a706-d9fa85b36a34

```

## Load your data

The second step consists in defining the data to be fitted.

``` python
from jaxspec.data import ObsConfiguration
obs = ObsConfiguration.from_pha_file('obs_1.pha', low_energy=0.3, high_energy=12)
```

## Perform the inference

``` python
import numpyro.distributions as dist
from jaxspec.fit import NUTSFitter

prior = {
    "powerlaw_1": {
        "alpha": dist.Uniform(0, 10),
        "norm": dist.LogUniform(1e-5, 1e-2)
    },
    "blackbody_1": {
        "kT": dist.Uniform(0, 10),
        "norm": dist.LogUniform(1e-8, 1e-4)
    },
    "tbabs_1":
        {"N_H": dist.Uniform(0, 1)}
}

forward = NUTSFitter(model, obs)
result = forward.fit(prior, num_chains=4, num_warmup=5000, num_samples=5000)
```

## Gather results

Finally, you can print the results, in a LaTeX table for example. The `result.table()`
will return a $\LaTeX$ compilable table. You can also plot the parameter covariances using the `plot_corner` method.

``` python
result.plot_corner()
```

![Corner plot](statics/fitting.png)

You can also plot the posterior predictives

``` python
result.plot_ppc()
```

![Posterior predictive plot](statics/fitting_ppc.png)
