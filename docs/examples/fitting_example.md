# `jaxspec` fitting speedrun

In this example, the basic spectral fitting workflow is illustrated on a XMM-Newton observation of the
pulsating candidate NGC 7793 ULX-4 from [Quintin & $al.$ (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5485Q/abstract).

``` python
import numpyro

numpyro.enable_x64()
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
```

## Define your model

The first step consists in building a model using various components available in `jaxspec`.

``` python
from jaxspec.model.additive import Powerlaw, Blackbodyrad
from jaxspec.model.multiplicative import Tbabs

spectral_model = Tbabs()*(Powerlaw() + Blackbodyrad())
```

Which will produce the following model:

```mermaid
graph LR
    d2ec7634-2cde-4173-9be2-6a9d1cf9fa41("Tbabs (1)")
    2dc512bd-fd4c-4e8d-ac5a-adfa5b90f3a6{"$$\times$$"}
    2e1ceb31-75f1-4514-adb2-b07de972a22a("Powerlaw (1)")
    1d55d73e-7e10-4180-b96f-e5c1327ef037{"$$+$$"}
    d2c16179-e1f5-4dc9-8dee-6757f279b9e2("Blackbodyrad (1)")
    out("Output")
    d2ec7634-2cde-4173-9be2-6a9d1cf9fa41 --> 2dc512bd-fd4c-4e8d-ac5a-adfa5b90f3a6
    2dc512bd-fd4c-4e8d-ac5a-adfa5b90f3a6 --> out
    2e1ceb31-75f1-4514-adb2-b07de972a22a --> 1d55d73e-7e10-4180-b96f-e5c1327ef037
    1d55d73e-7e10-4180-b96f-e5c1327ef037 --> 2dc512bd-fd4c-4e8d-ac5a-adfa5b90f3a6
    d2c16179-e1f5-4dc9-8dee-6757f279b9e2 --> 1d55d73e-7e10-4180-b96f-e5c1327ef037
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
from jaxspec.fit import MCMCFitter

obsconf = load_example_obsconf("NGC7793_ULX4_PN")

prior = {
    "powerlaw_1_alpha": dist.Uniform(0, 5),
    "powerlaw_1_norm": dist.LogUniform(1e-5, 1e-2),
    "blackbodyrad_1_kT": dist.Uniform(0, 5),
    "blackbodyrad_1_norm": dist.LogUniform(1e-2, 1e2),
    "tbabs_1_N_H": dist.Uniform(0, 1)
}

forward = MCMCFitter(model, obs)
result = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000)
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
