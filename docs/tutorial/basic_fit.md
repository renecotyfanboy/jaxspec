This page demonstrate how to perform a basic spectral fitting operation using JAXspec. 

??? tip "Prerequisites for execution"

    ```python
    import numpyro
    from jax.config import config
    import arviz as az
    
    config.update("jax_enable_x64", True)
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)
    ```

## Model building 

The first step consists in building your model using the various components available in JAXspec. Using the 


```python
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model = Tbabs()*Powerlaw()
```


## Loading the observations

```python
from jaxspec.data.util import example_observations as obs_list
from jaxspec.fit import BayesianModel

forward = BayesianModel(model, list(obs_list.values()))
```

## Build a prior distribution for the model


```python
import numpyro.distributions as dist

prior = {
    'powerlaw_1': {
        'alpha': dist.Uniform(0, 10),
        'norm': dist.Exponential(1e4)},
    'tbabs_1': {
        'N_H': dist.Uniform(0, 1)}}
```

## Launch MCMC and gather results

```python
result = forward.fit(prior, mcmc_kwargs={'progress_bar': False})
```
