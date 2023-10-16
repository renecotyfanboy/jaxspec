This page demonstrate how to perform a basic spectral fitting operation using JAXspec, using either bayesian or 
frequentist tools. The usual workflow consists in the following steps:

1. [Model building](#model-building) using the available components implemented in JAXspec. The API page shows the 
[additive](../references/additive.md) and [multiplicative](../references/multiplicative.md) components already implemented. It is also 
possible to define your own components. 
2. [Loading the observations](#loading-the-observations)
3. [Perform the fit](#perform-the-fit) using either bayesian or frequentist tools

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

## Perform the fit

Once the model is built, the observations are loaded and the prior distribution is defined, the MCMC can be launched. 
The `fit` method of the `BayesianModel` class takes as argument the prior distribution and the keyword arguments of the 
`numpyro.mcmc.MCMC` class. The `fit` method returns a `Result` object. 

```python
result = forward.fit(prior, mcmc_kwargs={'progress_bar': False})
```

=== "Results"
    | Parameter      | Component      | Value                                               |
    | :------------: | :------------: | --------------------------------------------------- |
    | Normalization  | Powerlaw (1)   | $\left( 412.5^{+5.0}_{-7.9} \right) \times 10^{-6}$ |
    | $\alpha$       | Powerlaw (1)   | $1.844^{+0.018}_{-0.016}$                           |
    | $N_H$          | TbAbs (1)      | $\left( 154.8^{+5.2}_{-5.7} \right) \times 10^{-3}$ |

=== "Source Latex"
    ```latex
    \begin{table}
        \centering
        \caption{Results of the fit}
        \label{tab:results}
        \begin{tabular}{cccc}
            \hline
            Model & powerlaw_1_alpha & powerlaw_1_norm & tbabs_1_N_H \\ 
            \hline
            Chain 0 & $1.844^{+0.018}_{-0.016}$ & $\left( 412.5^{+5.0}_{-7.9} \right) \times 10^{-6}$ & $\left( 154.8^{+5.2}_{-5.7} \right) \times 10^{-3}$ \\ 
            \hline
        \end{tabular}
    \end{table}
    ```

