# Add a background model to your fit

Most of the time, the X-ray spectrum is extracted around the source, and
an additional spectrum is extracted further away to get a *representative*
X-ray background in the vicinity of your observation. With `jaxspec`, this
spectrum is automatically loaded as long as it is defined in the header of
your observation.

To include this in your spectral fitting, you should add a `BackgroundModel` to your code as follows. The simplest
approach is equivalent to subtract the background to the observed spectrum when performing the fit.

``` python
from jaxspec.model.background import SubtractedBackground

forward = NUTSFitter(model, obs, background_model=SubtractedBackground())
result = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": True})

result.plot_ppc()
```

![Subtracted background](statics/subtract_background.png)

The `SubtractedBackground` simply account for the observed counts without propagating any kind of dispersion, which
is clearly bad when we want to get spectral parameters with a comprehensive error budget. The simplest way to deal with
it is to consider each background bin as a Poisson realisation of a counting process, which is achieved here using
`BackgroundWithError`.

``` python
from jaxspec.model.background import BackgroundWithError

forward = NUTSFitter(model, obs, background_model=BackgroundWithError())
result = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": True})

result.plot_ppc()
```

![Subtracted background with errors](statics/subtract_background_with_errors.png)

The other way to deal with this is to directly fit a Gaussian process on the folded background spectrum to integrate
energy and bins correlations in a very empiric way. This can be done using a `GaussianProcessBackground`. The number of
nodes will drive the flexibility of the Gaussian process, and it should always be lower than the number of channels.

``` python
from jaxspec.model.background import GaussianProcessBackground

forward = NUTSFitter(model, obs, background_model=GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20))
result = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": True})

result.plot_ppc()
```

![Subtracted background with errors](statics/background_gp.png)