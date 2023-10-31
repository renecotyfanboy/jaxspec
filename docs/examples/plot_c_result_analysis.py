r"""
# Analyse the results using arviz

JAXspec provides a convenient way to analyse the results of a fit using the [`arviz`](https://python.arviz.org/en/stable/)
library. This library provides powerful tool to explore Bayesian models. In this example, we will show how to use `arviz`
to analyse the results of a fit.
"""

from doc_variable import result  # Loading a result object from another notebook
import arviz as az

# %% New cell
# From the result object, you can access the `inference_data` attribute, which is an `arviz.InferenceData` object.
# This leverage the use of every arviz function to analyse the results of the fit.
# # Trace plot
# This visualization is useful to see the evolution of the parameters during the sampling process. It can be used to
# diagnose convergence issues. The ideal situation is when the chains are well mixed and randomly scattered around the
# target distribution. If instead, chains are stuck in some region of the parameter space, or show some trends, this
# might indicate that the sampler did not explore the full parameter space.
with az.style.context("arviz-darkgrid", after_reset=True):
    az.plot_trace(result.inference_data, compact=False)

# %% New cell
# A more quantitative way to assess the convergence of the chains is to use the `summary` function. This function
# provides a summary of the posterior distribution of the parameters, including the mean, the standard deviation, and
# the 95% highest posterior density interval.

az.summary(result.inference_data.posterior)

# %% New cell
# The `r_hat` column provides a measure of the convergence of the chains. The closer to 1, the better. A value larger
# than 1.1 is a sign of convergence issues. This statistic can be directly computed using the `r_hat` function.
# ??? info
#     The `r_hat` statistic computed by `arviz` is proposed by
#     [Vehtari et al. (2019)](https://arxiv.org/abs/1903.08008).

rhat = az.rhat(result.inference_data.posterior)
print(rhat)

# %% New cell
# # Pair plot
# This visualization is useful to see the correlation between the parameters. The ideal situation is when the
# parameters are uncorrelated, which means that the posterior distribution is close to a multivariate Gaussian
# distribution.
with az.style.context("arviz-darkgrid", after_reset=True):
    az.plot_pair(result.inference_data)

# %% New cell
# Take a look at [arviz's documentation](https://python.arviz.org/en/stable/examples/index.html) to see what else you
# can do with this library.
