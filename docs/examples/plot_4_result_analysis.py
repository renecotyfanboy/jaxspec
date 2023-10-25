r"""
# Analyse the results using arviz

JAXspec provides a convenient way to analyse the results of a fit using the [`arviz`](https://python.arviz.org/en/stable/)
library. This library provides powerful tool to explore Bayesian models. In this example, we will show how to use `arviz`
to analyse the results of a fit.
"""

import jaxspec
import arviz as az