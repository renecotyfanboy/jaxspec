## Conjugate Background

To overcome this issue, we came up with an implementation of what we refer as the _Conjugate Background_. This is a
fairly simple and fast way to integrate the uncertainty of the measured background in the spectral parameters
determination. The idea is very similar to supposing that the number of photon observed in each bin is realisation of
a Poisson random variable $\mathcal{P}(\lambda_{\text{Bkg}})$. Except this time we assume that the expected rate
$\lambda_{\text{Bkg}}$ is such that $\lambda \sim \Gamma(\alpha, \beta)$. In `jaxspec`, we chose to fix the prior
parameters to $\alpha = \text{Counts}_{\text{Bkg}} + 1$ and $\beta = 1$. Doing so, the prior distribution is such that
$\mathbb{E}[\lambda_{\text{Bkg}}] = \text{Counts}_{\text{Bkg}} +1$ and $\text{Var}[\lambda_{\text{Bkg}}] =
\text{Counts}_{\text{Bkg}}+1$.

In this situation, the $\Gamma$ is a conjugate
prior of a Poisson likelihood, which gives an analytical solution to the Bayesian inference problem in each bin of the
background. Indeed, one can show that the posterior distribution of the expected countrate has the following form:

$$ p(\lambda_{\text{Bkg}}) \sim \Gamma \left( \alpha, \beta \right) \implies
p\left(\lambda_{\text{Bkg}} | \text{Counts}_{\text{Bkg}}\right) \sim \text{NB}\left(\alpha, \frac{\beta}{\beta +1}
\right) $$

where $\text{NB}$ is a Negative Binomial distribution, which can be directly sampled from when during the fitting
procedure.
