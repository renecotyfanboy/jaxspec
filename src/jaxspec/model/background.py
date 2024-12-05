from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax.scipy.integrate import trapezoid
from numpyro.distributions import Poisson
from tinygp import GaussianProcess, kernels

from .._fit._build_model import build_prior, forward_model
from .abc import SpectralModel


class BackgroundModel(ABC):
    """
    Handles the background modelling in our spectra. This is handled in a separate class for now
    since backgrounds can be phenomenological models fitted directly on the folded spectrum. This is not the case for
    the source model, which is fitted on the unfolded spectrum. This might be changed later.
    """

    @abstractmethod
    def numpyro_model(self, observation, name: str = "", observed=True):
        """
        Build the model for the background.
        """
        pass


class SubtractedBackground(BackgroundModel):
    """
    Define a model where the observed background is simply subtracted from the observed.

    !!! danger

        This is not a good way to model the background, as it does not account for the fact that the measured
        background is a Poisson realisation of the true background's countrate. This is why we prefer a
        [`ConjugateBackground`][jaxspec.model.background.ConjugateBackground].

    """

    def numpyro_model(self, observation, name: str = "", observed=True):
        _, observed_counts = observation.out_energies, observation.folded_background.data
        numpyro.deterministic(f"bkg_{name}", observed_counts)

        return observed_counts


class BackgroundWithError(BackgroundModel):
    """
    Define a model where the observed background is subtracted from the observed accounting for its intrinsic spread. It
    fits a countrate for each background bin assuming a Poisson distribution.
    """

    def numpyro_model(self, obs, name: str = "", observed=True):
        # We can't use the build_prior_function method here because the parameter size varies
        # with the current observation. It must be instantiated in place.
        # Gamma in numpyro is parameterized by concentration and rate (alpha/beta)

        _, observed_counts = obs.out_energies, obs.folded_background.data
        alpha = observed_counts + 1
        beta = 1
        countrate = numpyro.sample(f"_bkg_{name}_countrate", dist.Gamma(alpha, rate=beta))

        with numpyro.plate(f"bkg_{name}_plate", len(observed_counts)):
            numpyro.sample(
                f"bkg_{name}", dist.Poisson(countrate), obs=observed_counts if observed else None
            )

        return countrate


class GaussianProcessBackground(BackgroundModel):
    """
    Define a Gaussian Process to model the background. The GP is built using the
    [`tinygp`](https://tinygp.readthedocs.io/en/stable/guide.html) library.
    """

    def __init__(
        self,
        e_min: float,
        e_max: float,
        n_nodes: int = 30,
        kernel: kernels.Kernel = kernels.Matern52,
    ):
        """
        Build the Gaussian Process background model.

        Parameters:
            e_min: The lower bound of the energy range.
            e_max: The upper bound of the energy range.
            n_nodes: The number of nodes used by the GP, must be lower than the number of channel.
            kernel: The kernel used by the GP.
        """
        self.e_min = e_min
        self.e_max = e_max
        self.n_nodes = n_nodes
        self.kernel = kernel

    def numpyro_model(self, obs, name: str = "", observed=True):
        energy, observed_counts = obs.out_energies, obs.folded_background.data

        if (observed_counts is not None) and (self.n_nodes >= len(observed_counts)):
            raise RuntimeError(
                "More nodes than channels in the observation associated with GaussianProcessBackground."
            )

        else:
            observed_counts = jnp.asarray(observed_counts)

        # The parameters of the GP model
        mean = numpyro.sample(
            f"_bkg_{name}_mean", dist.Normal(jnp.log(jnp.mean(observed_counts)), 2.0)
        )
        sigma = numpyro.sample(f"_bkg_{name}_sigma", dist.HalfNormal(3.0))
        rho = numpyro.sample(f"_bkg_{name}_rho", dist.HalfNormal(10.0))

        # Set up the kernel and GP objects
        kernel = sigma**2 * self.kernel(rho)
        nodes = jnp.linspace(0, 1, self.n_nodes)
        gp = GaussianProcess(kernel, nodes, diag=1e-5 * jnp.ones_like(nodes), mean=mean)

        log_rate = numpyro.sample(f"_bkg_{name}_log_rate_nodes", gp.numpyro_dist())

        interp_count_rate = jnp.exp(
            jnp.interp(energy, nodes * (self.e_max - self.e_min) + self.e_min, log_rate)
        )
        count_rate = trapezoid(interp_count_rate, energy, axis=0)

        # Finally, our observation model is Poisson
        with numpyro.plate("bkg_plate_" + name, len(observed_counts)):
            numpyro.sample(
                f"bkg_{name}", dist.Poisson(count_rate), obs=observed_counts if observed else None
            )

        return count_rate


class SpectralModelBackground(BackgroundModel):
    def __init__(self, spectral_model: "SpectralModel", prior_distributions, sparse=False):
        self.spectral_model = spectral_model
        self.prior = prior_distributions
        self.sparse = sparse

    def numpyro_model(self, observation, name: str = "", observed=True):
        params = build_prior(self.prior, prefix=f"_bkg_{name}_")
        bkg_model = jax.jit(
            lambda par: forward_model(self.spectral_model, par, observation, sparse=self.sparse)
        )
        bkg_countrate = bkg_model(params)

        with numpyro.plate("bkg_plate_" + name, len(observation.folded_background)):
            numpyro.sample(
                "bkg_" + name,
                Poisson(bkg_countrate),
                obs=observation.folded_background.data if observed else None,
            )

        return bkg_countrate


'''
class ConjugateBackground(BackgroundModel):
    r"""
    This class fit an expected rate $\\lambda$ in each bin of the background spectrum. Assuming a Gamma prior
    distribution, we can analytically derive the posterior as a Negative binomial distribution.

    $$ p(\\lambda_{\text{Bkg}}) \\sim \\Gamma \\left( \alpha, \beta \right) \\implies
    p\\left(\\lambda_{\text{Bkg}} | \text{Counts}_{\text{Bkg}}\right) \\sim \text{NB}\\left(\alpha, \frac{\beta}{\beta +1}
    \right) $$

    !!! info
        Here, $\alpha$ and $\beta$ are set to $\alpha = \text{Counts}_{\text{Bkg}} + 1$ and $\beta = 1$. Doing so,
        the prior distribution is such that $\\mathbb{E}[\\lambda_{\text{Bkg}}] = \text{Counts}_{\text{Bkg}} +1$ and
        $\text{Var}[\\lambda_{\text{Bkg}}] = \text{Counts}_{\text{Bkg}}+1$. The +1 is to avoid numerical issues when the
        counts are 0, and add a small scatter even if the measured background is effectively null.

    ??? abstract "References"

        - https://en.wikipedia.org/wiki/Conjugate_prior
        - https://www.acsu.buffalo.edu/~adamcunn/probability/gamma.html
        - https://bayesiancomputationbook.com/markdown/chp_01.html?highlight=conjugate#conjugate-priors
        - https://vioshyvo.github.io/Bayesian_inference/conjugate-distributions.html

    """

    def numpyro_model(self, energy, observed_counts, name: str = "bkg", observed=True):
        # Gamma in numpyro is parameterized by concentration and rate (alpha/beta)
        # alpha = observed_counts + 1
        # beta = 1

        with numpyro.plate(f"{name}_plate", len(observed_counts)):
            countrate = numpyro.sample(f"{name}", dist.Gamma(2 * observed_counts + 1, 2), obs=None)

        return countrate
'''
