from abc import ABC, abstractmethod

import jax
import numpyro
import numpyro.distributions as dist

from flax import nnx
from numpyro.distributions import Poisson

from .abc import SpectralModel


class BackgroundModel(ABC, nnx.Module):
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
        numpyro.deterministic(f"bkg/~/{name}", observed_counts)

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
        countrate = numpyro.sample(f"bkg/~/_{name}_countrate", dist.Gamma(alpha, rate=beta))

        with numpyro.plate(f"bkg/~/{name}_plate", len(observed_counts)):
            numpyro.sample(
                f"bkg/~/{name}", dist.Poisson(countrate), obs=observed_counts if observed else None
            )

        return countrate


class SpectralModelBackground(BackgroundModel):
    def __init__(self, spectral_model: "SpectralModel", prior_distributions, sparse=False):
        self.spectral_model = spectral_model
        self.prior = nnx.data(prior_distributions)
        self.sparse = sparse

    def numpyro_model(self, observation, name: str = "", observed=True):
        from jaxspec.fit._build_model import build_prior, forward_model

        params = build_prior(self.prior, prefix=f"_bkg_{name}_")
        bkg_model = jax.jit(
            lambda par: forward_model(self.spectral_model, par, observation, sparse=self.sparse)
        )
        bkg_countrate = bkg_model(params)

        with numpyro.plate("bkg/~/plate_" + name, len(observation.folded_background)):
            numpyro.sample(
                "bkg/~/" + name,
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
