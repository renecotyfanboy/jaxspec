import numpyro.distributions as dist
import pytest

from jaxspec.fit import MCMCFitter
from jaxspec.model.additive import Blackbodyrad, Powerlaw
from jaxspec.model.background import (
    BackgroundWithError,
    GaussianProcessBackground,
    SpectralModelBackground,
    SubtractedBackground,
)

spectral_model_background = Powerlaw() + Blackbodyrad()

prior_background = {
    "powerlaw_1_alpha": dist.Uniform(0, 5),
    "powerlaw_1_norm": dist.LogUniform(1e-8, 1e-3),
    "blackbodyrad_1_kT": dist.Uniform(0, 5),
    "blackbodyrad_1_norm": dist.LogUniform(1e-6, 1e-1),
}


@pytest.mark.parametrize(
    "bkg_model",
    [
        pytest.param(GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20), id="GP background"),
        pytest.param(SubtractedBackground(), id="Subtracted background"),
        pytest.param(BackgroundWithError(), id="Subtracted background with errs"),
        pytest.param(
            SpectralModelBackground(spectral_model_background, prior_background),
            id="Spectral model background",
        ),
    ],
)
def test_background_model(obs_model_prior, bkg_model):
    obs_list, model, prior = obs_model_prior
    forward = MCMCFitter(model, prior, obs_list[0], background_model=bkg_model)
    result = forward.fit(
        num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False}
    )
    result.plot_ppc(title=f"Test {bkg_model.__class__.__name__}")


"""
def test_conjugate_bkg(obs_model_prior):
    from jaxspec.fit import MCMCFitter
    from jaxspec.model.background import ConjugateBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = ConjugateBackground()
    forward = MCMCFitter(model, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(prior, num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
"""
