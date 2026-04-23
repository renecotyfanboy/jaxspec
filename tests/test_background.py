import numpyro.distributions as dist
import pytest

from jaxspec.fit import MCMCFitter
from jaxspec.model.additive import Blackbodyrad, Powerlaw
from jaxspec.model.background import (
    BackgroundWithError,
    SpectralModelBackground,
    SubtractedBackground,
)

spectral_model_background = Powerlaw() + Blackbodyrad()

prior_background = {
    "background.powerlaw_1.alpha": dist.Uniform(0, 5),
    "background.powerlaw_1.norm": dist.LogUniform(1e-7, 1e-3),
    "background.blackbodyrad_1.kT": dist.Uniform(0, 5),
    "background.blackbodyrad_1.norm": dist.LogUniform(1e-5, 1e-1),
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "bkg_model, extra_prior",
    [
        pytest.param(SubtractedBackground(), {}, id="Subtracted background"),
        pytest.param(BackgroundWithError(), {}, id="Subtracted background with errs"),
        pytest.param(
            SpectralModelBackground(spectral_model_background),
            prior_background,
            id="Spectral model background",
        ),
    ],
)
def test_background_model(obs_model_prior, bkg_model, extra_prior):
    obs_list, model, prior = obs_model_prior
    merged_prior = {**prior, **extra_prior}
    forward = MCMCFitter(model, merged_prior, obs_list[0], background_model=bkg_model)
    result = forward.fit(
        num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False}
    )
    result.plot_ppc(title=f"Test {bkg_model.__class__.__name__}")
