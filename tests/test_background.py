import matplotlib.pyplot as plt


def test_gp_bkg(obs_model_prior):
    from jaxspec.fit import MCMCFitter
    from jaxspec.model.background import GaussianProcessBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20)
    forward = MCMCFitter(model, prior, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(
        num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False}
    )
    res_1.plot_ppc()
    plt.suptitle("Gaussian Process Background")
    plt.show()


def test_subtract_bkg(obs_model_prior):
    from jaxspec.fit import MCMCFitter
    from jaxspec.model.background import SubtractedBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackground()
    forward = MCMCFitter(model, prior, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(
        num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False}
    )

    res_1.plot_ppc()
    plt.suptitle("Subtracted Background")
    plt.show()


def test_subtract_bkg_with_error(obs_model_prior):
    from jaxspec.fit import MCMCFitter
    from jaxspec.model.background import BackgroundWithError

    obs_list, model, prior = obs_model_prior
    bkg_model = BackgroundWithError()
    forward = MCMCFitter(model, prior, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(
        num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False}
    )

    res_1.plot_ppc()
    plt.suptitle("Subtracted Background with Error")
    plt.show()


def test_spectral_model_background(obs_model_prior):
    import numpyro.distributions as dist

    from jaxspec.fit import MCMCFitter
    from jaxspec.model.additive import Blackbodyrad, Powerlaw
    from jaxspec.model.background import SpectralBackgroundModel

    obs_list, model, prior = obs_model_prior

    spectral_model_background = Powerlaw() + Blackbodyrad()

    prior_background = {
        "powerlaw_1_alpha": dist.Uniform(0, 5),
        "powerlaw_1_norm": dist.LogUniform(1e-8, 1e-3),
        "blackbodyrad_1_kT": dist.Uniform(0, 5),
        "blackbodyrad_1_norm": dist.LogUniform(1e-5, 1e1),
    }

    bkg_model = SpectralBackgroundModel(spectral_model_background, prior_background)
    forward = MCMCFitter(model, prior, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(
        num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False}
    )

    res_1.plot_ppc()
    plt.suptitle("Spectral model Background")
    plt.show()


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
