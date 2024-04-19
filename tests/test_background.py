import matplotlib.pyplot as plt


def test_gp_bkg(obs_model_prior):
    from jaxspec.fit import BayesianFitter
    from jaxspec.model.background import GaussianProcessBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20)
    forward = BayesianFitter(model, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(prior, num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False})
    res_1.plot_ppc()
    plt.suptitle("Gaussian Process Background")
    plt.show()


def test_subtract_bkg(obs_model_prior):
    from jaxspec.fit import BayesianFitter
    from jaxspec.model.background import SubtractedBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackground()
    forward = BayesianFitter(model, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(prior, num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    plt.suptitle("Subtracted Background")
    plt.show()


def test_subtract_bkg_with_error(obs_model_prior):
    from jaxspec.fit import BayesianFitter
    from jaxspec.model.background import BackgroundWithError

    obs_list, model, prior = obs_model_prior
    bkg_model = BackgroundWithError()
    forward = BayesianFitter(model, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(prior, num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    plt.suptitle("Subtracted Background with Error")
    plt.show()


"""
def test_conjugate_bkg(obs_model_prior):
    from jaxspec.fit import BayesianFitter
    from jaxspec.model.background import ConjugateBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = ConjugateBackground()
    forward = BayesianFitter(model, obs_list[0], background_model=bkg_model)

    res_1 = forward.fit(prior, num_chains=4, num_warmup=100, num_samples=100, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
"""
