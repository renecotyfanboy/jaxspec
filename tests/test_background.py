def test_gp_bkg(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import GaussianProcessBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = GaussianProcessBackground(e_min=0.3, e_max=8, n_nodes=20)
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()


def test_subtract_bkg(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import SubtractedBackground

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackground()
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()


def test_subtract_bkg_with_error(obs_model_prior):
    from jaxspec.fit import BayesianModel
    from jaxspec.model.background import SubtractedBackgroundWithError

    obs_list, model, prior = obs_model_prior
    bkg_model = SubtractedBackgroundWithError()
    forward = BayesianModel(model, obs_list, background_model=bkg_model)

    res_1, res_2, res_3 = forward.fit(prior, num_chains=4, num_warmup=1000, num_samples=1000, mcmc_kwargs={"progress_bar": False})

    res_1.plot_ppc()
    res_2.plot_ppc()
    res_3.plot_ppc()
