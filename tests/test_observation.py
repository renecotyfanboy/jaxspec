from jaxspec.fit import BayesianFitter


def test_sparsify_matrix_in_model(obs_model_prior):
    obsconfigurations, model, prior = obs_model_prior

    for obsconf in obsconfigurations:
        forward_model = BayesianFitter(model, obsconf, background_model=None, sparsify_matrix=True)
        forward_model.fit(prior, num_chains=4, num_warmup=10, num_samples=10, mcmc_kwargs={"progress_bar": False})
