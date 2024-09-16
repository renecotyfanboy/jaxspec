from jaxspec.fit import MCMCFitter


def test_sparsify_matrix_in_model(obs_model_prior):
    obsconfigurations, model, prior = obs_model_prior

    for obsconf in obsconfigurations:
        forward_model = MCMCFitter(
            model, prior, obsconf, background_model=None, sparsify_matrix=True
        )
        forward_model.fit(
            num_chains=4, num_warmup=10, num_samples=10, mcmc_kwargs={"progress_bar": False}
        )
