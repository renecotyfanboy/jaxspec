def test_convergence(get_individual_mcmc_results, get_joint_mcmc_result):
    for result in get_individual_mcmc_results + get_joint_mcmc_result:
        assert result.converged
