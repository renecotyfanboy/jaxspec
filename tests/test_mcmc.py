def test_convergence(get_individual_mcmc_results, get_joint_result):
    for result in get_individual_mcmc_results + get_joint_result:
        assert result.converged
