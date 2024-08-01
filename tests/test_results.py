import matplotlib.pyplot as plt

from jaxspec.analysis.compare import plot_corner_comparison


def test_plot_ppc(request, get_result_list):
    for name, result in zip(*get_result_list):
        result.plot_ppc(percentile=(5, 95))
        plt.suptitle(f"{request.node.name} {name}")
        plt.show()


def test_plot_ppc_units(request, get_result_list):
    for name, result in zip(*get_result_list):
        for x_unit in ["angstrom", "keV", "Hz", "nm"]:
            result.plot_ppc(percentile=(5, 95), x_unit=x_unit)
            plt.suptitle(f"{request.node.name} {name}")
            plt.close()


def test_plot_ppc_dtypes(request, get_result_list):
    for name, result in zip(*get_result_list):
        for y_type in ["counts", "countrate", "photon_flux", "photon_flux_density"]:
            result.plot_ppc(percentile=(5, 95), y_type=y_type)
            plt.suptitle(f"{request.node.name} {name}")
            plt.close()


def test_plot_corner(request, get_result_list):
    for name, result in zip(*get_result_list):
        result.plot_corner()
        plt.suptitle(f"{request.node.name} {name}")
        plt.show()


def test_table(request, get_result_list):
    print(request.node.name)
    for name, result in zip(*get_result_list):
        print(result.table())


def test_compare(request, get_result_list):
    plot_corner_comparison({name: res for name, res in zip(*get_result_list)})
    plt.suptitle(request.node.name)
    plt.show()


def test_posterior_photon_flux(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2
    result.photon_flux(e_min, e_max, register=True)
    assert f"photon_flux_{e_min:.1f}_{e_max:.1f}" in list(result.inference_data.posterior.keys())


def test_posterior_energy_flux(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2
    result.energy_flux(e_min, e_max, register=True)
    assert f"energy_flux_{e_min:.1f}_{e_max:.1f}" in list(result.inference_data.posterior.keys())


def test_posterior_luminosity(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2
    result.luminosity(e_min, e_max, register=True)
    assert f"luminosity_{e_min:.1f}_{e_max:.1f}" in list(result.inference_data.posterior.keys())
