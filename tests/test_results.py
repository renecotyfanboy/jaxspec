import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from jaxspec.analysis.compare import plot_corner_comparison


@pytest.mark.slow
def test_plot_ppc(get_result_list):
    for name, result in zip(*get_result_list):
        result.plot_ppc(plot_components=True, plot_background=False)


@pytest.mark.slow
@pytest.mark.parametrize("rebin", ["min_counts", "grouping", "both"])
def test_plot_rebin(get_result_list, rebin):
    name, result = next(zip(*get_result_list))

    if rebin != "both":
        result.plot_ppc(plot_components=True, plot_background=False, **{rebin: 10})

    else:
        with pytest.raises(TypeError):
            result.plot_ppc(plot_components=True, plot_background=False, rebin=10, min_counts=10)


@pytest.mark.slow
@pytest.mark.parametrize("scale", ["linear", "semilogx", "semilogy", "loglog"])
def test_plot_scales(get_result_list, scale):
    name, result = next(zip(*get_result_list))
    result.plot_ppc(plot_components=True, plot_background=False, scale=scale)


@pytest.mark.slow
def test_plot_ppc_components(get_result_list):
    for name, result in zip(*get_result_list):
        result.plot_ppc(n_sigmas=2)


@pytest.mark.slow
@pytest.mark.parametrize("x_unit", ["angstrom", "keV", "Hz", "nm", "attoGauss"])
def test_plot_ppc_units(get_result_list, x_unit):
    for name, result in zip(*get_result_list):
        if x_unit == "attoGauss":
            with pytest.raises(ValueError):
                result.plot_ppc(x_unit=x_unit)

        else:
            result.plot_ppc(x_unit=x_unit)


@pytest.mark.slow
@pytest.mark.parametrize("y_type", ["counts", "countrate", "photon_flux", "photon_flux_density"])
def test_plot_ppc_dtypes(get_result_list, y_type):
    for name, result in zip(*get_result_list):
        result.plot_ppc(y_type=y_type)


@pytest.mark.slow
def test_plot_corner(get_result_list):
    for name, result in zip(*get_result_list):
        result.plot_corner()


@pytest.mark.slow
def test_to_chain(get_result_list):
    for name, result in zip(*get_result_list):
        result.to_chain(name)


@pytest.mark.slow
def test_table(request, get_result_list):
    print(request.node.name)
    for name, result in zip(*get_result_list):
        print(result.table())


@pytest.mark.slow
def test_compare(request, get_result_list):
    plot_corner_comparison({name: res for name, res in zip(*get_result_list)})
    plt.suptitle(request.node.name)
    plt.show()


@pytest.mark.slow
def test_posterior_photon_flux(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2
    result.photon_flux(e_min, e_max, register=True)
    assert f"derived.photon_flux_{e_min:.1f}_{e_max:.1f}" in list(
        result.inference_data.posterior.keys()
    )


@pytest.mark.slow
def test_posterior_energy_flux(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2
    result.energy_flux(e_min, e_max, register=True)
    assert f"derived.energy_flux_{e_min:.1f}_{e_max:.1f}" in list(
        result.inference_data.posterior.keys()
    )


@pytest.mark.slow
def test_posterior_luminosity(get_joint_mcmc_result):
    result = get_joint_mcmc_result[0]
    e_min, e_max = 0.7, 1.2

    with pytest.raises(ValueError):
        result.luminosity(e_min, e_max, register=True)

    with pytest.raises(ValueError):
        result.luminosity(e_min, e_max, distance=10 * u.kpc, redshift=0.1, register=True)

    result.luminosity(e_min, e_max, redshift=0.1, register=True)

    assert f"derived.luminosity_{e_min:.1f}_{e_max:.1f}" in list(
        result.inference_data.posterior.keys()
    )
