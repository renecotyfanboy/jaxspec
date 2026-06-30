import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from jaxspec.analysis.compare import plot_corner_comparison
from jaxspec.analysis.results import _compute_denominator, _resolve_y_units


def test_resolve_y_units_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown y_type"):
        _resolve_y_units("bogus", u.keV)


def test_compute_denominator_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown y_type"):
        _compute_denominator("bogus", exposure=1.0, integrated_arf=1.0, xbins=None)


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

    with pytest.raises(NotImplementedError):
        result.luminosity(e_min, e_max, observer_frame=False)

    with pytest.raises(ValueError):
        result.luminosity(e_min, e_max, register=True)

    with pytest.raises(ValueError):
        result.luminosity(e_min, e_max, distance=10 * u.kpc, redshift=0.1, register=True)

    # Distance-only path exercises the distance -> redshift conversion.
    result.luminosity(e_min, e_max, distance=10 * u.kpc)

    result.luminosity(e_min, e_max, redshift=0.1, register=True)

    assert f"derived.luminosity_{e_min:.1f}_{e_max:.1f}" in list(
        result.inference_data.posterior.keys()
    )


@pytest.mark.slow
def test_plot_ppc_min_counts_grouping_mutually_exclusive(get_result_list):
    name, result = next(zip(*get_result_list))
    with pytest.raises(ValueError, match="mutually exclusive"):
        result.plot_ppc(min_counts=10, grouping=10)
