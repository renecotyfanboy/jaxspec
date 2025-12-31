import os

import astropy.units as u
import numpy as np
import pytest

from jaxspec.model.additive import Powerlaw
from jaxspec.util.online_storage import table_manager

xspec = pytest.importorskip("xspec")


@pytest.fixture
def load_jaxspec_data(request, monkeypatch):
    from jaxspec.data import ObsConfiguration, Observation

    file_pha = table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits")
    pn_rmf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf")
    pn_arf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.arf")
    dir_path = os.path.dirname(file_pha)
    monkeypatch.chdir(os.path.join(os.path.dirname(request.fspath.dirname), dir_path))

    low_energy, high_energy = 0.5, 8.0
    folding = ObsConfiguration.from_pha_file(
        file_pha, low_energy=low_energy, high_energy=high_energy
    )
    observation = Observation.from_pha_file(file_pha)

    return folding, observation


@pytest.fixture
def load_xspec_data(request, monkeypatch):
    low_energy, high_energy = 0.5, 8.0

    file_pha = table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits")
    pn_rmf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf")
    pn_arf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.arf")
    dir_path = os.path.dirname(file_pha)
    monkeypatch.chdir(os.path.join(os.path.dirname(request.fspath.dirname), dir_path))

    xspec.AllData.clear()
    xspec.Plot.xAxis = "keV"
    xspec_observation = xspec.Spectrum(file_pha)
    xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

    return xspec_observation


def test_obs_constitantcy(load_xspec_data, load_jaxspec_data):
    file_pha = table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits")
    xspec_observation = load_xspec_data
    folding, observation = load_jaxspec_data

    assert np.isclose(
        xspec_observation.exposure, float(observation.exposure), 0.1
    ), f"Exposure is the same as loaded by XSPEC for {file_pha}"
    assert len(xspec_observation.values) == len(
        folding.coords["folded_channel"]
    ), f"The number of grouped channel is not the same as XSPEC {file_pha}"


def test_bins(load_xspec_data, load_jaxspec_data):
    file_pha = table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits")
    xspec_observation = load_xspec_data
    folding, observation = load_jaxspec_data

    xspec_in_energies = np.vstack(
        [
            np.asarray(xspec_observation.response.energies)[0:-1],
            np.asarray(xspec_observation.response.energies)[1:],
        ]
    )

    xspec_out_energies = np.asarray(xspec_observation.energies).T

    assert np.isclose(
        folding.out_energies, xspec_out_energies
    ).all(), f"The grouped channel energy bins are not the same as XSPEC {file_pha}"
    assert np.isclose(
        folding.in_energies, xspec_in_energies
    ).all(), f"The unfolded channel energy bins are not the same as XSPEC {file_pha}"


def test_flux_computation():
    xspec.AllData.clear()
    xspec.AllModels.clear()

    xspec.AllData.dummyrsp(lowE=0.2, highE=1.7, nBins=10_000)
    m = xspec.Model("powerlaw")
    m.powerlaw.PhoIndex = 2.0
    m.powerlaw.norm = 1.0

    xspec.AllModels.calcFlux("0.5 1.5")

    phflux_xspec = m.flux[3]  # ph/cm^2/s
    eflux_xspec = m.flux[0]  # erg/cm^2/s

    factor = (1 * u.keV).to(u.erg).value
    phflux_jaxspec = Powerlaw().photon_flux(
        {"powerlaw_1_norm": 1.0, "powerlaw_1_alpha": 2.0}, e_low=0.5, e_high=1.5, n_points=10_000
    )

    eflux_jaxspec = (
        Powerlaw().energy_flux(
            {"powerlaw_1_norm": 1.0, "powerlaw_1_alpha": 2.0},
            e_low=0.5,
            e_high=1.5,
            n_points=10_000,
        )
        * factor
    )

    assert np.isclose(
        phflux_xspec, phflux_jaxspec
    ), f"Mismatch between XSPEC and jaxspec on photon flux, got {phflux_xspec} and {phflux_jaxspec}"
    assert np.isclose(
        eflux_xspec, eflux_jaxspec
    ), f"Mismatch between XSPEC and jaxspec on energy flux, got {eflux_xspec} and {eflux_jaxspec}"
