import os

import numpy as np
import pytest

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
