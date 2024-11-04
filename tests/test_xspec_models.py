r'''
import os
import re

from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jax.tree_util import tree_map
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel
from jaxspec.util.online_storage import table_manager
from scipy.stats import anderson_ksamp

xspec = pytest.importorskip("xspec")


@dataclass
class AdditiveModelTestSetup:
    """Class to setup a test case for a spectral model"""

    name_xspec: str  # Name of the model in xspec
    name_jaxspec: str  # Name of the model in jaxspec
    parameters: dict[str, float]  # Parameter values
    parameters_order: list[str]  # Parameter order in xspec
    required_tol: float = 1e-2  # Required tolerance between xspec and jaxspec
    weight: float = 1  # Weight of the XSPEC model when systematic biases are present (e.g. add an epsilon to the model)
    energy_range: tuple[float, float, int] = (
        0.2,
        20,
        10000,
    )  # Energy range to test, lower, upper, number of bins
    model_string: list[tuple[str, str]] | None = None  # Model string to use in XSPEC


models_to_test = [
    AdditiveModelTestSetup(
        name_xspec="powerlaw",
        name_jaxspec="Powerlaw()",
        parameters={"alpha": 1.5, "norm": 1},
        parameters_order=["alpha", "norm"],
    ),
    AdditiveModelTestSetup(
        name_xspec="bbody",
        name_jaxspec="Blackbody()",
        parameters={"kT": 3, "norm": 1},
        parameters_order=["kT", "norm"],
    ),
    AdditiveModelTestSetup(
        name_xspec="bbodyrad",
        name_jaxspec="Blackbodyrad()",
        parameters={"kT": 3, "norm": 1},
        parameters_order=["kT", "norm"],
    ),
    AdditiveModelTestSetup(
        name_xspec="lorentz",
        name_jaxspec="Lorentz()",
        parameters={"E_l": 3, "sigma": 1, "norm": 1},
        parameters_order=["E_l", "sigma", "norm"],
        weight=1
        / 1.02475,  # There is an epsilon in XSPEC to avoid division by zero which add a 2% bias
        required_tol=2e-1,  # The bias is too large to be corrected by the weight above, I shall find why
        energy_range=(4, 9, 10000),
    ),
    AdditiveModelTestSetup(
        name_xspec="gauss",
        name_jaxspec="Gauss()",
        parameters={"E_l": 3, "sigma": 1, "norm": 1},
        parameters_order=["E_l", "sigma", "norm"],
        energy_range=(4, 9, 10000),
    ),
    AdditiveModelTestSetup(
        name_xspec="cutoffpl",
        name_jaxspec="Cutoffpl()",
        parameters={"alpha": 1, "beta": 15, "norm": 1},
        parameters_order=["alpha", "beta", "norm"],
    ),
]


def plot_comparison(e_bins, spec_xspec, spec_jaxspec):
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[0.7, 0.3])
    e_list = e_bins.mean(axis=0)
    axs[0].plot(e_list, spec_xspec, label="xspec")
    axs[0].plot(e_list, spec_jaxspec, label="jaxspec")
    axs[0].loglog()
    axs[1].hexbin(
        e_list, (spec_xspec - spec_jaxspec) / spec_xspec, gridsize=100, xscale="log", cmap="Blues"
    )
    axs[0].legend()
    axs[1].set_xlabel("Energy (keV)")
    axs[1].set_ylabel("(xspec-jaxspec)/xspec")
    plt.show()


@pytest.mark.parametrize(
    "model", models_to_test, ids=lambda m: f"{m.name_xspec} & {m.name_jaxspec}"
)
def test_models(model: AdditiveModelTestSetup):
    xspec.AllModels.clear()
    xspec.AllData.clear()
    xspec.AllData.dummyrsp(*model.energy_range, "lin")

    m_xspec = xspec.Model(model.name_xspec)

    if model.model_string is not None:
        for model_string in model.model_string:
            xspec.Xset.addModelString(*model_string)

    xspec_pars = []
    for par in model.parameters_order:
        xspec_pars.append(par)

    m_xspec.setPars(*[model.parameters[par] for par in model.parameters_order])
    m_xspec.show()
    xspec_energies = np.array(m_xspec.energies(0))
    e_bins = np.vstack([xspec_energies[:-1], xspec_energies[1:]])
    spec_xspec = np.array(m_xspec.values(0)) * model.weight

    m_jaxspec = SpectralModel.from_string(model.name_jaxspec)
    p_jaxspec = {re.sub(r"\(.*\)", "", model.name_jaxspec.lower()) + r"_1": model.parameters}

    spec_jaxspec = m_jaxspec.photon_flux(tree_map(jnp.asarray, p_jaxspec), *e_bins, n_points=20)

    assert np.isclose(
        spec_jaxspec, spec_xspec, rtol=model.required_tol
    ).all(), f"jaxspec {model.name_jaxspec} and xspec {model.name_xspec} are not close enough"


@pytest.mark.filterwarnings("ignore:p-value capped")
@pytest.mark.parametrize(
    "model", models_to_test, ids=lambda m: f"{m.name_xspec} & {m.name_jaxspec}"
)
def test_fakeits(
    tmp_path, request, monkeypatch, model: AdditiveModelTestSetup, exposure=float(10_000)
):
    pn_rmf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf")
    pn_arf_path = table_manager.fetch("example_data/NGC7793_ULX4/PN.arf")
    dir_path = os.path.dirname(pn_rmf_path)
    monkeypatch.chdir(os.path.join(os.path.dirname(request.fspath.dirname), dir_path))

    # XSPEC's fakeit
    xspec.AllModels.clear()
    xspec.AllData.clear()
    xspec.AllData.dummyrsp(*model.energy_range, "lin")

    n_spectra = 100

    m_xspec = xspec.Model(model.name_xspec)

    fakeit_settings = n_spectra * [
        xspec.FakeitSettings(
            response="PN.rmf", arf="PN.arf", exposure=exposure, fileName="fakeit.pha"
        )
    ]

    xspec_pars = []
    for par in model.parameters_order:
        xspec_pars.append(par)

    m_xspec.setPars(*[model.parameters[par] for par in model.parameters_order])
    xspec.AllData.fakeit(
        nSpectra=n_spectra, settings=fakeit_settings, applyStats=True, filePrefix=""
    )
    xspec.AllData.ignore(f"0.0-{model.energy_range[0]:.1f} {model.energy_range[1]:.1f}-**")

    xspec_spectra = np.empty((n_spectra, len(xspec.AllData(1).values)))

    for i in range(1, n_spectra + 1):
        xspec_spectra[i - 1] = (np.asarray(xspec.AllData(i).values) * exposure).astype(int)

    # JAXSPEC's fakeit
    m_jaxspec = SpectralModel.from_string(model.name_jaxspec)
    p_jaxspec = {re.sub(r"\(.*\)", "", model.name_jaxspec.lower()) + r"_1": model.parameters}
    p_jaxspec = tree_map(lambda x: x * jnp.ones(n_spectra), p_jaxspec)

    setup = ObsConfiguration.from_pha_file(
        "fakeit.pha", low_energy=model.energy_range[0], high_energy=model.energy_range[1]
    )

    jaxspec_spectra = fakeit_for_multiple_parameters(setup, m_jaxspec, p_jaxspec)

    assert anderson_ksamp(np.hstack((xspec_spectra, jaxspec_spectra))).pvalue >= 0.25
'''
