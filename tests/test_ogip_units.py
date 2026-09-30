"""Response folding uses physical energies and areas, not unconverted FITS numbers."""

import astropy.units as u
import numpy as np
import pytest
import sparse

from astropy.io import fits

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataARF, DataPHA, DataRMF


def write_calibrations(directory, *, energy_unit="eV", area_unit="m2"):
    """Write the same two-bin response in independently chosen physical units."""
    energy_factor = u.keV.to(energy_unit) if energy_unit is not None else 1
    area_factor = (u.cm**2).to(area_unit) if area_unit is not None else 1
    low = np.array([1.0, 2.0]) * energy_factor
    high = np.array([2.0, 3.0]) * energy_factor
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="D", unit=energy_unit, array=low),
            fits.Column(name="ENERG_HI", format="D", unit=energy_unit, array=high),
            fits.Column(name="N_GRP", format="I", array=[1, 1]),
            fits.Column(name="F_CHAN", format="I", array=[0, 0]),
            fits.Column(name="N_CHAN", format="I", array=[2, 2]),
            fits.Column(name="MATRIX", format="2D", array=[[0.75, 0.25], [0.25, 0.75]]),
        ],
        name="MATRIX",
    )
    matrix.header["TLMIN4"] = 0
    # Deliberately store the detector axis in another unit than the photon axis.
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=[0, 1]),
            fits.Column(name="E_MIN", format="D", unit="keV", array=[1, 2]),
            fits.Column(name="E_MAX", format="D", unit="keV", array=[2, 3]),
        ],
        name="EBOUNDS",
    )
    # The ARF grid is always keV; equality must compare converted energies.
    area = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="D", unit="keV", array=[1, 2]),
            fits.Column(name="ENERG_HI", format="D", unit="keV", array=[2, 3]),
            fits.Column(
                name="SPECRESP",
                format="D",
                unit=area_unit,
                array=np.array([100, 200]) * area_factor,
            ),
        ],
        name="SPECRESP",
    )
    rmf, arf = directory / "response.rmf", directory / "area.arf"
    fits.HDUList([fits.PrimaryHDU(), matrix, ebounds]).writeto(rmf)
    fits.HDUList([fits.PrimaryHDU(), area]).writeto(arf)
    return rmf, arf


@pytest.mark.parametrize("energy_unit,area_unit", [("eV", "m2"), ("keV", "cm2"), (None, None)])
def test_equivalent_fits_units_give_identical_photon_counts(tmp_path, energy_unit, area_unit):
    rmf, arf = write_calibrations(tmp_path, energy_unit=energy_unit, area_unit=area_unit)
    instrument = Instrument.from_ogip_file(rmf, arf)
    np.testing.assert_array_equal(instrument.e_min_unfolded, [1, 2])
    np.testing.assert_array_equal(instrument.e_max_channel, [2, 3])
    np.testing.assert_allclose(instrument.area, [100, 200], rtol=1e-14)
    observation = ObsConfiguration.mock_from_instrument(instrument, exposure=2)
    # Independent sum: 2 seconds * R_ij * area_j * integrated photon flux_j.
    expected = np.array([350.0, 650.0])
    np.testing.assert_allclose(observation.transfer_matrix.data @ np.array([1, 2]), expected)


def test_quantity_array_inputs_and_exposure_are_converted():
    instrument = Instrument.from_matrix(
        sparse.eye(2),
        np.array([0.01, 0.02]) * u.m**2,
        np.array([1000, 2000]) * u.eV,
        np.array([2000, 3000]) * u.eV,
        np.array([1, 2]) * u.keV,
        np.array([2, 3]) * u.keV,
    )
    np.testing.assert_allclose(instrument.area, [100, 200])
    np.testing.assert_array_equal(instrument.e_min_unfolded, [1, 2])
    assert DataPHA([1, 2], [4, 5], 0.5 * u.ks).exposure == 500.0


@pytest.mark.parametrize("bad_unit", [u.s, u.dimensionless_unscaled])
def test_incompatible_explicit_energy_units_fail_before_relabeling(bad_unit):
    with pytest.raises(ValueError, match="Photon lower energies must have units"):
        Instrument.from_matrix(sparse.eye(1), [1], [1] * bad_unit, [2], [1], [2])
    with pytest.raises(ValueError, match="ARF ENERG_LO must have units"):
        DataARF([1] * bad_unit, [2], [1])


def test_wrong_fits_unit_reports_the_field(tmp_path):
    rmf, arf = write_calibrations(tmp_path)
    with fits.open(rmf, mode="update") as hdus:
        hdus["MATRIX"].header["TUNIT1"] = "s"
    with pytest.raises(ValueError, match="RMF ENERG_LO must have units"):
        DataRMF.from_file(rmf)
    with fits.open(arf, mode="update") as hdus:
        hdus["SPECRESP"].header["TUNIT3"] = "keV"
    with pytest.raises(ValueError, match="ARF SPECRESP must have units"):
        DataARF.from_file(arf)


def test_incompatible_area_and_exposure_units_are_rejected():
    with pytest.raises(ValueError, match="Effective area must have units"):
        Instrument.from_matrix(sparse.eye(1), [1] * u.m, [1], [2], [1], [2])
    with pytest.raises(ValueError, match="PHA exposure must have units"):
        DataPHA([1], [4], 10 * u.cm)


def test_quantity_mock_exposure_preserves_expected_counts():
    instrument = Instrument.from_matrix(sparse.eye(1), [4], [1], [2], [1], [2])
    observed = ObsConfiguration.mock_from_instrument(instrument, exposure=0.5 * u.ks)
    assert float(observed.exposure) == 500
    np.testing.assert_array_equal(observed.transfer_matrix.data.todense(), [[2000]])


@pytest.mark.parametrize("exposure", [0, -1, np.nan, np.inf, [1, 2]])
def test_exposure_validation_is_shared_by_file_and_array_boundaries(exposure):
    with pytest.raises(ValueError, match="exposure must be a finite, positive scalar"):
        DataPHA([1], [4], exposure)
    with pytest.raises(ValueError, match="exposure must be a finite, positive scalar"):
        Observation.from_matrix([4], sparse.eye(1), [1], [0], exposure)
