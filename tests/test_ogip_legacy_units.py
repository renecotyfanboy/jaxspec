"""Legacy redistribution units preserve weights without hiding area ambiguities."""

import hashlib

import astropy.units as u
import numpy as np
import pytest

from astropy.io import fits
from astropy.units import UnitsWarning
from test_ogip_rmf_groups import write_response

from jaxspec.data import Instrument
from jaxspec.data.ogip import DataRMF


def write_cxc_response(path, **changed_header):
    """Attach the specific primary-documented CXC metadata to a known response."""
    path = write_response(path, matrix_unit="au")
    metadata = {
        "TELESCOP": "CHANDRA",
        "INSTRUME": "ACIS",
        "HDUCLASS": "OGIP",
        "HDUCLAS1": "RESPONSE",
        "HDUCLAS2": "RSP_MATRIX",
        "HDUCLAS3": "REDIST",
    }
    metadata.update(changed_header)
    with fits.open(path, mode="update") as hdus:
        hdus["MATRIX"].header.update(metadata)
    return path


def test_documented_cxc_unit_preserves_values_file_bytes_and_provenance(tmp_path, recwarn):
    path = write_cxc_response(tmp_path / "acis.rmf")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    response = DataRMF.from_file(path)
    assert not recwarn.list
    np.testing.assert_array_equal(
        response.matrix, [[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]]
    )
    assert not response.includes_effective_area
    assert response.matrix_unit_original == "au"
    assert (
        "https://cxc.cfa.harvard.edu/csc1/data_products/usage/" in response.compatibility_notes[0]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert fits.getheader(path, "MATRIX")["TUNIT6"] == "au"


@pytest.mark.parametrize(
    "change",
    [
        {"TELESCOP": "OTHER"},
        {"INSTRUME": "HRC"},
        {"HDUCLASS": "OTHER"},
        {"HDUCLAS1": "OTHER"},
        {"HDUCLAS2": "OTHER"},
    ],
)
def test_unproven_au_context_is_not_silently_reinterpreted(tmp_path, change):
    path = write_cxc_response(tmp_path / "unknown.rmf", **change)
    with pytest.warns((UserWarning, UnitsWarning)) as emitted:
        response = DataRMF.from_file(path)
    assert any("Ignoring RMF MATRIX unit au" in str(item.message) for item in emitted)
    assert not response.includes_effective_area
    assert response.matrix_unit_original == "au"
    assert not response.compatibility_notes
    np.testing.assert_array_equal(
        response.matrix, [[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]]
    )


@pytest.mark.parametrize("storage", ["fixed", "variable"])
@pytest.mark.parametrize("unit", ["none", "counts/photon", "count/photon", "au"])
def test_legacy_redistribution_units_warn_and_preserve_weights(tmp_path, storage, unit):
    path = write_response(tmp_path / "legacy.rmf", storage=storage, matrix_unit=unit)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.warns((UserWarning, UnitsWarning)) as emitted:
        response = DataRMF.from_file(path)
    assert any("Ignoring RMF MATRIX unit" in str(item.message) for item in emitted)
    np.testing.assert_array_equal(
        response.matrix, [[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]]
    )
    assert not response.includes_effective_area
    assert response.matrix_unit_original == unit
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("storage", ["fixed", "variable"])
@pytest.mark.parametrize("unit", ["none", "counts/photon", "count/photon", "au"])
def test_legacy_units_cannot_supply_a_complete_response_scale(tmp_path, storage, unit, recwarn):
    path = write_response(tmp_path / "unknown.rsp", storage=storage, matrix_unit=unit, full=True)
    with pytest.raises(ValueError, match="MATRIX units"):
        DataRMF.from_file(path)


def test_cxc_full_response_cannot_use_redistribution_unit(tmp_path, recwarn):
    path = write_cxc_response(tmp_path / "acis.rsp", HDUCLAS3="FULL")
    with pytest.raises(ValueError, match="MATRIX units"):
        DataRMF.from_file(path)


@pytest.mark.parametrize("unit", ["m", "s", "keV", "AU"])
@pytest.mark.parametrize("full", [False, True])
def test_known_incompatible_matrix_units_remain_errors(tmp_path, unit, full):
    path = write_response(tmp_path / "incompatible.rmf", matrix_unit=unit, full=full)
    with pytest.raises(ValueError, match="MATRIX units"):
        DataRMF.from_file(path)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    "unit,expected,has_area", [(u.m**2, 10000, True), (u.percent, 0.01, False)]
)
def test_recognised_matrix_units_preserve_scale_without_warning(
    complete, unit, expected, has_area, recwarn
):
    response = DataRMF(
        [1],
        [2],
        [1],
        [[0]],
        [[1]],
        [[1]] * unit,
        [0],
        [1],
        [2],
        includes_effective_area=complete,
    )
    np.testing.assert_array_equal(response.matrix, [[expected]])
    assert response.includes_effective_area == (complete or has_area)
    assert not recwarn.list


def test_cxc_redistribution_applies_the_separate_effective_area(tmp_path):
    rmf = write_cxc_response(tmp_path / "acis.rmf")
    area = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="D", unit="keV", array=[1, 2, 3]),
            fits.Column(name="ENERG_HI", format="D", unit="keV", array=[2, 3, 4]),
            fits.Column(name="SPECRESP", format="D", unit="cm2", array=[10, 20, 30]),
        ],
        name="SPECRESP",
    )
    arf = tmp_path / "acis.arf"
    fits.HDUList([fits.PrimaryHDU(), area]).writeto(arf)
    instrument = Instrument.from_ogip_file(rmf, arf)
    np.testing.assert_array_equal(
        instrument.redistribution.data.todense() * instrument.area.data,
        [[0, 0, 22.5], [2.5, 0, 0], [5, 0, 0], [1.25, 0, 0]],
    )
    assert instrument.attrs["response_matrix_unit_original"] == "au"
    assert "dimensionless" in instrument.attrs["response_matrix_compatibility_notes"][0]
