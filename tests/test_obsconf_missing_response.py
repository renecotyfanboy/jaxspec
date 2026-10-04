"""A PHA without a response should explain how to supply the missing calibration."""

import numpy as np
import pytest

from astropy.io import fits

from jaxspec.data import ObsConfiguration


def write_pha(path, response):
    """Write a two-channel spectrum, optionally naming a response in its header."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[0, 1]),
            fits.Column(name="COUNTS", format="J", array=[2, 3]),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10.0, GROUPING=0, QUALITY=0, BACKSCAL=1.0, AREASCAL=1.0)
    if response is not None:
        spectrum.header["RESPFILE"] = response
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    return path


def write_rsp(path):
    """Write a combined response with diagonal effective areas of 10 and 20 cm²."""
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", unit="keV", array=[1, 2]),
            fits.Column(name="ENERG_HI", format="E", unit="keV", array=[2, 3]),
            fits.Column(name="N_GRP", format="J", array=[1, 1]),
            fits.Column(name="F_CHAN", format="J", array=[0, 0]),
            fits.Column(name="N_CHAN", format="J", array=[2, 2]),
            fits.Column(name="MATRIX", format="2E", array=[[10, 0], [0, 20]]),
        ],
        name="SPECRESP MATRIX",
    )
    matrix.header.update(TLMIN4=0, DETCHANS=2, HDUCLAS3="FULL")
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=[0, 1]),
            fits.Column(name="E_MIN", format="E", unit="keV", array=[1, 2]),
            fits.Column(name="E_MAX", format="E", unit="keV", array=[2, 3]),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), matrix, ebounds]).writeto(path)
    return path


@pytest.mark.parametrize(
    ("header_response", "explicit_response"),
    [
        (None, None),
        ("", None),
        (" ", None),
        (" NONE ", None),
        ("none", None),
        (None, ""),
        (None, "   "),
        ("valid.rsp", ""),
    ],
)
def test_absent_response_explains_how_to_supply_one(tmp_path, header_response, explicit_response):
    """Missing header values and blank explicit paths produce an actionable error."""
    pha = write_pha(tmp_path / "source.pha", header_response)
    write_rsp(tmp_path / "valid.rsp")
    with pytest.raises(ValueError) as error:
        ObsConfiguration.from_pha_file(pha, rmf_path=explicit_response)
    message = str(error.value)
    assert str(pha) in message
    assert "RESPFILE" in message
    assert "rmf_path=" in message


@pytest.mark.parametrize("header_response", [None, "unavailable.rmf"])
@pytest.mark.parametrize("as_string", [False, True])
def test_explicit_response_overrides_missing_header_without_requiring_arf(
    tmp_path, header_response, as_string
):
    """An external combined RSP works even when the PHA has no usable response link."""
    pha = write_pha(tmp_path / "source.pha", header_response)
    response = write_rsp(tmp_path / " response.rsp ")
    config = ObsConfiguration.from_pha_file(pha, rmf_path=str(response) if as_string else response)
    np.testing.assert_array_equal(config.folded_counts, [2, 3])
    np.testing.assert_allclose(config.transfer_matrix.data.todense(), [[100, 0], [0, 200]])


@pytest.mark.parametrize("explicit", [False, True])
def test_named_missing_response_remains_a_file_error(tmp_path, explicit):
    """A supplied filename that does not exist differs from an unspecified response."""
    pha = write_pha(tmp_path / "source.pha", None if explicit else "missing.rmf")
    with pytest.raises(FileNotFoundError, match=r"missing\.rmf"):
        ObsConfiguration.from_pha_file(pha, rmf_path=tmp_path / "missing.rmf" if explicit else None)
