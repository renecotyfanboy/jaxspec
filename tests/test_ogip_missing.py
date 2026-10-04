"""Missing event data and active calibration entries must never become numeric payloads."""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from astropy.io import fits
from astropy.table import MaskedColumn
from astropy.utils.masked import Masked

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataARF, DataPHA, DataRMF


@pytest.fixture(params=[np.ma.array, MaskedColumn])
def missing(request):
    """Represent the same missing channel through array and FITS-table interfaces."""
    return lambda values: request.param(values, mask=[False, True])


@pytest.mark.parametrize(
    "field", ["counts", "channel", "quality", "grouping", "backscal", "areascal"]
)
def test_missing_pha_measurement_cannot_be_read_as_hidden_value(missing, field):
    """Even a plausible positive payload underneath a mask is not measured data."""
    values = dict(
        counts=[2, 999],
        channel=[0, 1],
        quality=[0, 0],
        grouping=[1, -1],
        backscal=[1.0, 1.0],
        areascal=[1.0, 1.0],
        exposure=10.0,
    )
    values[field] = missing(values[field])
    with pytest.raises(ValueError, match="masked or missing"):
        DataPHA(**values)


@pytest.mark.parametrize("field", ["counts", "background"])
def test_missing_observation_counts_are_not_implicitly_imputed(missing, field):
    """The programmatic observation path uses the same raw-count contract as PHA."""
    values = dict(counts=[2, 3], background=[0, 1])
    values[field] = missing(values[field])
    with pytest.raises(ValueError, match="masked or missing"):
        Observation.from_matrix(**values, grouping=np.eye(2), channel=[0, 1], quality=0, exposure=1)


def test_positive_fits_null_counts_are_rejected(tmp_path):
    """TNULL can mark a positive integer that would otherwise pass count validation."""
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=[0, 1]),
            fits.Column(name="COUNTS", format="J", null=999, array=[2, 999]),
        ],
        name="SPECTRUM",
    )
    spectrum.header.update(EXPOSURE=10.0, GROUPING=0, QUALITY=0, BACKSCAL=1.0, AREASCAL=1.0)
    path = tmp_path / "missing-counts.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    with pytest.raises(ValueError, match=r"PHA counts.*masked or missing"):
        DataPHA.from_file(path)


@pytest.mark.parametrize("field", ["energ_lo", "energ_hi", "specresp"])
def test_missing_arf_entries_are_not_used_as_calibration(missing, field):
    """An absent area or photon boundary cannot be replaced by its hidden payload."""
    values = dict(energ_lo=[1.0, 2.0], energ_hi=[2.0, 3.0], specresp=[10.0, 20.0])
    values[field] = missing(values[field])
    with pytest.raises(ValueError, match="masked or missing"):
        DataARF(**values)


def test_unit_conversion_preserves_missing_area_detection():
    """Astropy MaskedQuantity must be checked before plain Quantity conversion."""
    area = Masked([0.01, 0.02] * u.m**2, mask=[False, True])
    with pytest.raises(ValueError, match=r"Effective area.*masked or missing"):
        Instrument.from_matrix(np.eye(2), area, [1, 2], [2, 3], [1, 2], [2, 3])


def test_missing_dense_response_entries_are_rejected():
    """A missing detector probability is neither a known zero nor a known response."""
    response = np.ma.array([[0.7, 0.2], [0.1, 0.6]], mask=[[False, False], [True, False]])
    with pytest.raises(ValueError, match=r"Redistribution matrix.*masked or missing"):
        Instrument.from_matrix(response, [1, 1], [1, 2], [2, 3], [1, 2], [2, 3])


@pytest.mark.parametrize("field", ["n_grp", "f_chan", "n_chan", "matrix"])
def test_missing_active_rmf_entries_are_rejected(field):
    """All compressed-row fields need known values for active response groups."""
    values = dict(n_grp=[1], f_chan=[[0, 0]], n_chan=[[2, 0]], matrix=[[0.7, 0.3, 99.0]])
    mask = np.zeros(np.shape(values[field]), dtype=bool)
    mask.flat[0] = True
    values[field] = np.ma.array(values[field], mask=mask)
    with pytest.raises(ValueError, match="masked or missing"):
        DataRMF([1], [2], **values, channel=[0, 1], e_min=[1, 2], e_max=[2, 3])


def test_null_padding_in_compressed_response_is_ignored():
    """Only N_GRP groups and their N_CHAN probabilities constitute the RMF."""
    rmf = DataRMF(
        [1],
        [2],
        [1],
        np.ma.array([[0, 99]], mask=[[False, True]]),
        np.ma.array([[2, 99]], mask=[[False, True]]),
        np.ma.array([[0.7, 0.3, 99]], mask=[[False, False, True]]),
        [0, 1],
        [1, 2],
        [2, 3],
    )
    np.testing.assert_array_equal(rmf.matrix, [[0.7], [0.3]])


def test_fully_unmasked_containers_preserve_count_folding():
    """Having mask metadata alone does not make a complete observation invalid."""
    counts = MaskedColumn([2, 3], mask=False)
    observation = Observation.from_matrix(counts, np.eye(2), [0, 1], 0, 10)
    instrument = Instrument.from_matrix(
        np.ma.array(np.eye(2), mask=False),
        Masked([1, 2] * u.cm**2, mask=False),
        [1, 2],
        [2, 3],
        [1, 2],
        [2, 3],
    )
    configured = ObsConfiguration.from_instrument(instrument, observation)
    np.testing.assert_array_equal(configured.folded_counts, [2, 3])
    np.testing.assert_array_equal(configured.transfer_matrix.data @ np.array([3, 4]), [30, 80])


def test_pandas_mask_method_is_not_missing_value_metadata():
    """Complete Series and DataFrames remain valid numerical observation inputs."""
    pha = DataPHA(pd.Series([0, 1]), pd.Series([2, 3]), 10, quality=pd.Series([0, 0]))
    instrument = Instrument.from_matrix(
        pd.DataFrame([[0.7, 0.2], [0.1, 0.6]]),
        pd.Series([1, 2]),
        pd.Series([1, 2]),
        pd.Series([2, 3]),
        [1, 2],
        [2, 3],
    )
    observation = ObsConfiguration.from_instrument(instrument, Observation.from_ogip_container(pha))
    np.testing.assert_array_equal(observation.folded_counts, [2, 3])
    np.testing.assert_allclose(observation.transfer_matrix.data @ np.array([3, 4]), [37, 51])


def test_pandas_missing_counts_still_fail_numeric_validation():
    """Ignoring a mask method cannot admit actual missing pandas measurements."""
    with pytest.raises(ValueError, match="finite"):
        DataPHA([0, 1], pd.Series([2, np.nan]), 10)
