"""Every accepted in-memory response format must support actual count folding."""

import numpy as np
import pytest
import sparse

from scipy import sparse as scipy_sparse

from jaxspec.data import Instrument, ObsConfiguration


@pytest.mark.parametrize(
    "constructor",
    [np.array, sparse.COO, sparse.GCXS]
    + [
        getattr(scipy_sparse, f"{format}_{kind}")
        for format in ("csr", "csc", "coo", "lil", "dok", "dia", "bsr")
        for kind in ("array", "matrix")
    ],
)
def test_response_array_formats_preserve_non_diagonal_detector_predictions(constructor):
    """Matrix conversion retains channel leakage, collecting area and exposure."""
    matrix = np.array([[0.7, 0.2], [0.1, 0.6]])
    area, photons, exposure = np.array([20.0, 40.0]), np.array([2.0, 3.0]), 10.0
    instrument = Instrument.from_matrix(
        constructor(matrix), area, [1, 2], [2, 3], [1, 2], [2, 3], channel=[10, 11]
    )
    observation = ObsConfiguration.mock_from_instrument(instrument, exposure)
    actual = observation.transfer_matrix.data @ photons
    np.testing.assert_allclose(actual, exposure * (matrix @ (area * photons)), rtol=1e-14)
    np.testing.assert_array_equal(observation.channel, [10, 11])
    assert isinstance(instrument.redistribution.data, sparse.COO)


def test_implicit_nonzero_sparse_fill_is_rejected_before_observation_construction():
    """A sparse constant background cannot be passed silently to zero-filled SciPy storage."""
    matrix = sparse.COO(np.array([[0.7, 0.2], [0.1, 0.6]]), fill_value=0.2)
    with pytest.raises(ValueError, match="zero fill_value"):
        Instrument.from_matrix(matrix, [1, 1], [1, 2], [2, 3], [1, 2], [2, 3])


def test_diagonal_storage_padding_is_not_a_response_weight():
    """Unused DIA padding must not reject an otherwise nonnegative response."""
    matrix = scipy_sparse.dia_array(([[0.7, 0.6], [-99, 0.2]], [0, 1]), shape=(2, 2))
    instrument = Instrument.from_matrix(matrix, [1, 1], [1, 2], [2, 3], [1, 2], [2, 3])
    np.testing.assert_array_equal(instrument.redistribution.data.todense(), [[0.7, 0.2], [0, 0.6]])


@pytest.mark.parametrize("format", ["lil", "dok", "dia"])
def test_sparse_format_conversion_still_rejects_negative_active_weights(format):
    """Converting storage must never hide a physically invalid detector response."""
    matrix = scipy_sparse.csr_array([[0.7, -0.2], [0, 0.6]]).asformat(format)
    with pytest.raises(ValueError, match="Redistribution matrix"):
        Instrument.from_matrix(matrix, [1, 1], [1, 2], [2, 3], [1, 2], [2, 3])
