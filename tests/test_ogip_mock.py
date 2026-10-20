"""Synthetic observations retain the detector identities of their real response."""

import numpy as np
import pytest
import sparse

from jaxspec.data import Instrument, ObsConfiguration


@pytest.mark.parametrize("channels", [None, [1, 2, 3], [10, 12, 13]])
def test_mock_observation_preserves_response_labels_and_physical_folding(channels):
    """Offset and gapped labels must simulate the same response-weighted photon counts."""
    instrument = Instrument.from_matrix(
        sparse.eye(3), [1, 2, 3], [1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], channel=channels
    )
    mock = ObsConfiguration.mock_from_instrument(instrument, exposure=2.0)
    np.testing.assert_array_equal(mock.channel, np.arange(3) if channels is None else channels)
    np.testing.assert_array_equal(mock.folded_counts, [0, 0, 0])
    np.testing.assert_allclose(mock.transfer_matrix.data @ np.ones(3), [2, 4, 6])
