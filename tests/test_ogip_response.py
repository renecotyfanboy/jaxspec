"""Detector labels and response calibration must refer to the same channels."""

from types import SimpleNamespace

import numpy as np
import pytest
import sparse

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.data.ogip import DataARF, DataPHA, DataRMF


def labelled_instrument():
    """An offset detector numbering scheme makes positional misalignment visible."""
    edges = np.arange(1.0, 6.0)
    return Instrument.from_matrix(
        sparse.eye(4),
        [1, 2, 3, 4],
        edges[:-1],
        edges[1:],
        edges[:-1],
        edges[1:],
        channel=[4, 5, 6, 7],
    )


def labelled_observation(channels):
    """A PHA may include only a subset of the calibrated detector channels."""
    pha = DataPHA(channels, np.arange(len(channels)) + 10, 2.0)
    return Observation.from_ogip_container(pha)


def test_pha_subset_uses_matching_response_rows():
    configured = ObsConfiguration.from_instrument(
        labelled_instrument(), labelled_observation([5, 7])
    )
    np.testing.assert_array_equal(configured.channel, [5, 7])
    np.testing.assert_array_equal(configured.folded_counts, [10, 11])
    np.testing.assert_array_equal(configured.out_energies, [[2, 4], [3, 5]])
    # Only photon bins with nonzero support on the selected rows are retained.
    np.testing.assert_array_equal(configured.transfer_matrix.data.todense(), [[4, 0], [0, 8]])
    np.testing.assert_array_equal(configured.in_energies, [[2, 4], [3, 5]])


@pytest.mark.parametrize("channels", [[3, 4], [4, 8], [0, 1, 2, 3]])
def test_unknown_pha_channel_cannot_silently_use_a_different_response_row(channels):
    with pytest.raises(ValueError, match=r"channel identifiers.*absent"):
        ObsConfiguration.from_instrument(labelled_instrument(), labelled_observation(channels))


@pytest.mark.parametrize("channels", [[4, 4, 6, 7], [4, 6, 5, 7], [1, 2], [4, 5.5, 6, 7]])
def test_response_labels_must_be_unambiguous(channels):
    edges = np.arange(1.0, 6.0)
    with pytest.raises(ValueError, match="channel identifiers"):
        Instrument.from_matrix(
            sparse.eye(4),
            np.ones(4),
            edges[:-1],
            edges[1:],
            edges[:-1],
            edges[1:],
            channel=channels,
        )


def calibration_files(monkeypatch, *, arf_low=(1.0, 2.0), arf_high=(2.0, 3.0)):
    """Supply small calibration objects to isolate the physical grid check."""
    response = SimpleNamespace(
        sparse_matrix=sparse.eye(2),
        channel=np.array([1, 2]),
        energ_lo=np.array([1.0, 2.0]),
        energ_hi=np.array([2.0, 3.0]),
        e_min=np.array([1.0, 2.0]),
        e_max=np.array([2.0, 3.0]),
    )
    area = SimpleNamespace(
        energ_lo=np.array(arf_low), energ_hi=np.array(arf_high), specresp=np.array([10.0, 20.0])
    )
    monkeypatch.setattr(DataRMF, "from_file", lambda path: response)
    monkeypatch.setattr(DataARF, "from_file", lambda path: area)


def test_same_length_calibration_grids_cannot_be_silently_mismatched(monkeypatch):
    calibration_files(monkeypatch, arf_low=[1.0, 2.1], arf_high=[2.1, 3.0])
    with pytest.raises(ValueError, match="ARF and RMF photon-energy grids differ"):
        Instrument.from_ogip_file("response.rmf", "wrong.arf")


def test_calibration_storage_roundoff_is_accepted(monkeypatch):
    calibration_files(monkeypatch, arf_low=[1.0, 2.0 + 1e-7])
    instrument = Instrument.from_ogip_file("response.rmf", "matched.arf")
    np.testing.assert_array_equal(instrument.area, [10.0, 20.0])
    np.testing.assert_array_equal(instrument.channel, [1, 2])


def test_integral_channel_quantities_preserve_real_detector_labels():
    """Astropy represents FITS channels with a 'chan' unit as floating quantities."""
    import astropy.units as u

    edges = np.arange(1.0, 4.0)
    instrument = Instrument.from_matrix(
        sparse.eye(2),
        np.ones(2),
        edges[:-1],
        edges[1:],
        edges[:-1],
        edges[1:],
        channel=np.array([0, 1]) * u.def_unit("chan"),
    )
    np.testing.assert_array_equal(instrument.channel, [0, 1])
    assert instrument.channel.dtype.kind in "iu"


def test_combined_response_stays_sparse_and_handles_zero_area(monkeypatch):
    """Factoring an RSP needs only its nonzeros, even with an insensitive energy bin."""

    class SparseResponse:
        sparse_matrix = sparse.COO(np.array([[2.0, 0.0], [3.0, 0.0]]))
        channel = np.array([1, 2])
        energ_lo = e_min = np.array([1.0, 2.0])
        energ_hi = e_max = np.array([2.0, 3.0])

        @property
        def matrix(self):
            raise AssertionError("Loading a combined response must not allocate a dense matrix.")

    monkeypatch.setattr(DataRMF, "from_file", lambda path: SparseResponse())
    instrument = Instrument.from_ogip_file("combined.rsp")
    np.testing.assert_array_equal(instrument.area, [5.0, 0.0])
    np.testing.assert_allclose(instrument.redistribution.data.todense(), [[0.4, 0.0], [0.6, 0.0]])
    np.testing.assert_allclose(
        instrument.redistribution.data.todense() * instrument.area.data,
        [[2.0, 0.0], [3.0, 0.0]],
    )
