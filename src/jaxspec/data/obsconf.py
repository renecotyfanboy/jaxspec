import numpy as np
import scipy
import sparse
import xarray as xr

from .instrument import Instrument
from .observation import Observation


class ObsConfiguration(xr.Dataset):
    """
    Class to store the data of a folding model, which is the link between the unfolded and folded spectra.
    """

    transfer_matrix: xr.DataArray
    """The transfer matrix"""
    area: xr.DataArray
    """The effective area of the instrument"""
    exposure: xr.DataArray
    """The total exposure"""
    folded_counts: xr.DataArray
    """The observed counts, after grouping"""
    folded_background: xr.DataArray
    """The background counts, after grouping"""

    __slots__ = (
        "transfer_matrix",
        "area",
        "exposure",
        "folded_counts",
        "folded_background",
    )

    @property
    def in_energies(self):
        """
        The energy bounds of the unfolded bins in keV. The shape is (2, n_bins).
        """

        in_energies = np.stack(
            (
                np.asarray(self.coords["e_min_unfolded"], dtype=np.float64),
                np.asarray(self.coords["e_max_unfolded"], dtype=np.float64),
            )
        )

        return in_energies

    @property
    def out_energies(self):
        """
        The energy bounds of the folded bins in keV. The shape is (2, n_bins).
        """

        out_energies = np.stack(
            (
                np.asarray(self.coords["e_min_folded"].data, dtype=np.float64),
                np.asarray(self.coords["e_max_folded"].data, dtype=np.float64),
            )
        )

        return out_energies

    @classmethod
    def from_pha_file(
        cls,
        pha_path,
        rmf_path: str | None = None,
        arf_path: str | None = None,
        bkg_path: str | None = None,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
    ):
        r"""
        Build the observation configuration from a PHA file.

        Parameters:
            pha_path: The path to the PHA file.
            rmf_path: The path to the RMF file.
            arf_path: The path to the ARF file.
            bkg_path: The path to the background file.
            low_energy: The lower bound of the energy range to consider.
            high_energy: The upper bound of the energy range to consider.
        """

        from .util import data_path_finder

        arf_path_default, rmf_path_default, bkg_path_default = data_path_finder(pha_path)

        arf_path = arf_path_default if arf_path is None else arf_path
        rmf_path = rmf_path_default if rmf_path is None else rmf_path
        bkg_path = bkg_path_default if bkg_path is None else bkg_path

        instrument = Instrument.from_ogip_file(rmf_path, arf_path=arf_path)
        observation = Observation.from_pha_file(pha_path, bkg_path=bkg_path)

        return cls.from_instrument(
            instrument, observation, low_energy=low_energy, high_energy=high_energy
        )

    @classmethod
    def from_instrument(
        cls,
        instrument: Instrument,
        observation: Observation,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
    ):
        r"""
        Build the observation configuration from an [`Instrument`][jaxspec.data.Instrument] and an [`Observation`][jaxspec.data.Observation] object.

        Parameters:
            instrument: The instrument object.
            observation: The observation object.
            low_energy: The lower bound of the energy range to consider.
            high_energy: The upper bound of the energy range to consider.

        """
        # First we unpack all the xarray data to classical np array for efficiency
        # We also exclude the bins that are flagged with bad quality on the instrument
        quality_filter = observation.quality.data == 0
        grouping = (
            scipy.sparse.csr_array(observation.grouping.data.to_scipy_sparse()) * quality_filter
        )
        e_min_channel = instrument.coords["e_min_channel"].data
        e_max_channel = instrument.coords["e_max_channel"].data
        e_min_unfolded = instrument.coords["e_min_unfolded"].data
        e_max_unfolded = instrument.coords["e_max_unfolded"].data
        redistribution = scipy.sparse.csr_array(instrument.redistribution.data.to_scipy_sparse())
        area = instrument.area.data
        exposure = observation.exposure.data

        # Computing the lower and upper energies of the bins after grouping
        # This is just a trick to compute it without 10 lines of code
        grouping_nan = observation.grouping.data * quality_filter
        grouping_nan.fill_value = np.nan
        e_min = sparse.nanmin(grouping_nan * e_min_channel, axis=1).todense()
        e_max = sparse.nanmax(grouping_nan * e_max_channel, axis=1).todense()

        # Compute the transfer matrix
        transfer_matrix = grouping @ (redistribution * area * exposure)

        # Exclude bins out of the considered energy range, and bins without contribution from the RMF

        row_idx = (e_min > low_energy) & (e_max < high_energy) & (grouping.sum(axis=1) > 0)
        col_idx = (e_min_unfolded > 0) & (redistribution.sum(axis=0) > 0)

        # Apply this reduction to all the relevant arrays
        transfer_matrix = sparse.COO.from_scipy_sparse(transfer_matrix[row_idx][:, col_idx])
        folded_counts = observation.folded_counts.data[row_idx]
        folded_backratio = observation.folded_backratio.data[row_idx]
        area = instrument.area.data[col_idx]
        e_min_folded = e_min[row_idx]
        e_max_folded = e_max[row_idx]
        e_min_unfolded = e_min_unfolded[col_idx]
        e_max_unfolded = e_max_unfolded[col_idx]

        if observation.folded_background is not None:
            folded_background = observation.folded_background.data[row_idx]
            folded_background_unscaled = observation.folded_background_unscaled.data[row_idx]
        else:
            folded_background = np.zeros_like(folded_counts)
            folded_background_unscaled = np.zeros_like(folded_counts)

        data_dict = {
            "transfer_matrix": (
                ["folded_channel", "unfolded_channel"],
                transfer_matrix,
                {
                    "description": "Transfer matrix to use to fold the incoming spectrum. It is built and restricted using the grouping, redistribution matrix, effective area, quality flags and energy bands defined by the user."
                },
            ),
            "area": (
                ["unfolded_channel"],
                area,
                {
                    "description": "Effective area with the same restrictions as the transfer matrix.",
                    "units": "cm^2",
                },
            ),
            "exposure": ([], exposure, {"description": "Total exposure", "unit": "s"}),
            "folded_counts": (
                ["folded_channel"],
                folded_counts,
                {
                    "description": "Folded counts after grouping, with the same restrictions as the transfer matrix.",
                    "unit": "photons",
                },
            ),
            "folded_backratio": (
                ["folded_channel"],
                folded_backratio,
                {
                    "description": "Background scaling after grouping, with the same restrictions as the transfer matrix."
                },
            ),
            "folded_background": (
                ["folded_channel"],
                folded_background,
                {
                    "description": "Folded background counts after grouping, with the same restrictions as the transfer matrix.",
                    "unit": "photons",
                },
            ),
            "folded_background_unscaled": (
                ["folded_channel"],
                folded_background_unscaled,
                {
                    "description": "To be done",
                    "unit": "photons",
                },
            ),
        }

        return cls(
            data_dict,
            coords={
                "e_min_folded": (
                    ["folded_channel"],
                    e_min_folded,
                    {"description": "Low energy of folded channel"},
                ),
                "e_max_folded": (
                    ["folded_channel"],
                    e_max_folded,
                    {"description": "High energy of folded channel"},
                ),
                "e_min_unfolded": (
                    ["unfolded_channel"],
                    e_min_unfolded,
                    {"description": "Low energy of unfolded channel"},
                ),
                "e_max_unfolded": (
                    ["unfolded_channel"],
                    e_max_unfolded,
                    {"description": "High energy of unfolded channel"},
                ),
            },
            attrs=observation.attrs | instrument.attrs,
        )

    @classmethod
    def mock_from_instrument(
        cls,
        instrument: Instrument,
        exposure: float,
        low_energy: float = 1e-300,
        high_energy: float = 1e300,
    ):
        """
        Create a mock observation configuration from an instrument object. The fake observation will have zero counts.

        Parameters:
            instrument: The instrument object.
            exposure: The total exposure of the mock observation.
            low_energy: The lower bound of the energy range to consider.
            high_energy: The upper bound of the energy range to consider.
        """

        n_channels = len(instrument.coords["instrument_channel"])

        observation = Observation.from_matrix(
            np.zeros(n_channels),
            sparse.eye(n_channels),
            np.arange(n_channels),
            np.zeros(n_channels, dtype=bool),
            exposure,
            backratio=np.ones(n_channels),
            attributes={"description": "Mock observation"} | instrument.attrs,
        )

        return cls.from_instrument(
            instrument, observation, low_energy=low_energy, high_energy=high_energy
        )

    def plot_counts(self, **kwargs):
        return self.folded_counts.plot.step(x="e_min_folded", where="post", **kwargs)
