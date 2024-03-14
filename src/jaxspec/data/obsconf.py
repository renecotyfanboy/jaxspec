import numpy as np
import xarray as xr
import sparse
from .instrument import Instrument
from .observation import Observation


def densify_xarray(xarray):
    return xr.DataArray(xarray.data.todense(), dims=xarray.dims, coords=xarray.coords, attrs=xarray.attrs, name=xarray.name)


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
                np.asarray(self.coords["e_min_folded"].data.todense(), dtype=np.float64),
                np.asarray(self.coords["e_max_folded"].data.todense(), dtype=np.float64),
            )
        )

        return out_energies

    @classmethod
    def from_pha_file(
        cls, pha_path, rmf_path=None, arf_path=None, bkg_path=None, low_energy: float = 1e-20, high_energy: float = 1e20
    ):
        from .util import data_loader

        pha, arf, rmf, bkg, metadata = data_loader(pha_path, arf_path=arf_path, rmf_path=rmf_path, bkg_path=bkg_path)

        instrument = Instrument.from_matrix(
            rmf.sparse_matrix,
            arf.specresp if arf is not None else np.ones_like(rmf.energ_lo),
            rmf.energ_lo,
            rmf.energ_hi,
            rmf.e_min,
            rmf.e_max,
        )

        if bkg is not None:
            backratio = np.where(bkg.backscal > 0.0, pha.backscal / np.where(bkg.backscal > 0, bkg.backscal, 1.0), 0.0)
        else:
            backratio = np.ones_like(pha.counts)

        observation = Observation.from_matrix(
            pha.counts,
            pha.grouping,
            pha.channel,
            pha.quality,
            pha.exposure,
            background=bkg.counts if bkg is not None else None,
            backratio=backratio,
            attributes=metadata,
        )

        return cls.from_instrument(instrument, observation, low_energy=low_energy, high_energy=high_energy)

    @classmethod
    def from_instrument(
        cls, instrument: Instrument, observation: Observation, low_energy: float = 1e-20, high_energy: float = 1e20
    ):
        # Exclude the bins flagged with bad quality
        quality_filter = observation.quality == 0
        grouping = observation.grouping * quality_filter

        # Computing the lower and upper energies of the bins after grouping
        # This is just a trick to compute it without 10 lines of code
        e_min = (xr.where(grouping > 0, grouping, np.nan) * instrument.coords["e_min_channel"]).min(
            skipna=True, dim="instrument_channel"
        )

        e_max = (xr.where(grouping > 0, grouping, np.nan) * instrument.coords["e_max_channel"]).max(
            skipna=True, dim="instrument_channel"
        )

        transfer_matrix = grouping @ (instrument.redistribution * instrument.area * observation.exposure)
        transfer_matrix = transfer_matrix.assign_coords({"e_min_folded": e_min, "e_max_folded": e_max})

        # Exclude bins out of the considered energy range, and bins without contribution from the RMF
        row_idx = densify_xarray(((e_min > low_energy) & (e_max < high_energy)) * (grouping.sum(dim="instrument_channel") > 0))

        col_idx = densify_xarray(
            (instrument.coords["e_min_unfolded"] > 0) * (instrument.redistribution.sum(dim="instrument_channel") > 0)
        )

        # The transfer matrix is converted locally to csr format to allow FAST slicing
        transfer_matrix_scipy = transfer_matrix.data.to_scipy_sparse().tocsr()
        transfer_matrix_reduced = transfer_matrix_scipy[row_idx.data][:, col_idx.data]
        transfer_matrix_reduced = sparse.COO.from_scipy_sparse(transfer_matrix_reduced)

        # A dummy zero matrix is put so that the slicing in xarray is fast
        transfer_matrix.data = sparse.zeros_like(transfer_matrix.data)
        transfer_matrix = transfer_matrix[row_idx][:, col_idx]

        # The reduced transfer matrix is put back in the xarray
        transfer_matrix.data = transfer_matrix_reduced

        folded_counts = observation.folded_counts.copy().where(row_idx, drop=True)

        if observation.folded_background is not None:
            folded_background = observation.folded_background.copy().where(row_idx, drop=True)

        else:
            folded_background = None

        return cls(
            {
                "transfer_matrix": transfer_matrix,
                "area": instrument.area.copy().where(col_idx, drop=True),
                "exposure": observation.exposure,
                "folded_backratio": observation.folded_backratio.copy().where(row_idx, drop=True),
                "folded_counts": folded_counts,
                "folded_background": folded_background,
            }
        )
