import numpy as np
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
                np.asarray(self.coords["e_min_folded"], dtype=np.float64),
                np.asarray(self.coords["e_max_folded"], dtype=np.float64),
            )
        )

        return out_energies

    @classmethod
    def from_pha_file(cls, pha_file, low_energy: float = 1e-20, high_energy: float = 1e20):
        from .util import data_loader

        pha, arf, rmf, bkg, metadata = data_loader(pha_file)

        instrument = Instrument.from_matrix(rmf.matrix, arf.specresp, rmf.energ_lo, rmf.energ_hi, rmf.e_min, rmf.e_max)

        observation = Observation.from_matrix(
            pha.counts,
            pha.grouping,
            pha.channel,
            pha.quality,
            pha.exposure,
            background=bkg.counts if bkg is not None else None,
            backratio=pha.backscal / bkg.backscal if bkg is not None else 1.0,
            attributes=metadata,
        )

        return cls.from_instrument(instrument, observation, low_energy=low_energy, high_energy=high_energy)

    @classmethod
    def from_instrument(
        cls, instrument: Instrument, observation: Observation, low_energy: float = 1e-20, high_energy: float = 1e20
    ):
        grouping = observation.grouping.copy()

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

        # Exclude the bins flagged with bad quality
        quality_filter = observation.quality == 0
        grouping[:, ~quality_filter] = 0

        row_idx = xr.ones_like(e_min, dtype=bool)
        row_idx *= (e_min > low_energy) & (e_max < high_energy)  # Strict exclusion as in XSPEC
        row_idx *= grouping.sum(dim="instrument_channel") > 0  # Exclude channels with no contribution

        col_idx = xr.ones_like(instrument.area, dtype=bool)
        col_idx *= instrument.coords["e_min_unfolded"] > 0.0  # Exclude channels with 0. as lower energy bound
        col_idx *= instrument.redistribution.sum(dim="instrument_channel") > 0  # Exclude channels with no contribution

        transfer_matrix = transfer_matrix.where(row_idx & col_idx, drop=True)
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
                "backratio": observation.backratio,
                "folded_counts": folded_counts,
                "folded_background": folded_background,
            }
        )
