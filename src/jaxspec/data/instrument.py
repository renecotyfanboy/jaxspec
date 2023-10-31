import os
import numpy as np
from typing import Union
from .ogip import DataARF, DataRMF


class Instrument:
    """
    Class to store the data of an observational setup, including the ARF and RMF files. An instance of this class
    includes the energy band, exposure time and expected grouping to mock data.
    """

    arf: DataARF
    rmf: DataRMF
    exposure: float
    in_energies: np.ndarray
    transfer_matrix: np.ndarray

    def __init__(
        self,
        arf: DataARF,
        rmf: DataRMF,
        exposure: float,
        grouping: np.ndarray,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
    ):
        """
        This is the basic constructor for an instrumental setup.
        It is recommended to build the Instrument object using the
        [`from_ogip_file`][jaxspec.data.instrument.Instrument.from_ogip_file] constructor.

        Parameters:
            arf: The ARF data.
            rmf: The RMF data.
            exposure: The exposure time in second.
            grouping: The grouping matrix.
            low_energy: The lower energy bound.
            high_energy: The higher energy bound.
        """

        # These are set in the rebin method
        self.in_energies = None
        self.out_energies = None
        self._row_idx = None
        self._col_idx = None

        self.arf = arf
        self.rmf = rmf
        self.exposure = exposure
        self.low_energy = low_energy
        self.high_energy = high_energy
        self.grouping = grouping
        self.rebin(grouping)

    @classmethod
    def from_ogip_file(
        cls,
        arf_file: Union[str, os.PathLike],
        rmf_file: Union[str, os.PathLike],
        exposure: float,
        grouping: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Load the data from OGIP files.

        Parameters:
            arf_file: The ARF file path.
            rmf_file: The RMF file path.
            exposure: The exposure time in second.
            grouping: The grouping matrix.
        """

        arf = DataARF.from_file(arf_file)
        rmf = DataRMF.from_file(rmf_file)

        # Use an identity matrix if no grouping is provided.
        if grouping is None:
            grouping = np.eye(len(rmf.channel))

        return cls(arf, rmf, exposure, grouping, **kwargs)

    def rebin(self, grouping: np.ndarray):
        """
        This method allows to rebin the instrumental setup given a grouping matrix. It will compute the new transfer
        matrix and the new energy grids from the ARF and RMF files.

        Parameters:
            grouping: The grouping matrix.

        Warning:
            If you want to rebin multiple times, you should pass the grouping matrix which correspond to the
            multiplication of each of your rebinning matrices. Applying multiple rebin method will not stack the
            different grouping matrices.
        """

        arf = self.arf
        rmf = self.rmf

        low_energy = self.low_energy
        high_energy = self.high_energy

        # Out energies considering grouping
        e_min = np.nanmin(np.where(grouping > 0, grouping, np.nan) * rmf.e_min.value[None, :], axis=1)
        e_max = np.nanmax(np.where(grouping > 0, grouping, np.nan) * rmf.e_max.value[None, :], axis=1)

        row_idx = np.ones(grouping.shape[0], dtype=bool)
        row_idx *= (e_min >= low_energy) & (e_max <= high_energy)

        col_idx = np.ones(rmf.energ_lo.shape, dtype=bool)
        col_idx *= rmf.energ_lo.value > 0.0  # Exclude channels with 0. as lower energy bound
        col_idx *= rmf.full_matrix.sum(axis=0) > 0  # Exclude channels with no contribution

        # In energy grid for the model, we integrate it using trapezoid evaluated on edges (2 points)
        in_energies = np.stack(
            (
                np.asarray(rmf.energ_lo, dtype=np.float64),
                np.asarray(rmf.energ_hi, dtype=np.float64),
            )
        )

        # Transfer matrix computation considering grouping
        transfer_matrix = grouping @ (rmf.full_matrix * arf.specresp * self.exposure)

        # Store the excluded channels
        self._col_idx = col_idx
        self._row_idx = row_idx

        # Selecting only the channels that are not masked
        self.in_energies = in_energies[:, col_idx]
        self.out_energies = np.stack((e_min[row_idx], e_max[row_idx]))
        self.transfer_matrix = transfer_matrix[row_idx, :][:, col_idx]
