import os
import numpy as np
from typing import Union
from .ogip import DataPHA, DataARF, DataRMF


class Observation:
    """
    Class to store the data of an observation, including the PHA, ARF and RMF files.
    """

    arf: DataARF
    rmf: DataRMF
    pha: DataPHA
    exposure: float
    energies: np.ndarray
    transfer_matrix: np.ndarray
    observed_counts: np.ndarray

    def __init__(self,
                 pha: DataPHA,
                 arf: DataARF,
                 rmf: DataRMF,
                 low_energy: float = np.finfo(float).eps,
                 high_energy: float = np.inf):

        self.arf = arf
        self.rmf = rmf
        self.pha = pha
        self.exposure = pha.exposure
        grouping = pha.grouping

        # Out energies considering grouping
        e_min = np.nanmin(np.where(grouping > 0, grouping, np.nan) * rmf.e_min.value[None, :], axis=1)
        e_max = np.nanmax(np.where(grouping > 0, grouping, np.nan) * rmf.e_max.value[None, :], axis=1)

        row_idx = np.ones(grouping.shape[0], dtype=bool)
        row_idx *= (e_min >= low_energy) & (e_max <= high_energy)

        col_idx = np.ones(rmf.energ_lo.shape, dtype=bool)
        col_idx *= rmf.energ_lo.value > 0.  # Exclude channels with 0. as lower energy bound
        col_idx *= rmf.full_matrix.sum(axis=0) > 0  # Exclude channels with no contribution

        # Energy grid for the model, we integrate it using trapezoid evaluated on edges (2 points)
        energies = np.stack((np.asarray(rmf.energ_lo, dtype=np.float64), np.asarray(rmf.energ_hi, dtype=np.float64)))

        # Transfer matrix computation considering grouping
        transfer_matrix = pha.grouping @ (rmf.full_matrix * arf.specresp * pha.exposure)

        # Selecting only the channels that are not masked
        self.energies = energies[:, col_idx]
        self.transfer_matrix = transfer_matrix[row_idx, :][:, col_idx]
        self.observed_counts = (grouping @ np.asarray(pha.counts.value, dtype=np.int64))[row_idx]

    @classmethod
    def from_pha_file(cls, pha_file: Union[str, os.PathLike], **kwargs):
        """
        Build an Observation object from a PHA file.
        PHA file must contain the ARF and RMF filenames in the header.
        PHA, ARF and RMF files are expected to be in the same directory.

        :param pha_file: PHA file path

        """

        directory = os.path.dirname(pha_file)

        pha = DataPHA.from_file(pha_file)

        if pha.ancrfile is None or pha.respfile is None:
            raise ValueError("PHA file must contain the ARF and RMF filenames in the header.")

        arf = DataARF.from_file(os.path.join(directory, pha.ancrfile))
        rmf = DataRMF.from_file(os.path.join(directory, pha.respfile))

        return cls(pha, arf, rmf, **kwargs)

    def __str__(self):
        return f"obs_{self.pha.id}"
