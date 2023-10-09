import os
import numpy as np
from typing import Union
from .ogip import DataPHA, DataARF, DataRMF
from .instrument import Instrument


class Observation(Instrument):
    """
    Class to store the data of an observation, including the PHA, ARF and RMF files.

    Note:
        This inherits from the [`Instrument`][jaxspec.data.instrument.Instrument] class, it includes every attribute from
        an instrumental setup, including the energy band, exposure time and grouping. It can be used in replacement of
        [`Instrument`][jaxspec.data.instrument.Instrument] in any function taking an instrumental setup as an argument.
    """

    pha: DataPHA
    observed_counts: np.ndarray

    def __init__(self,
                 pha: DataPHA,
                 arf: DataARF,
                 rmf: DataRMF,
                 low_energy: float = np.finfo(float).eps,
                 high_energy: float = np.inf):

        self.pha = pha
        super().__init__(
            arf,
            rmf,
            pha.exposure,
            pha.grouping,
            low_energy=low_energy,
            high_energy=high_energy)

    @classmethod
    def from_pha_file(cls, pha_file: Union[str, os.PathLike], **kwargs):
        """
        Build an Instrument object from a PHA file.
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

    def rebin(self, grouping):

        super().rebin(grouping)

        # We also need to rebin the observed counts when there is an observation attached to the instrumental setup
        self.observed_counts = (grouping @ np.asarray(self.pha.counts.value, dtype=np.int64))[self._row_idx]