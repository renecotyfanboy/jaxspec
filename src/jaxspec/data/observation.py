import os
import numpy as np
from . import data_loader
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
    observed_bkg: np.ndarray | None

    def __init__(
        self,
        pha: DataPHA,
        arf: DataARF,
        rmf: DataRMF,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
        ignore_bad_channel: bool = True,
        bkg: DataPHA = None,
        background_subtracted: bool = False,
    ):
        r"""
        This is the basic constructor for an observation.
        It is recommended to build the [`Observation`][jaxspec.data.observation.Observation] object using the
        [`from_pha_file`][jaxspec.data.observation.Observation.from_pha_file] constructor.

        Parameters:
            pha: The PHA data.
            arf: The ARF data.
            rmf: The RMF data.
            low_energy: The lower energy bound.
            high_energy: The higher energy bound.
            ignore_bad_channel: Whether to ignore bad channels ([quality $\neq$ 0](https://heasarc.gsfc.nasa.gov/docs/asca/abc/node9.html#SECTION00923000000000000000)) or not. Defaults to True
            bkg: The background data.
            background_subtracted: Whether the provided PHA is already background subtracted or not.

        !!! warning

            We found that the `HDUCLAS2` fits keyword, which signal whether the spectrum is background-subtracted or not,
            might be misused within the various X-ray data software. So at this time, the user must provide
            this information by himself. See [this issue](https://github.com/renecotyfanboy/jaxspec/issues/99) for more
            details.
        """

        self.pha = pha
        self.bkg = bkg
        self.bkg_subtracted = background_subtracted

        if ignore_bad_channel:
            self.quality_filter = self.pha.quality == 0

        else:
            self.quality_filter = np.ones_like(self.pha.quality)

        super().__init__(
            arf,
            rmf,
            pha.exposure,
            pha.grouping,
            low_energy=low_energy,
            high_energy=high_energy,
        )

    @classmethod
    def from_pha_file(cls, pha_file: str | os.PathLike, **kwargs):
        """
        Build an Instrument object from a PHA file.
        PHA file must contain the ARF and RMF filenames in the header.
        PHA, ARF and RMF files are expected to be in the same directory.

        Parameters:
            pha_file: PHA file path

        """

        pha, arf, rmf, bkg = data_loader(pha_file)

        return cls(pha, arf, rmf, bkg=bkg, **kwargs)

    def rebin(self, grouping):
        # This clearly needs a refactor, even I get lost in the code i've written

        super().rebin(grouping)

        # We also need to rebin the observed counts when there is an observation attached to the instrumental setup

        self.observed_counts = (grouping @ np.asarray(self.pha.counts.value, dtype=np.int64))[self._row_idx]
        self.observed_bkg = (
            (grouping @ np.asarray(self.bkg.counts.value, dtype=np.int64))[self._row_idx] if self.bkg is not None else None
        )

        if self.bkg_subtracted:
            self.observed_counts += self.observed_bkg
