import numpy as np
import os
import astropy.units as u
from astropy.table import QTable
from astropy.io import fits


class DataPHA:
    r"""
    Class to handle PHA data defined with OGIP standards.

    ??? info "References"
        * [The OGIP standard PHA file format](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html)
    """

    def __init__(
        self,
        channel,
        counts,
        exposure,
        grouping=None,
        quality=None,
        backfile=None,
        respfile=None,
        ancrfile=None,
        backscal=1.0,
        areascal=1.0,
    ):
        self.channel = channel
        self.counts = counts
        self.exposure = exposure

        self.quality = quality
        self.backfile = backfile
        self.respfile = respfile
        self.ancrfile = ancrfile
        self.backscal = backscal
        self.areascal = areascal

        if grouping is not None:
            # Indices array of beginning of each group
            b_grp = np.where(grouping == 1)[0]
            # Indices array of ending of each group
            e_grp = np.hstack((b_grp[1:], len(channel)))
            # Matrix to multiply with counts/channel to have counts/group
            grp_matrix = np.zeros((len(b_grp), len(channel)), dtype=bool)

            for i in range(len(b_grp)):
                grp_matrix[i, b_grp[i] : e_grp[i]] = 1

        else:
            grp_matrix = np.eye(len(channel))

        self.grouping = grp_matrix

    @classmethod
    def from_file(cls, pha_file: str | os.PathLike):
        """
        Load the data from a PHA file.

        Parameters:
            pha_file: The PHA file path.
        """

        data = QTable.read(pha_file, "SPECTRUM")
        header = fits.getheader(pha_file, "SPECTRUM")

        if "QUALITY" in data.colnames:
            quality = data["QUALITY"]
        else:
            if header.get("QUALITY") == 0:
                quality = np.ones(len(data["CHANNEL"]), dtype=bool)
            else:
                raise ValueError("No quality column found in the PHA file.")

        # Grouping and quality parameters are in binned PHA dataset
        # Backfile, respfile and ancrfile are in primary header
        kwargs = {
            "grouping": data["GROUPING"] if "GROUPING" in data.colnames else None,
            "quality": quality,
            "backfile": header.get("BACKFILE"),
            "respfile": header.get("RESPFILE"),
            "ancrfile": header.get("ANCRFILE"),
            "backscal": header.get("BACKSCAL", 1.0),
            "areascal": header.get("AREASCAL", 1.0),
        }

        return cls(data["CHANNEL"], data["COUNTS"], header["EXPOSURE"], **kwargs)


class DataARF:
    r"""
    Class to handle ARF data defined with OGIP standards.

    ??? info "References"
        * [The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats)](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)
        * [The Calibration Requirements for Spectral Analysis Addendum: Changes log](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html)
    """

    def __init__(self, energ_lo, energ_hi, specresp):
        self.specresp = specresp
        self.energ_lo = energ_lo
        self.energ_hi = energ_hi

    @classmethod
    def from_file(cls, arf_file: str | os.PathLike):
        """
        Load the data from an ARF file.

        Parameters:
            arf_file: The ARF file path.
        """

        arf_table = QTable.read(arf_file)

        return cls(
            arf_table["ENERG_LO"],
            arf_table["ENERG_HI"],
            arf_table["SPECRESP"].to(u.cm**2).value,
        )


class DataRMF:
    r"""
    Class to handle RMF data defined with OGIP standards.

    ??? info "References"
        * [The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats)](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)
        * [The Calibration Requirements for Spectral Analysis Addendum: Changes log](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html)

    """

    def __init__(self, energ_lo, energ_hi, n_grp, f_chan, n_chan, matrix, channel, e_min, e_max):
        # RMF stuff
        self.energ_lo = energ_lo  # "Entry" energies
        self.energ_hi = energ_hi  # "Entry" energies
        self.n_grp = n_grp  # "Entry" energies
        self.f_chan = f_chan
        self.n_chan = n_chan
        self.matrix_entry = matrix

        # Detector channels
        self.channel = channel
        self.e_min = e_min
        self.e_max = e_max

        self.full_matrix = np.zeros(self.n_grp.shape + self.channel.shape)

        for i, n_grp in enumerate(self.n_grp):
            base = 0

            if np.size(self.f_chan[i]) == 1:
                # ravel()[0] allows to get the value of the array without triggering numpy's conversion from
                # multidimensional array to scalar
                low = int(self.f_chan[i].ravel()[0])
                high = min(
                    int(self.f_chan[i].ravel()[0] + self.n_chan[i].ravel()[0]),
                    self.full_matrix.shape[1],
                )

                self.full_matrix[i, low:high] = self.matrix_entry[i][0 : high - low]

            else:
                for j in range(n_grp):
                    low = self.f_chan[i][j]
                    high = min(self.f_chan[i][j] + self.n_chan[i][j], self.full_matrix.shape[1])

                    self.full_matrix[i, low:high] = self.matrix_entry[i][base : base + self.n_chan[i][j]]

                    base += self.n_chan[i][j]

        # Transposed matrix so that we just have to multiply by the spectrum
        self.matrix = self.full_matrix.T
        # self.matrix = bsr_matrix(self.full_matrix.T).T
        # self.sparse_matrix = sparse.BCOO.fromdense(jnp.copy(self.full_matrix))

    @classmethod
    def from_file(cls, rmf_file: str | os.PathLike):
        """
        Load the data from an RMF file.

        Parameters:
            rmf_file: The RMF file path.
        """

        matrix_table = QTable.read(rmf_file, "MATRIX")
        ebounds_table = QTable.read(rmf_file, "EBOUNDS")

        return cls(
            matrix_table["ENERG_LO"],
            matrix_table["ENERG_HI"],
            matrix_table["N_GRP"],
            matrix_table["F_CHAN"],
            matrix_table["N_CHAN"],
            matrix_table["MATRIX"],
            ebounds_table["CHANNEL"],
            ebounds_table["E_MIN"],
            ebounds_table["E_MAX"],
        )
