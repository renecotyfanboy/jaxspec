import os

import astropy.units as u
import numpy as np
import sparse

from astropy.io import fits
from astropy.table import QTable


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
        flags=None,
    ):
        self.channel = np.asarray(channel, dtype=int)
        self.counts = np.asarray(counts, dtype=int)
        self.exposure = float(exposure)

        self.quality = np.asarray(quality, dtype=int)
        self.backfile = backfile
        self.respfile = respfile
        self.ancrfile = ancrfile
        self.backscal = np.asarray(backscal, dtype=float)
        self.areascal = np.asarray(areascal, dtype=float)
        self.flags = flags

        if grouping is not None:
            # Indices array of the beginning of each group
            b_grp = np.where(grouping == 1)[0]
            # Indices array of the ending of each group
            e_grp = np.hstack((b_grp[1:], len(channel)))

            # Prepare data for sparse matrix
            rows = []
            cols = []
            data = []

            for i in range(len(b_grp)):
                for j in range(b_grp[i], e_grp[i]):
                    rows.append(i)
                    cols.append(j)
                    data.append(True)

            # Create a COO sparse matrix
            grp_matrix = sparse.COO(
                (data, (rows, cols)), shape=(len(b_grp), len(channel)), fill_value=0
            )

        else:
            # Identity matrix case, use sparse for efficiency
            grp_matrix = sparse.eye(len(channel), format="coo", dtype=bool)

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
        flags = []

        if header.get("HDUCLAS3") == "RATE":
            raise ValueError(
                f"The HDUCLAS3={header.get('HDUCLAS3')} keyword in the PHA file is not supported."
                f"Please open an issue if this is required."
            )

        if header.get("HDUCLAS4") == "TYPE:II":
            raise ValueError(
                f"The HDUCLAS4={header.get('HDUCLAS4')} keyword in the PHA file is not supported."
                f"Please open an issue if this is required."
            )

        if header.get("GROUPING") == 0:
            grouping = None
        elif "GROUPING" in data.colnames:
            grouping = data["GROUPING"]
        else:
            raise ValueError("No grouping column found in the PHA file.")

        if header.get("QUALITY") == 0:
            quality = np.zeros(len(data["CHANNEL"]), dtype=bool)
        elif "QUALITY" in data.colnames:
            quality = data["QUALITY"]
        else:
            raise ValueError("No QUALITY column found in the PHA file.")

        if "BACKSCAL" in header:
            backscal = header["BACKSCAL"] * np.ones_like(data["CHANNEL"])
        elif "BACKSCAL" in data.colnames:
            backscal = data["BACKSCAL"]
        else:
            raise ValueError("No BACKSCAL found in the PHA file.")

        backscal = np.where(backscal == 0, 1.0, backscal)

        if "AREASCAL" in header:
            areascal = header["AREASCAL"]
        elif "AREASCAL" in data.colnames:
            areascal = data["AREASCAL"]
        else:
            raise ValueError("No AREASCAL found in the PHA file.")

        if header.get("HDUCLAS2") == "NET":
            flags.append("NET")

        kwargs = {
            "grouping": grouping,
            "quality": quality,
            "backfile": header.get("BACKFILE"),
            "respfile": header.get("RESPFILE"),
            "ancrfile": header.get("ANCRFILE"),
            "backscal": backscal,
            "areascal": areascal,
            "flags": flags,
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

    def __init__(
        self,
        energ_lo,
        energ_hi,
        n_grp,
        f_chan,
        n_chan,
        matrix,
        channel,
        e_min,
        e_max,
        low_threshold=0.0,
    ):
        # RMF stuff
        self.energ_lo = energ_lo  # "Entry" energies
        self.energ_hi = energ_hi  # "Entry" energies
        self.n_grp = n_grp
        self.f_chan = f_chan
        self.n_chan = n_chan
        self.matrix_entry = matrix

        # Detector channels
        self.channel = channel
        self.e_min = e_min
        self.e_max = e_max

        # Prepare data for sparse matrix
        rows = []
        cols = []
        data = []

        for i, n_grp_val in enumerate(self.n_grp):
            base = 0

            if np.size(self.f_chan[i]) == 1:
                low = int(self.f_chan[i].ravel()[0])
                high = min(
                    int(self.f_chan[i].ravel()[0] + self.n_chan[i].ravel()[0]),
                    len(self.channel),
                )

                rows.extend([i] * (high - low))
                cols.extend(range(low, high))
                data.extend(self.matrix_entry[i][0 : high - low])

            else:
                for j in range(n_grp_val):
                    low = self.f_chan[i][j]
                    high = min(self.f_chan[i][j] + self.n_chan[i][j], len(self.channel))

                    rows.extend([i] * (high - low))
                    cols.extend(range(low, high))
                    data.extend(self.matrix_entry[i][base : base + self.n_chan[i][j]])

                    base += self.n_chan[i][j]

        # Convert lists to numpy arrays
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)

        # Sometimes, zero elements are given in the matrix rows, so we get rid of them
        idxs = data > low_threshold

        # Create a COO sparse matrix and then convert to CSR for efficiency
        coo = sparse.COO(
            [rows[idxs], cols[idxs]], data[idxs], shape=(len(self.energ_lo), len(self.channel))
        )
        self.sparse_matrix = coo.T  # .tocsr()

    @property
    def matrix(self):
        return np.asarray(self.sparse_matrix.todense())

    @classmethod
    def from_file(cls, rmf_file: str | os.PathLike):
        """
        Load the data from an RMF file.

        Parameters:
            rmf_file: The RMF file path.
        """
        extension_names = [hdu[1] for hdu in fits.info(rmf_file, output=False)]

        if "MATRIX" in extension_names:
            matrix_extension = "MATRIX"

        elif "SPECRESP MATRIX" in extension_names:
            matrix_extension = "SPECRESP MATRIX"
            # raise NotImplementedError("SPECRESP MATRIX extension is not yet supported")

        else:
            raise ValueError("No MATRIX or SPECRESP MATRIX extension found in the RMF file")

        matrix_table = QTable.read(rmf_file, matrix_extension)
        ebounds_table = QTable.read(rmf_file, "EBOUNDS")

        matrix_header = fits.getheader(rmf_file, matrix_extension)

        f_chan_column_pos = list(matrix_table.columns).index("F_CHAN") + 1
        tlmin_fchan = int(matrix_header[f"TLMIN{f_chan_column_pos}"])

        return cls(
            matrix_table["ENERG_LO"],
            matrix_table["ENERG_HI"],
            matrix_table["N_GRP"],
            matrix_table["F_CHAN"] - tlmin_fchan,
            matrix_table["N_CHAN"],
            matrix_table["MATRIX"],
            ebounds_table["CHANNEL"],
            ebounds_table["E_MIN"],
            ebounds_table["E_MAX"],
        )
