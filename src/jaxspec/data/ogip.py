import os

import astropy.units as u
import numpy as np
import sparse

from astropy.io import fits
from astropy.table import QTable

from ._validation import (
    channel_vector,
    detector_channels,
    energy_bins,
    exposure_seconds,
    nonnegative_values,
    poisson_counts,
    quantity_values,
    require_unmasked,
)


def _from_header_or_column(header, data, key):
    """Read scalar header metadata or its per-channel column representation."""
    if key in header:
        return header[key]
    if key in data.colnames:
        return data[key]
    else:
        raise ValueError(f"No {key} found in the PHA file.")


def _reject_unsupported_hduclas(header, key, bad_value):
    """Reject a PHA classification unsupported by the event-count loader."""
    if header.get(key) == bad_value:
        raise ValueError(
            f"The {key}={bad_value} keyword in the PHA file is not supported."
            f"Please open an issue if this is required."
        )


def _rmf_integer_vector(values, *, name):
    """Decode active compressed-row integers without truncating or wrapping labels.

    A scalar FITS column and a one-element vector describe the same group. This
    helper accepts both, while leaving unused fixed-width padding to the caller.
    """
    array = np.asarray(require_unmasked(values, name=name)).reshape(-1)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite integers.")
    if array.dtype.kind == "f":
        valid = np.all(array == np.floor(array)) and np.all(
            (array >= -(2**63)) & (array < float(2**63))
        )
    elif array.dtype.kind == "u":
        valid = np.all(array <= np.iinfo(np.int64).max)
    else:
        valid = True
    if not valid:
        raise ValueError(f"{name} must contain exact signed 64-bit integers.")
    return array.astype(np.int64, copy=False)


def _rmf_matrix_unit(matrix):
    """Return the stored MATRIX unit scale and whether it includes collecting area.

    Redistribution probabilities are dimensionless. A complete RSP can instead
    store an area-valued matrix; convert that area to cm² before factorization.
    Unclassified legacy columns without units retain their numeric convention.
    """
    unit = getattr(matrix, "unit", None)
    if unit is None:
        return 1.0, False
    unit = u.Unit(unit)
    if unit.is_equivalent(u.cm**2):
        return unit.to(u.cm**2), True
    if unit.is_equivalent(u.dimensionless_unscaled):
        return unit.to(u.dimensionless_unscaled), False
    raise ValueError("RMF MATRIX units must be dimensionless or an effective area such as cm2.")


def _cxc_dimensionless_rmf(header, stored_unit):
    """Identify the documented Chandra ACIS redistribution ``au`` convention.

    CXC's source-catalog product documentation labels the dimensionless ACIS
    REDIST matrix ``au``. Interpret that exact context only; an ``au`` unit on
    another instrument or a complete-area response is not silently discarded.
    """
    expected = {
        "TELESCOP": "CHANDRA",
        "INSTRUME": "ACIS",
        "HDUCLASS": "OGIP",
        "HDUCLAS1": "RESPONSE",
        "HDUCLAS2": "RSP_MATRIX",
        "HDUCLAS3": "REDIST",
    }
    return stored_unit == "au" and all(
        str(header.get(key, "")).strip().upper() == value for key, value in expected.items()
    )


def _active_rmf_row(f_chan, n_chan, matrix, *, row, group_count, n_detector, scale):
    """Decode declared groups and weights before mapping a photon row to detectors.

    Only N_GRP groups and their summed N_CHAN weights are calibration data.
    Ignore all other fixed-width padding, preserve masks until active values are
    selected, and convert area units before response thresholding or assembly.
    """
    raw_starts = np.asanyarray(f_chan[row]).reshape(-1)
    raw_widths = np.asanyarray(n_chan[row]).reshape(-1)
    if min(raw_starts.size, raw_widths.size) < group_count:
        raise ValueError(f"RMF row {row} group arrays are shorter than N_GRP.")
    starts = _rmf_integer_vector(raw_starts[:group_count], name=f"RMF row {row} F_CHAN")
    widths = _rmf_integer_vector(raw_widths[:group_count], name=f"RMF row {row} N_CHAN")
    if np.any(widths <= 0) or np.any(widths > n_detector):
        raise ValueError(f"RMF row {row} N_CHAN must be positive and fit within EBOUNDS.")
    total = sum(int(width) for width in widths)
    row_values = np.asanyarray(matrix[row]).reshape(-1)
    if row_values.size < total:
        raise ValueError(f"RMF row {row} has fewer MATRIX values than its active N_CHAN sum.")
    row_values = nonnegative_values(row_values[:total], name=f"RMF row {row} MATRIX values")
    if scale != 1:
        row_values = row_values.astype(np.float64) * scale
        nonnegative_values(row_values, name=f"RMF row {row} converted MATRIX values")
    return starts, widths, row_values


def _compressed_rmf_matrix(n_grp, f_chan, n_chan, matrix, channels, *, threshold, scale):
    """Expand active OGIP groups into a sparse detector-by-photon response.

    N_GRP and N_CHAN identify active elements; fixed-width padding is ignored.
    A group spans contiguous *channel numbers*, so an undeclared label in that
    interval is an invalid calibration and cannot be skipped or clipped. The
    result stays sparse, including rows with N_GRP=0 and a completely empty RMF.
    Weights are converted to canonical units before keeping values strictly
    above ``threshold``. No column normalization is applied.
    """
    n_energy, n_detector = len(n_grp), len(channels)
    if any(len(column) != n_energy for column in (f_chan, n_chan, matrix)):
        raise ValueError("RMF compressed columns must have one row per photon-energy bin.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("RMF low_threshold must be finite and nonnegative.")
    detector_indices, energy_indices, weights = [], [], []
    for row, group_count in enumerate(n_grp):
        if group_count < 0:
            raise ValueError(f"RMF row {row} N_GRP must be nonnegative.")
        if group_count == 0:
            continue
        starts, widths, row_values = _active_rmf_row(
            f_chan,
            n_chan,
            matrix,
            row=row,
            group_count=group_count,
            n_detector=n_detector,
            scale=scale,
        )
        used = np.zeros(n_detector, dtype=bool)
        offset = 0
        for start, width in zip(starts, widths):
            start, width = int(start), int(width)
            end = start + width - 1
            if start < int(channels[0]) or end > int(channels[-1]):
                raise ValueError(f"RMF row {row} contains channel labels absent from EBOUNDS.")
            labels = np.arange(width, dtype=np.int64) + start
            positions = np.searchsorted(channels, labels)
            if not np.array_equal(channels[positions], labels):
                raise ValueError(f"RMF row {row} contains channel labels absent from EBOUNDS.")
            if np.any(used[positions]):
                raise ValueError(f"RMF row {row} has overlapping channel groups.")
            used[positions] = True
            values = row_values[offset : offset + width]
            keep = values > threshold
            detector_indices.append(positions[keep])
            energy_indices.append(np.full(np.count_nonzero(keep), row, dtype=np.int64))
            weights.append(values[keep])
            offset += width
    if not weights:
        return sparse.COO(
            np.empty((2, 0), dtype=np.int64), np.empty(0), shape=(n_detector, n_energy)
        )
    return sparse.COO(
        np.stack((np.concatenate(detector_indices), np.concatenate(energy_indices))),
        np.concatenate(weights),
        shape=(n_detector, n_energy),
    )


class DataPHA:
    r"""
    Store a Type-I OGIP spectrum and its detector-channel metadata.

    Use this container when reading a PHA file or constructing the equivalent
    input for an Observation. ``counts`` contains raw nonnegative integer events,
    and ``channel`` contains increasing, unique detector labels, not array offsets.
    ``exposure`` is in seconds; explicit time Quantities are converted.

    ``grouping`` uses the OGIP codes: 1 starts a group, -1 continues the previous
    group, and 0 marks an ungrouped channel. Omitting it leaves channels separate.
    The stored ``grouping`` is a sparse boolean matrix with grouped channels as
    rows and raw detector channels as columns. ``quality`` defaults to zero
    (usable); flags are retained here and applied when constructing a fit's
    observation configuration.

    ``backscal`` and ``areascal`` accept scalars or one value per channel.
    BACKSCAL describes the extraction-region scale; AREASCAL scales the source
    response. Source/background exposure and scaling ratios are combined by
    ``Observation.from_ogip_container`` when both spectra are available.
    Associated filenames and classification ``flags`` are retained as metadata.

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
        self.counts = poisson_counts(counts, name="PHA counts")
        self.channel = detector_channels(
            channel, size=len(self.counts), name="PHA channel identifiers"
        )
        self.exposure = exposure_seconds(exposure, name="PHA exposure")

        self.quality = channel_vector(
            0 if quality is None else quality, len(self.counts), name="QUALITY", dtype=int
        )
        self.backfile = backfile
        self.respfile = respfile
        self.ancrfile = ancrfile
        self.backscal = channel_vector(backscal, len(self.counts), name="BACKSCAL")
        self.areascal = channel_vector(areascal, len(self.counts), name="AREASCAL")
        self.flags = [] if flags is None else list(flags)

        if grouping is not None:
            grouping = np.asarray(require_unmasked(grouping, name="GROUPING"))
            if grouping.shape != self.counts.shape or not np.isin(grouping, [-1, 0, 1]).all():
                raise ValueError("GROUPING must contain one of -1, 0 or 1 per channel.")
            if grouping[0] == -1:
                raise ValueError("GROUPING begins with -1 before any group starts.")
            # Zero denotes an ungrouped channel; it must not disappear or get
            # appended to the preceding bin when mixed with grouped channels.
            rows = np.cumsum(grouping != -1) - 1
            grp_matrix = sparse.COO(
                np.stack((rows, np.arange(len(channel)))),
                np.ones(len(channel), dtype=bool),
                shape=(int(rows[-1]) + 1, len(channel)),
                fill_value=0,
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

        _reject_unsupported_hduclas(header, "HDUCLAS3", "RATE")
        _reject_unsupported_hduclas(header, "HDUCLAS4", "TYPE:II")

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

        backscal = _from_header_or_column(header, data, "BACKSCAL")
        areascal = _from_header_or_column(header, data, "AREASCAL")

        if header.get("HDUCLAS2") == "NET":
            flags.append("NET")
        # OGIP permits STAT_ERR in place of POISSERR. Do not silently discard
        # supplied statistical errors and assume a Poisson experiment instead.
        if header.get("POISSERR") is False or (
            "STAT_ERR" in data.colnames and header.get("POISSERR") is not True
        ):
            flags.append("NONPOISSON")

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

        if "COUNTS" in data.colnames:
            counts = data["COUNTS"]
        elif "RATE" in data.colnames:
            rate = quantity_values(data["RATE"], u.ct / u.s, name="PHA RATE")
            counts = rate * exposure_seconds(header["EXPOSURE"], name="PHA exposure")
        else:
            raise ValueError("No COUNTS or RATE column found in the PHA file.")

        return cls(data["CHANNEL"], counts, header["EXPOSURE"], **kwargs)


class DataARF:
    r"""
    Store an OGIP effective-area curve on the incident photon-energy grid.

    Pair this container with a redistribution-only RMF to construct an Instrument.
    ``energ_lo`` and ``energ_hi`` contain one lower and upper edge per photon bin
    in keV; ``specresp`` contains the corresponding effective areas in cm².
    Explicit units are converted to these defaults. Bins must be ordered and
    nonoverlapping apart from storage roundoff; zero effective area is valid.

    ??? info "References"
        * [The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats)](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)
        * [The Calibration Requirements for Spectral Analysis Addendum: Changes log](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html)
    """

    def __init__(self, energ_lo, energ_hi, specresp):
        self.energ_lo, self.energ_hi = energy_bins(
            quantity_values(energ_lo, u.keV, name="ARF ENERG_LO"),
            quantity_values(energ_hi, u.keV, name="ARF ENERG_HI"),
            name="ARF energies",
            roundoff_dtype=np.result_type(np.asarray(energ_lo), np.asarray(energ_hi)),
        )
        self.specresp = nonnegative_values(
            quantity_values(specresp, u.cm**2, name="ARF SPECRESP"),
            name="ARF SPECRESP",
            size=len(self.energ_lo),
        )

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
            arf_table["SPECRESP"],
        )


class DataRMF:
    r"""
    Decode an OGIP redistribution matrix or complete response for detector folding.

    Input rows describe incident photon-energy bins between ``energ_lo`` and
    ``energ_hi``. For each row, ``n_grp`` declares the active groups, ``f_chan``
    gives their first detector labels, and ``n_chan`` gives their widths.
    ``matrix`` holds the concatenated weights for those groups. Unused padding
    is ignored. Groups map to the actual ``channel`` labels, which must be
    increasing and unique; labels need not start at zero or be gap-free.

    ``e_min`` and ``e_max`` give nominal detector bounds in channel-label order;
    that order need not follow increasing energy. All energy arrays default to
    keV, and explicit units are converted. Redistribution weights are
    dimensionless; area-valued complete responses are converted to cm².
    ``includes_effective_area`` also identifies a unitless complete response
    whose weights already use cm², preventing a second ARF from being applied.

    The decoded ``sparse_matrix`` has shape ``(n_detector_channels, n_photon_bins)``.
    Weights at or below ``low_threshold`` are omitted after unit conversion;
    remaining columns are not renormalized. The default threshold retains all
    positive weights. Use ``sparse_matrix`` for large calibration responses.

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
