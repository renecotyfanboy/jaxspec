import os

import astropy.units as u
import numpy as np
import sparse

from astropy.io import fits
from astropy.table import QTable, Table

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
        if "BACKSCAL" in header:
            backscal = backscal * np.ones_like(data["CHANNEL"], dtype=float)
        areascal = _from_header_or_column(header, data, "AREASCAL")

        if header.get("HDUCLAS2") == "NET":
            flags.append("NET")
        if header.get("POISSERR") is False:
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
            counts = data["RATE"] * header["EXPOSURE"]
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
        *,
        includes_effective_area=False,
    ):
        self.energ_lo, self.energ_hi = energy_bins(
            quantity_values(energ_lo, u.keV, name="RMF ENERG_LO"),
            quantity_values(energ_hi, u.keV, name="RMF ENERG_HI"),
            name="RMF energies",
            roundoff_dtype=np.result_type(np.asarray(energ_lo), np.asarray(energ_hi)),
        )
        if np.shape(n_grp) != self.energ_lo.shape:
            raise ValueError("RMF N_GRP must have one value per photon-energy bin.")
        self.n_grp = _rmf_integer_vector(n_grp, name="RMF N_GRP")
        self.f_chan = f_chan
        self.n_chan = n_chan
        self.matrix_entry = matrix
        self.e_min, self.e_max = energy_bins(
            quantity_values(e_min, u.keV, name="EBOUNDS E_MIN"),
            quantity_values(e_max, u.keV, name="EBOUNDS E_MAX"),
            name="EBOUNDS energies",
            roundoff_dtype=np.result_type(np.asarray(e_min), np.asarray(e_max)),
            ordered=False,
        )
        self.channel = detector_channels(
            channel, size=len(self.e_min), name="RMF EBOUNDS channel identifiers"
        )
        scale, area_unit = _rmf_matrix_unit(matrix)
        self.matrix_unit_original = str(matrix.unit) if getattr(matrix, "unit", None) else None
        self.compatibility_notes = ()
        self.includes_effective_area = bool(includes_effective_area or area_unit)
        self.sparse_matrix = _compressed_rmf_matrix(
            self.n_grp,
            self.f_chan,
            self.n_chan,
            self.matrix_entry,
            self.channel,
            threshold=low_threshold,
            scale=scale,
        )

    @property
    def matrix(self):
        """Materialize a dense detector-by-photon array for inspection or small responses."""
        return np.asarray(self.sparse_matrix.todense())

    @classmethod
    def from_file(cls, rmf_file: str | os.PathLike):
        """
        Load one OGIP response matrix, preserving actual EBOUNDS channel labels.

        Fixed and variable-length compressed rows are supported, including zero
        groups. Files containing multiple MATRIX extensions currently raise an
        error because silently choosing one would discard part of the response.

        Parameters:
            rmf_file: The RMF file path.
        """
        with fits.open(rmf_file) as hdus:
            matrix_extensions = [
                index for index, hdu in enumerate(hdus) if hdu.name in ("MATRIX", "SPECRESP MATRIX")
            ]
            ebounds_extensions = [index for index, hdu in enumerate(hdus) if hdu.name == "EBOUNDS"]
            if not matrix_extensions:
                raise ValueError("No MATRIX or SPECRESP MATRIX extension found in the RMF file.")
            if len(matrix_extensions) > 1:
                raise ValueError(
                    "RMF files with multiple MATRIX extensions are not yet supported. "
                    "Supply a scientifically combined single-matrix response; selecting the "
                    "first extension would discard calibration components."
                )
            if len(ebounds_extensions) != 1:
                raise ValueError("The RMF must contain exactly one EBOUNDS extension.")
            matrix_extension, ebounds_extension = matrix_extensions[0], ebounds_extensions[0]
            matrix_header = hdus[matrix_extension].header.copy()
            stored_unit = hdus[matrix_extension].columns["MATRIX"].unit
            includes_area = str(matrix_header.get("HDUCLAS3", "")).strip().upper() == "FULL"
        # Table retains variable-length MATRIX units without trying to coerce
        # its ragged object column into a homogeneous Quantity. Conversion is
        # explicit in the constructor for both fixed and variable row storage.
        matrix_table = Table.read(rmf_file, matrix_extension)
        ebounds_table = Table.read(rmf_file, ebounds_extension)
        matrix_values = matrix_table["MATRIX"]
        compatibility_notes = ()

        result = cls(
            matrix_table["ENERG_LO"],
            matrix_table["ENERG_HI"],
            matrix_table["N_GRP"],
            matrix_table["F_CHAN"],
            matrix_table["N_CHAN"],
            matrix_values,
            ebounds_table["CHANNEL"],
            ebounds_table["E_MIN"],
            ebounds_table["E_MAX"],
            includes_effective_area=includes_area,
        )
        result.matrix_unit_original = stored_unit
        result.compatibility_notes = compatibility_notes
        return result
