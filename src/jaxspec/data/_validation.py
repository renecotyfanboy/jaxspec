"""Validate counts and calibration inputs before constructing Poisson observations."""

import astropy.units as u
import numpy as np
import sparse

from scipy.sparse import issparse


def require_unmasked(values, *, name):
    """Reject missing measurements before NumPy conversion can expose their payload.

    Masked FITS columns and NumPy arrays retain arbitrary numeric data under a
    mask. Interpreting that data as counts or calibration weights changes the
    experiment silently. Fully unmasked containers remain valid inputs.
    """
    mask = getattr(values, "mask", False)
    # pandas containers expose a .mask() transformation method, not a mask
    # array. They still undergo the same numeric and finite-value validation.
    if not callable(mask) and np.any(mask):
        raise ValueError(f"{name} contains masked or missing values; supply complete measurements.")
    return values


def quantity_values(values, unit, *, name):
    """Convert explicit physical units and otherwise use an API's stated default.

    OGIP recommends keV, cm² and seconds, but a unit-tagged FITS column may use
    equivalent units such as eV or m². Convert before dropping the Quantity;
    relabeling its raw numeric values would change folded counts. Plain arrays
    and FITS columns without a unit retain the caller's documented default.
    Explicit units must match the requested dimension. Dimensionless units are
    valid for dimensionless targets, such as redistribution weights.
    """
    require_unmasked(values, name=name)
    if getattr(values, "unit", None) is not None:
        try:
            values = u.Quantity(values, dtype=np.float64).to_value(unit)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must have units equivalent to {unit}.") from error
    return np.asarray(values, dtype=np.float64)


def exposure_seconds(values, *, name="Exposure"):
    """Return one finite positive exposure, converting explicit time Quantities.

    Use the same seconds convention for PHA input and synthetic observations so
    a mock exposure in ks cannot be interpreted as a count-rate scale of seconds.
    """
    value = quantity_values(values, "s", name=name)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive scalar.")
    return float(value)


def energy_bins(lower, upper, *, name, roundoff_dtype=None, ordered=True):
    """Validate bounds already expressed in keV without moving their edges.

    Callers convert explicit units with ``quantity_values`` before this check.

    Zero lower edges and gaps are legitimate calibration boundaries. Reversed,
    nonfinite or overlapping bins cannot represent the independent photon-bin
    integrals used by response folding. A tiny overlap from independently rounded
    calibration edges is accepted up to one relative machine epsilon of their
    original floating dtype, capped at 1% of either bin's width. Edges are never
    moved or averaged. This accommodates existing float32 NuSTAR ARF products.

    ``ordered=False`` validates nominal detector bounds per channel without
    rearranging them. Their channel labels need not follow increasing energy;
    RGS, for example, uses decreasing energies in wavelength-channel order.
    """
    lower = np.asarray(require_unmasked(lower, name=name))
    upper = np.asarray(require_unmasked(upper, name=name))
    if (
        lower.ndim != 1
        or not lower.size
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower < 0)
        or np.any(upper <= lower)
    ):
        raise ValueError(
            f"{name} must define finite, nonnegative energy bins with one lower and upper "
            "bound per bin and positive widths."
        )
    if not ordered:
        return lower, upper
    if np.any(lower[1:] <= lower[:-1]) or np.any(upper[1:] <= upper[:-1]):
        raise ValueError(f"{name} must contain increasing photon-energy bins.")
    dtype = np.dtype(roundoff_dtype or np.result_type(lower, upper))
    epsilon = np.finfo(dtype if dtype.kind == "f" else np.float64).eps
    roundoff = epsilon * np.maximum(np.abs(lower[1:]), np.abs(upper[:-1]))
    width_limit = 0.01 * np.minimum((upper - lower)[1:], (upper - lower)[:-1])
    if np.any(upper[:-1] - lower[1:] > np.minimum(roundoff, width_limit)):
        raise ValueError(f"{name} contain overlapping energy bins beyond storage roundoff.")
    return lower, upper


def nonnegative_values(values, *, name, size=None):
    """Validate response weights or areas without imposing unit column sums.

    Zero sensitivity is physical, and an RMF may contain detector efficiency.
    Negative or nonfinite values instead make the predicted Poisson rate invalid.
    """
    array = np.asarray(require_unmasked(values, name=name))
    if (
        array.dtype.kind not in "biuf"
        or (size is not None and array.shape != (size,))
        or not np.isfinite(array).all()
        or np.any(array < 0)
    ):
        suffix = " with one value per photon-energy bin" if size is not None else ""
        raise ValueError(f"{name} must contain finite, nonnegative values{suffix}.")
    return array


def response_matrix(matrix, *, shape):
    """Validate active response weights and store one zero-filled sparse COO matrix.

    Use this at the array construction boundary so dense inputs and supported
    SciPy/PyData sparse formats all support the same detector-folding operations.
    Unused sparse padding is not calibration data; active negative, nonfinite or
    masked weights must still fail before conversion.
    """
    if getattr(matrix, "unit", None) is not None:
        matrix = quantity_values(matrix, "", name="Redistribution matrix")
    if np.shape(matrix) != shape:
        raise ValueError("Redistribution matrix shape must match detector and photon bins.")
    if issparse(matrix):
        # DOK has no numeric .data array, LIL stores lists, and DIA .data
        # includes padding outside the matrix. COO exposes only active
        # entries without allocating a dense detector response.
        matrix = matrix.tocoo(copy=False)
    if isinstance(matrix, sparse.SparseArray):
        nonnegative_values(matrix.fill_value, name="Redistribution matrix")
        weights = matrix.data
    elif issparse(matrix):
        weights = matrix.data
    else:
        weights = matrix
    nonnegative_values(weights, name="Redistribution matrix")
    if isinstance(matrix, sparse.SparseArray):
        if matrix.fill_value != 0:
            raise ValueError(
                "Sparse redistribution matrices require zero fill_value. Supply a dense "
                "array or an explicit zero-filled sparse representation."
            )
        return sparse.COO(matrix)
    if issparse(matrix):
        return sparse.COO.from_scipy_sparse(matrix)
    return sparse.COO(np.asarray(matrix))


def poisson_counts(values, *, name="counts"):
    """Return exact int64 event counts, rejecting rounding and integer overflow.

    Use this at file and array boundaries: truncating fractional or negative
    measurements would silently change the data used by a Poisson likelihood.
    Floating-point arrays are accepted only when every value is an integer.
    """
    array = np.asarray(require_unmasked(values, name=name))
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional array of event counts.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite, nonnegative event counts.")
    if array.dtype.kind == "f":
        if np.any(array != np.floor(array)):
            raise ValueError(
                f"{name} contains fractional values. Supply raw event counts, not rates "
                "or a background-subtracted spectrum, for a Poisson likelihood."
            )
        in_range = np.all(array < float(2**63))
    else:
        in_range = np.all(array <= np.iinfo(np.int64).max)
    if not in_range:
        raise ValueError(f"{name} exceeds the signed 64-bit event-count range.")
    return array.astype(np.int64, copy=False)


def detector_channels(values, *, size=None, name="Detector channel identifiers"):
    """Preserve integer channel labels, including integral Astropy Quantity values.

    FITS CHANNEL columns are labels rather than array positions. A unit-tagged
    integer column can become floating point when Astropy constructs a Quantity;
    accept its exact integer values without rounding fractions or wrapping large
    identifiers during conversion to int64. Use the same check for PHA spectra,
    response EBOUNDS and programmatic observation/instrument construction.
    """
    array = np.asarray(require_unmasked(values, name=name))
    if (
        array.ndim != 1
        or not array.size
        or (size is not None and array.shape != (size,))
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be increasing, unique integers with one label per channel.")
    if array.dtype.kind == "f":
        valid = np.all(array == np.floor(array)) and np.all(
            (array >= -(2**63)) & (array < float(2**63))
        )
    elif array.dtype.kind == "u":
        valid = np.all(array <= np.iinfo(np.int64).max)
    else:
        valid = True
    if not valid:
        raise ValueError(f"{name} must be exact integers within the signed 64-bit range.")
    labels = array.astype(np.int64)
    if np.any(labels[1:] <= labels[:-1]):
        raise ValueError(f"{name} must be increasing, unique integers.")
    return labels


def channel_vector(values, size, *, name, dtype=float):
    """Broadcast scalar OGIP metadata or validate one value per detector channel.

    Both header scalars and Type-I vector columns are valid storage forms for
    area/background scaling. Shape checks prevent accidental channel mixing.
    """
    array = np.asarray(require_unmasked(values, name=name))
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    if np.issubdtype(np.dtype(dtype), np.integer) and np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain integers.")
    array = array.astype(dtype)
    if array.ndim == 0:
        array = np.full(size, array, dtype=dtype)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and scalar or have one value per channel.")
    return array


def poisson_grouping(grouping, size):
    """Normalize a non-overlapping event-sum matrix to sparse COO storage.

    Summing disjoint raw channels preserves independent Poisson counts; weighted
    or overlapping groups do not have the likelihood assumed by a PHA fit.
    """
    require_unmasked(grouping, name="Grouping")
    if not isinstance(grouping, sparse.COO):
        grouping = (
            sparse.COO.from_scipy_sparse(grouping)
            if issparse(grouping)
            else sparse.COO(np.asarray(grouping))
        )
    if (
        grouping.ndim != 2
        or grouping.fill_value != 0
        or grouping.shape[0] == 0
        or grouping.shape[1] != size
        or not np.isin(grouping.data, [0, 1]).all()
        or np.any(grouping.sum(axis=0).todense() > 1)
    ):
        raise ValueError("Grouping must sum disjoint detector channels with weights zero or one.")
    return grouping.astype(bool)
