"""Validate counts and calibration inputs before constructing Poisson observations."""

import astropy.units as u
import numpy as np


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
