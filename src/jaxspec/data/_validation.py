"""Validate counts and calibration inputs before constructing Poisson observations."""

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
