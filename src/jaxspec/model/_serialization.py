"""Versioned, JSON-compatible model descriptions with explicit numerical state."""

from __future__ import annotations

import math

import numpy as np


def _json_config(value):
    """Copy literal constructor configuration, rejecting opaque or non-finite values."""
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufU":
            raise ValueError("Model configuration must contain finite JSON literals only.")
        # Most NumPy scalars have an exact Python counterpart. Extended-precision
        # scalars can return another NumPy scalar from item(); reject those rather
        # than recurse forever or silently discard precision.
        value = value.item()
        if isinstance(value, np.generic):
            raise ValueError("Model configuration requires scalars representable as JSON literals.")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list | tuple):
        return [_json_config(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _json_config(item) for key, item in value.items()}
    raise ValueError("Model configuration must contain finite JSON literals only.")
