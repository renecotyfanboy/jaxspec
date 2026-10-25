"""Versioned, JSON-compatible model descriptions with explicit numerical state."""

from __future__ import annotations

import math

import jax
import numpy as np

from flax import nnx


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


def _encode_state_leaf(path, leaf):
    """Describe one tracked numerical leaf without executable Python objects."""
    variable = isinstance(leaf, nnx.Variable)
    value = leaf.get_value() if variable else leaf
    if type(value) in (float, int, bool):
        storage = "scalar"
    elif isinstance(value, np.generic):
        storage = "numpy_scalar"
    elif isinstance(value, np.ndarray):
        storage = "numpy"
    elif isinstance(value, jax.Array):
        storage = "jax"
    else:
        raise ValueError(f"Unsupported numerical state at {path!r}: {type(value).__name__}.")
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"State at {path!r} must contain finite real numbers or booleans.")
    if not path or any(type(part) not in (str, int) for part in path):
        raise ValueError(f"Unsupported numerical state path: {path!r}.")
    return {
        "path": list(path),
        "variable": f"{type(leaf).__module__}.{type(leaf).__qualname__}" if variable else None,
        "storage": storage,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "value": _json_config(array.tolist()),
    }


def _decode_state_leaf(record):
    """Validate a numerical-state record before any model constructor runs."""
    keys = {"path", "variable", "storage", "dtype", "shape", "value"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("Invalid numerical state record.")
    path, shape = record["path"], record["shape"]
    if not isinstance(path, list) or not path or any(type(p) not in (str, int) for p in path):
        raise ValueError("Numerical state paths must be non-empty lists of strings or integers.")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Numerical state shapes must be lists of non-negative integers.")
    if record["variable"] is not None and not isinstance(record["variable"], str):
        raise ValueError("Invalid numerical state variable type.")
    if record["storage"] not in ("scalar", "numpy_scalar", "numpy", "jax"):
        raise ValueError("Unknown numerical state storage kind.")
    if not isinstance(record["dtype"], str):
        raise ValueError("Invalid numerical state dtype.")
    try:
        dtype = np.dtype(record["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Numerical state requires a real numeric or boolean dtype.")
        values = np.asarray(record["value"])
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("Numerical state must contain finite real values.")
        array = values.astype(dtype)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid numerical state values or dtype.") from exc
    if 0 in shape and array.size == 0:
        array = array.reshape(shape)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError("Numerical state values do not match their finite shape and dtype.")
    if dtype.kind in "biu" and not np.array_equal(array, values):
        raise ValueError("Numerical state values cannot be represented by their dtype.")
    if record["storage"] in ("scalar", "numpy_scalar") and shape:
        raise ValueError("Scalar numerical state cannot have array dimensions.")
    return tuple(path), array
