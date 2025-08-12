from itertools import product

import jax.numpy as jnp

from jax.scipy.interpolate import RegularGridInterpolator


class RegularGridInterpolatorWithGrad(RegularGridInterpolator):
    """
    A subclass of SciPy's RegularGridInterpolator that also returns the gradient
    of each interpolated output with respect to input coordinates.

    Supports:
        - Linear interpolation
        - Out-of-bounds handling (fill_value=0)
        - Multi-dimensional output (e.g., RGB or vector fields)
        - Batched or single-point evaluation

    Returns:
        - values: shape (..., output_dim)
        - gradients: shape (..., input_dim, output_dim)
    """

    def _ndim_coords_from_arrays(self, points):
        """Convert a tuple of coordinate arrays to a (..., ndim)-shaped array."""
        ndim = len(self.grid)

        if isinstance(points, tuple) and len(points) == 1:
            # handle argument tuple
            points = points[0]
        if isinstance(points, tuple):
            p = jnp.broadcast_arrays(*points)
            for p_other in p[1:]:
                if p_other.shape != p[0].shape:
                    raise ValueError("coordinate arrays do not have the same shape")
            points = jnp.empty((*p[0].shape, len(points)), dtype=float)
            for j, item in enumerate(p):
                points = points.at[..., j].set(item)
        else:
            points = jnp.asarray(points)  # SciPy: asanyarray(points)
            if points.ndim == 1:
                if ndim is None:
                    points = points.reshape(-1, 1)
                else:
                    points = points.reshape(-1, ndim)
        return points

    def __init__(self, points, values, **kwargs):
        kwargs.setdefault("method", "linear")
        kwargs.setdefault("bounds_error", False)
        kwargs.setdefault("fill_value", 0.0)

        super().__init__(points, values, **kwargs)

    def value_and_grad(self, xi):
        ndim = len(self.grid)
        xi = self._ndim_coords_from_arrays(xi)
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)

        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        result = jnp.asarray(0.0)
        for edge_indices in edges:
            weight = jnp.asarray(1.0)
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= jnp.where(ei == 1, 1 - yi, yi)
            result += self.values[edge_indices] * weight[vslice]

        if not self.bounds_error and self.fill_value is not None:
            bc_shp = result.shape[:1] + (1,) * (result.ndim - 1)
            result = jnp.where(out_of_bounds.reshape(bc_shp), self.fill_value, result)

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])
