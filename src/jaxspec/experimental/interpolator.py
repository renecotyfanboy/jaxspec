import numpy as np

from scipy.interpolate import RegularGridInterpolator


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

    def __init__(self, points, values, **kwargs):
        kwargs.setdefault("method", "linear")
        kwargs.setdefault("bounds_error", False)
        kwargs.setdefault("fill_value", 0.0)
        self.points = [np.asarray(p) for p in points]
        self.input_dim = len(self.points)

        self.output_shape = values.shape[self.input_dim :]
        values_reshaped = values.reshape(*[len(p) for p in self.points], -1)  # flatten output

        super().__init__(self.points, values_reshaped, **kwargs)

    def __call__(self, xi, return_gradient=True):
        xi = np.atleast_2d(xi).astype(float)
        n_points, n_dims = xi.shape
        assert n_dims == self.input_dim, "Dim mismatch"

        # Interpolate values
        flat_vals = super().__call__(xi)  # shape: (n_points, output_dim)
        values = flat_vals.reshape(n_points, *self.output_shape)

        if not return_gradient:
            return values[0] if values.shape[0] == 1 else values

        gradients = np.zeros((n_points, self.input_dim, np.prod(self.output_shape)), dtype=float)

        for d, grid in enumerate(self.points):
            xq = xi[:, d]
            idx_upper = np.searchsorted(grid, xq, side="right")
            idx_lower = idx_upper - 1

            idx_lower = np.clip(idx_lower, 0, len(grid) - 2)
            idx_upper = np.clip(idx_upper, 1, len(grid) - 1)

            xi_low = xi.copy()
            xi_high = xi.copy()
            xi_low[:, d] = grid[idx_lower]
            xi_high[:, d] = grid[idx_upper]

            f_low = super().__call__(xi_low)
            f_high = super().__call__(xi_high)
            delta = (grid[idx_upper] - grid[idx_lower])[:, np.newaxis]

            grad = np.where(delta != 0, (f_high - f_low) / delta, 0.0)
            gradients[:, d, :] = grad

        # Reshape output properly
        gradients = gradients.reshape(n_points, self.input_dim, *self.output_shape)

        if values.shape[0] == 1:
            return values[0], gradients[0]
        else:
            return values, gradients
