import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from scipy.interpolate import BSpline
from tinygp import GaussianProcess, kernels

from ..model.instrument import GainModel, ShiftModel


def bspline_basis(n_basis: int, degree: int = 3, interval=(0.0, 1.0)):
    """
    Construct an open-uniform B-spline basis on a given interval.

    Parameters
    ----------
    n_basis : int
        Number of basis functions (X in the prompt). Must satisfy
        n_basis >= degree + 1 for an open-uniform knot vector.
    degree : int, optional
        Polynomial degree of the splines (default 3 → cubic).
    interval : tuple(float, float), optional
        The (start, end) of the domain (default (0, 1)).

    Returns:
    -------
    basis : list[BSpline]
        List of `n_basis` BSpline objects forming a basis over the interval.
    knots : ndarray
        The full knot vector, including endpoint multiplicities.
    """
    a, b = interval
    p = degree
    if n_basis < p + 1:
        raise ValueError(f"Need at least {p+1} basis functions (got {n_basis}).")

    # How many *internal* knots (not counting the duplicated endpoints)?
    n_internal = n_basis - p - 1  # open-uniform formula

    # Equally spaced internal knots (could be user-supplied instead)
    internal_knots = (
        np.linspace(a, b, n_internal + 2)[1:-1]  # drop the two ends
        if n_internal > 0
        else np.empty(0)
    )

    # Open-uniform knot vector: endpoints repeated p+1 times
    knots = np.concatenate((np.full(p + 1, a), internal_knots, np.full(p + 1, b)))

    # Coefficient matrix: each column of I generates one basis spline
    coeffs = np.eye(n_basis)

    # Build BSpline objects
    basis = [BSpline(knots, coeffs[i], p, extrapolate=False) for i in range(n_basis)]
    return basis, knots


class GaussianProcessGain(GainModel):
    def __init__(self, e_min, e_max, n_nodes=30):
        # self.prior_distribution = prior_distribution
        self.e_min = e_min
        self.e_max = e_max
        self.n_nodes = n_nodes
        self.kernel = kernels.Matern52

    def numpyro_model(self, observation_name: str):
        mean = numpyro.sample(f"ins/~/_{observation_name}_mean", dist.Normal(1.0, 0.3))

        sigma = numpyro.sample(f"ins/~/_{observation_name}_sigma", dist.HalfNormal(3.0))
        rho = numpyro.sample(f"ins/~/_{observation_name}_rho", dist.HalfNormal(10.0))

        # Set up the kernel and GP objects
        kernel = sigma**2 * self.kernel(rho)
        nodes = jnp.linspace(0, 1, self.n_nodes)
        gp = GaussianProcess(kernel, nodes, diag=1e-5 * jnp.ones_like(nodes), mean=mean)

        gain_sample = numpyro.sample(f"ins/~/{observation_name}_gain", gp.numpyro_dist())

        def gain(energy):
            return jnp.interp(
                energy.mean(axis=0),
                nodes * (self.e_max - self.e_min) + self.e_min,
                gain_sample,
                left=1.0,
                right=1.0,
            )

        return gain


class BsplineGain(GainModel):
    def __init__(self, e_min, e_max, n_nodes=6, grid_size=30):
        self.e_min = e_min
        self.e_max = e_max
        self.n_nodes = n_nodes
        self.egrid = jnp.linspace(e_min, e_max, grid_size)

        basis, knots = bspline_basis(n_nodes, 3, (e_min, e_max))

        self.gridded_basis = jnp.asarray([bi(self.egrid) for bi in basis])

    def numpyro_model(self, observation_name: str):
        coeff = numpyro.sample(
            f"ins/~/_{observation_name}_coeff",
            dist.Uniform(0 * jnp.ones(self.n_nodes), 2 * jnp.ones(self.n_nodes)),
        )

        def gain(energy):
            gridded_gain = jnp.dot(coeff, self.gridded_basis)

            return jnp.interp(energy.mean(axis=0), self.egrid, gridded_gain, left=1.0, right=1.0)

        return gain


class PolynomialGain(GainModel):
    def __init__(self, prior_distribution):
        self.prior_distribution = prior_distribution
        distribution_shape = prior_distribution.shape()
        self.degree = distribution_shape[0] if len(distribution_shape) > 0 else 0

    def numpyro_model(self, observation_name: str):
        polynomial_coefficient = numpyro.sample(
            f"ins/~/gain_{observation_name}", self.prior_distribution
        )

        if self.degree == 0:

            def gain(energy):
                return polynomial_coefficient

        else:

            def gain(energy):
                return jnp.polyval(polynomial_coefficient, energy.mean(axis=0))

        return gain


class PolynomialShift(ShiftModel):
    def __init__(self, prior_distribution):
        self.prior_distribution = prior_distribution
        distribution_shape = prior_distribution.shape()
        self.degree = distribution_shape[0] if len(distribution_shape) > 0 else 0

    def numpyro_model(self, observation_name: str):
        polynomial_coefficient = numpyro.sample(
            f"ins/~/shift_{observation_name}", self.prior_distribution
        )

        if self.degree == 0:
            # ensure that new_energy = energy + constant
            polynomial_coefficient = jnp.asarray([1.0, polynomial_coefficient])

        def shift(energy):
            return jnp.polyval(polynomial_coefficient, energy)

        return shift
