import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax import Array


def integrate_interval(func, a: float, b: float, n: int = 41) -> Array:
    """
    Integrate a function over an interval [a, b] using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        a: The lower bound of the interval
        b: The upper bound of the interval
        n: The number of points to use for the quadrature
    """

    # Change of variables to turn the integral from a to b into an integral from -1 to 1
    t = jnp.linspace(-3, 3, n)

    phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)

    x = (b - a) / 2 * phi + (b + a) / 2
    dx = (b - a) / 2 * dphi

    return trapezoid(jnp.nan_to_num(func(x) * dx), x=t)


def integrate_positive(func, n: int = 41) -> Array:
    """
    Integrate a function over the positive real axis using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        n: The number of points to use for the quadrature
    """
    t = jnp.linspace(-3, 3, n)

    x = jnp.exp(jnp.pi / 2 * jnp.sinh(t))
    dx = jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    return trapezoid(jnp.nan_to_num(func(x) * dx), x=t)
