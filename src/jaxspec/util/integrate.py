r"""
Module for integrating functions in a consistent way in `JAXspec`.
It uses on tanh-sinh (or double exponential) quadrature.

References:

* [Takahasi and Mori (1974)](https://ems.press/journals/prims/articles/2686)
* [Mori and Sugihara (2001)](https://doi.org/10.1016/S0377-0427(00)00501-X)
* [Tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) from Wikipedia
"""

import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax import Array


def integrate_interval(func, a: float, b: float, n: int = 51) -> Array:
    """
    Integrate a function over an interval [a, b] using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        a: The lower bound of the interval
        b: The upper bound of the interval
        n: The number of points to use for the quadrature
    """

    # Change of variables to turn the integral from a to b into an integral from -1 to 1
    t = jnp.linspace(-4.5, 4.5, n)

    phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)

    x = (b - a) / 2 * phi + (b + a) / 2
    dx = (b - a) / 2 * dphi

    return trapezoid(jnp.nan_to_num(func(x) * dx), x=t)


def integrate_positive(func, n: int = 51) -> Array:
    """
    Integrate a function over the positive real axis using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        n: The number of points to use for the quadrature
    """
    t = jnp.linspace(-4.5, 4.5, n)

    x = jnp.exp(jnp.pi / 2 * jnp.sinh(t))
    dx = jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    return trapezoid(jnp.nan_to_num(func(x) * dx), x=t)
