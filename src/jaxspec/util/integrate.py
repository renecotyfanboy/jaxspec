r"""
Module for integrating functions in a consistent way in `JAXspec`.
It uses on tanh-sinh (or double exponential) quadrature.

References:

* [Takahasi and Mori (1974)](https://ems.press/journals/prims/articles/2686)
* [Mori and Sugihara (2001)](https://doi.org/10.1016/S0377-0427(00)00501-X)
* [Tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) from Wikipedia
"""

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from typing import Callable


def interval_weights(a, b, n):
    # Change of variables to turn the integral from a to b into an integral from -1 to 1
    t = jnp.linspace(-3, 3, n)
    phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
    x = (b - a) / 2 * phi + (b + a) / 2
    dx = (b - a) / 2 * dphi

    return t, x, dx


def positive_weights(n):
    # Change of variables to turn the integral from a to b into an integral from -1 to 1
    t = jnp.linspace(-3, 3, n)
    x = jnp.exp(jnp.pi / 2 * jnp.sinh(t))
    dx = jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    return t, x, dx


def integrate_interval(integrand, n: int = 51) -> Callable:
    """
    Integrate a function over an interval [a, b] using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        a: The lower bound of the interval
        b: The upper bound of the interval
        n: The number of points to use for the quadrature
    """

    @jax.custom_jvp
    def f(a, b, *args):
        t, x, dx = interval_weights(a, b, n)

        return trapezoid(jnp.nan_to_num(integrand(x, *args) * dx), x=t)

    @f.defjvp
    def f_jvp(primals, tangents):
        a, b, *args = primals
        a_dot, b_dot, *args_dot = tangents

        t, x, dx = interval_weights(a, b, n)

        primal_out = f(a, b, *args)

        # Partial derivatives along other parameters
        jac = trapezoid(jnp.nan_to_num(jnp.asarray(jax.jacfwd(lambda args: integrand(x, *args))(args)) * dx), x=t, axis=-1)

        tangent_out = -integrand(a, *args) * a_dot + integrand(b, *args) * b_dot + jac @ jnp.asarray(args_dot)
        return primal_out, tangent_out

    return f


def integrate_positive(integrand, n: int = 51) -> Callable:
    """
    Integrate a function over the positive real axis using the tanh-sinh quadrature.

    Parameters:
        func: The function to integrate
        n: The number of points to use for the quadrature
    """
    # TODO : add erf to test derivative with regard to the integral limits

    @jax.custom_jvp
    def f(*args):
        t, x, dx = positive_weights(n)

        return trapezoid(jnp.nan_to_num(integrand(x, *args) * dx), x=t)

    @f.defjvp
    def f_jvp(primals, tangents):
        args = primals
        args_dot = tangents

        t, x, dx = positive_weights(n)

        primal_out = f(*args)

        # Partial derivatives along other parameters
        jac = trapezoid(jnp.nan_to_num(jnp.asarray(jax.jacfwd(lambda args: integrand(x, *args))(args)) * dx), x=t, axis=-1)

        tangent_out = jac @ jnp.asarray(args_dot)
        return primal_out, tangent_out

    return f
