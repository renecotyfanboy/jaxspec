r"""
Module for integrating functions in a consistent way in `jaxspec`.
It mainly relies on tanh-sinh (or double exponential) quadrature to perform the integration.

!!! info "References"

    * [Takahasi and Mori (1974)](https://ems.press/journals/prims/articles/2686)
    * [Mori and Sugihara (2001)](https://doi.org/10.1016/S0377-0427(00)00501-X)
    * [Tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) from Wikipedia
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.integrate import trapezoid
from typing import Callable


def interval_weights(a: float, b: float, n: int) -> tuple[Array, Array, Array]:
    """
    Return the weights for the tanh-sinh quadrature over the interval [a, b].
    """
    t = jnp.linspace(-3, 3, n)
    phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
    dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
    x = (b - a) / 2 * phi + (b + a) / 2
    dx = (b - a) / 2 * dphi

    return t, x, dx


def positive_weights(n: int) -> tuple[Array, Array, Array]:
    """
    Return the weights for the tanh-sinh quadrature over the positive real line.
    """
    t = jnp.linspace(-3, 3, n)
    x = jnp.exp(jnp.pi / 2 * jnp.sinh(t))
    dx = jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    return t, x, dx


def integrate_interval(integrand: Callable, n: int = 51) -> Callable:
    r"""
    Build a function which can compute the integral of the provided integrand over the interval $[a, b]$ using
    the tanh-sinh quadrature. Returns a function $F(a, b, \pmb{\theta})$ which takes the limits of the interval and
    the parameters of $f(x,\pmb{\theta})$ as inputs.

    $$
    F(a, b, \pmb{\theta}) = \int_a^b f(x,\pmb{\theta}) \text{d}x
    $$

    # Example usage

    ``` python
    pi = 4*integrate_interval(lambda x: 1/(1+x**2))(0, 1)
    print(pi) # 3.1415927
    ```

    # Example where the limits of the integral are parameters
    ``` python
    def erf(x):

        def integrand(t):
            return 2/jnp.sqrt(jnp.pi) * jnp.exp(-t**2)

        return integrate_interval(integrand)(0, x)

    print(erf(1)) # 0.84270084
    ```

    Parameters:
        integrand: The function to integrate
        n: The number of points to use for the quadrature

    Returns:
        The integral of the provided integrand over the interval $[a, b]$ as a callable
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


def integrate_positive(integrand: Callable, n: int = 51) -> Callable:
    r"""
    Build a function which can compute the integral of the provided integrand over the positive real line using
    the tanh-sinh quadrature. Returns a function $F(\pmb{\theta})$ which takes the parameters of the integrand
    $f(x,\pmb{\theta})$ as inputs.

    $$
    F(\pmb{\theta}) = \int_0^\infty f(x,\pmb{\theta}) \text{d}x
    $$

    # Example usage

    ``` python
    gamma = integrate_positive(lambda t, z: t**(z-1) * jnp.exp(-t))
    print(gamma(1/2)) # 1.7716383
    ```

    Parameters:
        integrand: The function to integrate
        n: The number of points to use for the quadrature

    Returns:
        The integral of the provided integrand over the positive real line as a callable
    """

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
