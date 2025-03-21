import inspect

from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import numpyro
import pytest

from jax.scipy.special import gamma, hyp1f1
from jax.typing import ArrayLike
from jaxspec.util.integrate import integrate_interval, integrate_positive

numpyro.enable_x64()
chex.set_n_cpu_devices(n=4)


@dataclass
class IntervalTestSetup:
    """Class to test integrating over an interval"""

    func: callable  # Function to integrate
    interval: tuple[float, float]  # Interval to integrate over
    result: ArrayLike  # Expected result


@dataclass
class PositiveTestSetup:
    """Class to test integrating over the positive real line"""

    func: callable  # Function to integrate
    result: ArrayLike  # Expected result


intervals_to_test = [
    IntervalTestSetup(lambda x: 1 / (1 + x), (0, 1), jnp.log(2)),
    IntervalTestSetup(lambda x: 1 / (1 + x**2), (0, 1), jnp.pi / 4),
    IntervalTestSetup(lambda x: x ** (-x), (0, 1), 1.291285997),
    IntervalTestSetup(lambda x: jnp.sin(x), (0, 2 * jnp.pi), 0),
]


positives_to_test = [
    PositiveTestSetup(lambda x: 1 / (1 + x) ** 2, 1),
    PositiveTestSetup(lambda x: 1 / (1 + x**2), jnp.pi / 2),
    PositiveTestSetup(lambda x: jnp.exp(-(x**2)), jnp.sqrt(jnp.pi) / 2),
    PositiveTestSetup(lambda x: jnp.sqrt(x) * jnp.exp(-x), jnp.sqrt(jnp.pi) / 2),
]


@pytest.mark.parametrize("setup", intervals_to_test)
def test_integrate_interval(setup: IntervalTestSetup):
    """
    Test integrating over an interval
    """

    assert jnp.isclose(
        integrate_interval(setup.func)(*setup.interval), setup.result
    ), inspect.getsource(setup.func)


@pytest.mark.parametrize("setup", positives_to_test)
def test_integrate_positive(setup: PositiveTestSetup):
    """
    Test integrating over the positive real line
    """

    assert jnp.isclose(integrate_positive(setup.func)(), setup.result), inspect.getsource(
        setup.func
    )


def test_integrate_interval_gradient():
    """
    Test the custom gradient using the integral definition of Kummer's hypergeometric function
    See https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric1F1/
    """

    def hyp1f1_integral(a, b, z):
        def integrand(x, a, b, z):
            return jnp.exp(z * x) * x ** (a - 1.0) * (1.0 - x) ** (-a + b - 1.0)

        return (
            integrate_interval(integrand)(0.0, 1.0, a, b, z) * gamma(b) / (gamma(a) * gamma(b - a))
        )

    a = jnp.asarray(1.5)
    b = jnp.asarray(10.0)
    z = jnp.asarray(0.5)

    assert jnp.isclose(jax.grad(hyp1f1_integral)(a, b, z), jax.grad(hyp1f1)(a, b, z))


def test_integrate_positive_gradient():
    """
    Test the custom gradient using the integral definition of the gamma function
    """

    gamma_integral = integrate_positive(lambda t, z: t ** (z - 1) * jnp.exp(-t))
    z = jnp.asarray(2.5)

    assert jnp.isclose(jax.grad(gamma_integral)(z), jax.grad(gamma)(z))
