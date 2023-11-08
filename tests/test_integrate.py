import inspect
import jax.numpy as jnp
import numpyro
from unittest import TestCase
from jaxspec.util.integrate import integrate_interval, integrate_positive

numpyro.enable_x64()


class TestIntegrate(TestCase):
    func_interval_to_test = [
        (lambda x: 1 / (1 + x), (0, 1), jnp.log(2)),
        (lambda x: 1 / (1 + x**2), (0, 1), jnp.pi / 4),
        (lambda x: x ** (-x), (0, 1), 1.291285997),
        (lambda x: jnp.sin(x), (0, 2 * jnp.pi), 0),
    ]

    func_positive_to_test = [
        (lambda x: 1 / (1 + x) ** 2, 1),
        (lambda x: 1 / (1 + x**2), jnp.pi / 2),
        (lambda x: jnp.exp(-(x**2)), jnp.sqrt(jnp.pi) / 2),
        (lambda x: jnp.sqrt(x) * jnp.exp(-x), jnp.sqrt(jnp.pi) / 2),
    ]

    def test_integrate_interval(self):
        """
        Test integrating over an interval
        """

        for func, interval, result in self.func_interval_to_test:
            assert jnp.isclose(integrate_interval(func, *interval), result)

    def test_integrate_positive(self):
        """
        Test integrating over the positive real line
        """

        for func, result in self.func_positive_to_test:
            assert jnp.isclose(integrate_positive(func), result), inspect.getsource(func)
