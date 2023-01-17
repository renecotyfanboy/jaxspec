import haiku as hk
import jax.numpy as jnp
from abc import ABC, abstractmethod


class ModelComponent(hk.Module, ABC):

    @abstractmethod
    def __call__(self, energy):
        """
        Return the model evaluated at a given energy
        """
        pass


class AdditiveComponent(ModelComponent, ABC):

    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on double exponential quadrature for
        finite intervals to compute an approximation of the integral of a model.

        References
        ----------
        * `Takahasi and Mori (1974) <https://ems.press/journals/prims/articles/2686>`_
        * `Mori and Sugihara (2001) <https://doi.org/10.1016/S0377-0427(00)00501-X>`_
        * `Tanh-sinh quadrature <https://en.wikipedia.org/wiki/Tanh-sinh_quadrature>`_ from Wikipedia

        """

        t = jnp.linspace(-4, 4, 71) # The number of points used is hardcoded and this is not ideal
        # Quadrature nodes as defined in reference
        phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
        # Change of variable to turn the integral from E_min to E_max into an integral from -1 to 1
        x = (e_max - e_min) / 2 * phi + (e_max + e_min) / 2
        dx = (e_max - e_min) / 2 * dphi

        return jnp.trapz(self(x) * dx, x=t)


class AnalyticalAdditive(AdditiveComponent, ABC):

    @abstractmethod
    def primitive(self, energy):
        r"""
        Analytical primitive of the model

        """
        pass

    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on the primitive of the model.
        """

        return self.primitive(e_max) - self.primitive(e_min)


class MultiplicativeComponent(ModelComponent, ABC):

    pass
