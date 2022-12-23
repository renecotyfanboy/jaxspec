import haiku as hk
import jax.numpy as jnp


def phi(t):
    return jnp.tanh(jnp.pi / 2 * jnp.sinh(t))


def dphi(t):
    return jnp.pi / 2 * jnp.cosh(t) * (1/jnp.cosh(jnp.pi / 2 * jnp.sinh(t))**2)


class Integrate(hk.Module):
    r"""
    Module for integrating a model between two energies. It relies on double exponential quadrature [1]_ [2]_ [3]_ for
    finite intervals to compute an approximation of the integral of a model.

    References
    ----------
    .. [1] `Takahasi and Mori (1974) <https://ems.press/journals/prims/articles/2686>`_
    .. [2] `Mori and Sugihara (2001) <https://doi.org/10.1016/S0377-0427(00)00501-X>`_
    .. [3] `Tanh-sinh quadrature <https://en.wikipedia.org/wiki/Tanh-sinh_quadrature>`_ from Wikipedia


    """

    def __init__(self, model: hk.Module, n_points=71):
        super(Integrate, self).__init__()
        self.model = model
        self.n = n_points
        self.t = jnp.linspace(-4, 4, self.n)
        self.x = phi(self.t)
        self.dx = dphi(self.t)


    def __call__(self, E_min, E_max):

        #Change of variables to turn the integral from E_min to E_max into an integral from -1 to 1
        x = (E_max-E_min)/2 * self.x + (E_max+E_min)/2
        dx = (E_max-E_min)/2 * self.dx

        return jnp.trapz(self.model(x) * dx, x=self.t)
