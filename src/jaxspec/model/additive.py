import haiku as hk
import jax.numpy as jnp
from .abc import AdditiveComponent
from haiku.initializers import Constant


class Powerlaw(AdditiveComponent):
    r"""
    A power law model

    .. math::
        \mathcal{M}\left( E \right) = K E^{-\alpha}

    Parameters
    ----------
        :math:`\alpha` : Photon index of the power law :math:`\left[\text{dimensionless}\right]`

        :math:`K` : Normalization :math:`\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]`
    """

    def __call__(self, energy):

        alpha = hk.get_parameter('alpha', [], init=Constant(11/3))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*energy**(-alpha)


class Lorentz(AdditiveComponent):
    r"""
    A Lorentzian line profile

    .. math::
        \mathcal{M}\left( E \right) = K\frac{\frac{\sigma}{2\pi}}{(E-E_L)^2 + \left(\frac{\sigma}{2}\right)^2}

    Parameters
    ----------
        :math:`E_L` : Energy of the line :math:`\left[\text{keV}\right]`

        :math:`\sigma` : FWHM of the line :math:`\left[\text{keV}\right]`

        :math:`K` : Normalization :math:`\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]`
    """

    def __call__(self, energy):

        line_energy = hk.get_parameter('E_l', [], init=Constant(1))
        sigma = hk.get_parameter('sigma', [], init=Constant(1))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*(sigma/(2*jnp.pi))/((energy-line_energy)**2 + (sigma/2)**2)

class LogParabola(AdditiveComponent):
    r"""
    A LogParabola model

    .. math::
        \mathcal{M}\left( E \right) = K \left( \frac{E}{E_{Pivot}} \right)^{-(\alpha + \beta \mathrm{ln}(E/E_{Pivot})) }

    Parameters
    ----------
        :math:`a` : Slope of the LogParabola at the pivot energy :math:`\left[\text{dimensionless}\right]`

        :math:`b` : Curve parameter of the LogParabola :math:`\left[\text{dimensionless}\right]`

        :math:`K` : Normalization at the pivot energy :math:`\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]`

        :math:`E_{Pivot }` : Pivot energy fixed at 1 keV :math:`\left[ \mathrm{keV}\right]`
    """

    def __call__(self, energy):

        a = hk.get_parameter('a', [], init=Constant(11/3))
        b = hk.get_parameter('b', [], init=Constant(0.2))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*energy**(-(a + b*jnp.log(energy)))