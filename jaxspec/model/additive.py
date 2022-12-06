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

        line_energy = hk.get_parameter('E_l', [], init=self.E_l_init)
        sigma = hk.get_parameter('sigma', [], init=self.sigma_init)
        norm = hk.get_parameter('norm', [], init=self.norm_init)

        return norm*(sigma/(2*jnp.pi))/((energy-line_energy)**2 + (sigma/2)**2)
