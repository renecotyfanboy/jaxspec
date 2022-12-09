import haiku as hk
import jax.numpy as jnp
import importlib.resources
from .abc import MultiplicativeComponent
from haiku.initializers import Constant
from astropy.table import Table


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum b

    .. math::
        D_{it} =

    Parameters
    ----------
        :math:`A` : amplitude of the modification

        :math:`f` : exponential factor

        :math:`E_c` : start energy of modification

    """
#\begin{cases}1 & \text{if bank $i$ issues ABs at time $t$}\\0 & \text{otherwise}\end{cases}
    def __call__(self, energy):

        amplitude = hk.get_parameter('A', [], init=Constant(1))
        factor = hk.get_parameter('f', [], init=Constant(1))
        pivot = hk.get_parameter('E_c', [], init=Constant(1))

        return jnp.where(energy >= pivot, 1. + amplitude*jnp.exp(-factor*energy), 1.)


class Tbabs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model

    Parameters
    ----------
        :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """
    def __init__(self):

        super(Tbabs, self).__init__()
        ref = importlib.resources.files('jaxspec') / 'tables/xsect_tbabs_wilm.fits'
        with importlib.resources.as_file(ref) as path:
            table = Table.read(path)
        self.energy = jnp.asarray(table['ENERGY'])
        self.sigma = jnp.asarray(table['SIGMA'])

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=Constant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)


class Phabs(MultiplicativeComponent):
    r"""
    A photoelectric absorption

    Parameters
    ----------
        :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """
    def __init__(self):

        super(Phabs, self).__init__()
        ref = importlib.resources.files('jaxspec') / 'tables/xsect_phabs_aspl.fits'
        with importlib.resources.as_file(ref) as path:
            table = Table.read(path)
        self.energy = jnp.asarray(table['ENERGY'])
        self.sigma = jnp.asarray(table['SIGMA'])

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=Constant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)
