import haiku as hk
import jax.numpy as jnp
from .abc import MultiplicativeComponent
from haiku.initializers import Constant
from astropy.table import Table


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum

    """

    def __call__(self, energy):

        amplitude = hk.get_parameter('A', [], init=Constant(1))
        factor = hk.get_parameter('f', [], init=Constant(1))
        pivot = hk.get_parameter('E_c', [], init=Constant(1))

        return jnp.where(energy >= pivot, 1. + amplitude*jnp.exp(-factor*energy), 1.)


class TbAbs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model

    Parameters
    ----------
        :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """
    def __init__(self):

        super(TbAbs, self).__init__()
        # Fixing incoming path issues
        path = 'tables/xsect_tbabs_wilm.fits'
        table = Table.read(path)
        self.energy = jnp.asarray(table['ENERGY'])
        self.sigma = jnp.asarray(table['SIGMA'])

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=Constant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)
