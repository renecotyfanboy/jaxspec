import haiku as hk
import jax.numpy as jnp
import numpy as np
import importlib.resources
from .abc import MultiplicativeComponent
from haiku.initializers import Constant as HaikuConstant
from astropy.table import Table


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum.

    .. math::
        \mathcal{M}(E) = \begin{cases}1 + A \exp \left(-fE\right) & \text{if $E>E_c$}\\1 & \text{if $E<E_c$}\end{cases}

    Parameters
    ----------
        * :math:`A` : amplitude of the modification :math:`\left[\text{dimensionless}\right]`
        * :math:`f` : exponential factor :math:`\left[\text{keV}^{-1}\right]`
        * :math:`E_c` : start energy of modification :math:`\left[\text{keV}\right]`

    """

    def __call__(self, energy):

        amplitude = hk.get_parameter('A', [], init=HaikuConstant(1))
        factor = hk.get_parameter('f', [], init=HaikuConstant(1))
        pivot = hk.get_parameter('E_c', [], init=HaikuConstant(1))

        return jnp.where(energy >= pivot, 1. + amplitude*jnp.exp(-factor*energy), 1.)


class Tbabs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model.

    Parameters
    ----------
        * :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """

    ref = importlib.resources.files('jaxspec') / 'tables/xsect_tbabs_wilm.fits'
    with importlib.resources.as_file(ref) as path:
        table = Table.read(path)
    energy = np.asarray(table['ENERGY']).astype(np.float32)
    sigma = np.asarray(table['SIGMA']).astype(np.float32)

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.)

        return jnp.exp(-nh*sigma)


class Phabs(MultiplicativeComponent):
    r"""
    A photoelectric absorption model.

    Parameters
    ----------
        * :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """

    ref = importlib.resources.files('jaxspec') / 'tables/xsect_phabs_aspl.fits'
    with importlib.resources.as_file(ref) as path:
        table = Table.read(path)
    energy = jnp.asarray(table['ENERGY']).astype(jnp.float32)
    sigma = jnp.asarray(table['SIGMA']).astype(jnp.float32)

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)


class Wabs(MultiplicativeComponent):
    r"""
    A photo-electric absorption using Wisconsin (Morrison & McCammon 1983) cross-sections.

    Parameters
    ----------
        * :math:`N_H` : equivalent hydrogen column density :math:`\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]`

    """
    def __init__(self):

        super(Wabs, self).__init__()
        ref = importlib.resources.files('jaxspec') / 'tables/xsect_wabs_angr.fits'
        with importlib.resources.as_file(ref) as path:
            table = Table.read(path)
        self.energy = jnp.asarray(table['ENERGY'])
        self.sigma = jnp.asarray(table['SIGMA'])

    def __call__(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)


class Gabs(MultiplicativeComponent):
    r"""
    A Gaussian absorption model.

    .. math::
        \mathcal{M}(E) = \exp \left( - \frac{\tau}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{\left(E-E_0\right)^2}{2 \sigma^2} \right) \right)

    .. note::
        The optical depth at line center is :math:`\tau/(\sqrt{2 \pi} \sigma)`.

    Parameters
    ----------
        * :math:`\tau` : absorption strength :math:`\left[\text{dimensionless}\right]`
        * :math:`\sigma` : absorption width :math:`\left[\text{keV}\right]`
        * :math:`E_0` : absorption center :math:`\left[\text{keV}\right]`

    """

    def __call__(self, energy):

        tau = hk.get_parameter('tau', [], init=HaikuConstant(1))
        sigma = hk.get_parameter('sigma', [], init=HaikuConstant(1))
        center = hk.get_parameter('E_0', [], init=HaikuConstant(1))

        return jnp.exp(-tau/(jnp.sqrt(2*jnp.pi)*sigma)*jnp.exp(-0.5*((energy-center)/sigma)**2))


class Highecut(MultiplicativeComponent):
    r"""
    A high-energy cutoff model.

    .. math::
        \mathcal{M}(E) = \begin{cases} \exp \left( \frac{E_c - E}{E_f} \right)& \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}

    Parameters
    ----------
        * :math:`E_c` : cutoff energy :math:`\left[\text{keV}\right]`
        * :math:`E_f` : e-folding energy :math:`\left[\text{keV}\right]`
    """

    def __call__(self, energy):

        cutoff = hk.get_parameter('E_c', [], init=HaikuConstant(1))
        folding = hk.get_parameter('E_f', [], init=HaikuConstant(1))

        return jnp.where(energy <= cutoff, 1., jnp.exp((cutoff-energy)/folding))
