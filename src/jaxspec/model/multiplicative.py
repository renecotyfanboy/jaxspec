from __future__ import annotations

from abc import ABC, abstractmethod

import haiku as hk
import jax.numpy as jnp
import numpy as np
import importlib.resources
from haiku.initializers import Constant as HaikuConstant
from astropy.table import Table
from . import ModelComponent


class MultiplicativeComponent(ModelComponent, ABC):
    type = 'multiplicative'

    @abstractmethod
    def continuum(self, energy):
        ...


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum.

    $$
    \mathcal{M}(E) = \begin{cases}1 + A \exp \left(-fE\right) & \text{if $E>E_c$}\\1 & \text{if $E<E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $A$ : amplitude of the modification $\left[\text{dimensionless}\right]$
        * $f$ : exponential factor $\left[\text{keV}^{-1}\right]$
        * $E_c$ : start energy of modification $\left[\text{keV}\right]$

    """

    def continuum(self, energy):

        amplitude = hk.get_parameter('A', [], init=HaikuConstant(1))
        factor = hk.get_parameter('f', [], init=HaikuConstant(1))
        pivot = hk.get_parameter('E_c', [], init=HaikuConstant(1))

        return jnp.where(energy >= pivot, 1. + amplitude*jnp.exp(-factor*energy), 1.)


class Tbabs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model.

    ??? abstract "Parameters"
        * $N_H$ : equivalent hydrogen column density $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    """

    ref = importlib.resources.files('jaxspec') / 'tables/xsect_tbabs_wilm.fits'
    with importlib.resources.as_file(ref) as path:
        table = Table.read(path)
    energy = jnp.asarray(np.array(table['ENERGY']), dtype=np.float32)
    sigma = jnp.asarray(np.array(table['SIGMA']), dtype=np.float32)

    def continuum(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.)

        return jnp.exp(-nh*sigma)


class Phabs(MultiplicativeComponent):
    r"""
    A photoelectric absorption model.

    ??? abstract "Parameters"
        * $N_H$ : equivalent hydrogen column density $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    """

    ref = importlib.resources.files('jaxspec') / 'tables/xsect_phabs_aspl.fits'
    with importlib.resources.as_file(ref) as path:
        table = Table.read(path)
    energy = jnp.asarray(np.array(table['ENERGY']), dtype=np.float32)
    sigma = jnp.asarray(np.array(table['SIGMA']), dtype=np.float32)

    def continuum(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.)

        return jnp.exp(-nh*sigma)


class Wabs(MultiplicativeComponent):
    r"""
    A photo-electric absorption using Wisconsin (Morrison & McCammon 1983) cross-sections.

    ??? abstract "Parameters"
        * $N_H$ : equivalent hydrogen column density $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    """

    ref = importlib.resources.files('jaxspec') / 'tables/xsect_wabs_angr.fits'
    with importlib.resources.as_file(ref) as path:
        table = Table.read(path)
    energy = jnp.asarray(np.array(table['ENERGY']), dtype=np.float32)
    sigma = jnp.asarray(np.array(table['SIGMA']), dtype=np.float32)

    def continuum(self, energy):

        nh = hk.get_parameter('N_H', [], init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.)

        return jnp.exp(-nh*sigma)


class Gabs(MultiplicativeComponent):
    r"""
    A Gaussian absorption model.

    $$
        \mathcal{M}(E) = \exp \left( - \frac{\tau}{\sqrt{2 \pi} \sigma} \exp
        \left( -\frac{\left(E-E_0\right)^2}{2 \sigma^2} \right) \right)
    $$

    ??? abstract "Parameters"
        * $\tau$ : absorption strength $\left[\text{dimensionless}\right]$
        * $\sigma$ : absorption width $\left[\text{keV}\right]$
        * $E_0$ : absorption center $\left[\text{keV}\right]$

    !!! note
        The optical depth at line center is $\tau/(\sqrt{2 \pi} \sigma)$.

    """

    def continuum(self, energy):

        tau = hk.get_parameter('tau', [], init=HaikuConstant(1))
        sigma = hk.get_parameter('sigma', [], init=HaikuConstant(1))
        center = hk.get_parameter('E_0', [], init=HaikuConstant(1))

        return jnp.exp(-tau/(jnp.sqrt(2*jnp.pi)*sigma)*jnp.exp(-0.5*((energy-center)/sigma)**2))


class Highecut(MultiplicativeComponent):
    r"""
    A high-energy cutoff model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp \left( \frac{E_c - E}{E_f} \right)& \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $E_c$ : cutoff energy $\left[\text{keV}\right]$
        * $E_f$ : e-folding energy $\left[\text{keV}\right]$
    """

    def continuum(self, energy):

        cutoff = hk.get_parameter('E_c', [], init=HaikuConstant(1))
        folding = hk.get_parameter('E_f', [], init=HaikuConstant(1))

        return jnp.where(energy <= cutoff, 1., jnp.exp((cutoff-energy)/folding))


class Zedge(MultiplicativeComponent):
    r"""
    A redshifted absorption edge model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp \left( -D \left(\frac{E(1+z)}{E_c}\right)^3 \right)
        & \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $E_c$ : threshold energy
        * $E_f$ : absorption depth at the threshold
        * $z$ : redshift
    """

    def continuum(self, energy):

        E_c = hk.get_parameter('E_c', [], init=HaikuConstant(1))
        D = hk.get_parameter('D', [], init=HaikuConstant(1))
        z = hk.get_parameter('z', [], init=HaikuConstant(0))

        return jnp.where(energy <= E_c, 1., jnp.exp(-D*(energy*(1+z)/E_c)**3))