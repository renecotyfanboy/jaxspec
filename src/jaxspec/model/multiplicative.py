from __future__ import annotations

import haiku as hk
import jax.numpy as jnp
import numpy as np

from astropy.table import Table
from haiku.initializers import Constant as HaikuConstant

from ..util.online_storage import table_manager
from .abc import MultiplicativeComponent


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum.

    $$
    \mathcal{M}(E) = \begin{cases}1 + A \exp \left(-fE\right) &
    \text{if $E>E_c$}\\1 & \text{if $E<E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $A$ : Amplitude of the modification $\left[\text{dimensionless}\right]$
        * $f$ : Exponential factor $\left[\text{keV}^{-1}\right]$
        * $E_c$ : Start energy of modification $\left[\text{keV}\right]$

    """

    def continuum(self, energy):
        amplitude = hk.get_parameter("A", [], float, init=HaikuConstant(1))
        factor = hk.get_parameter("f", [], float, init=HaikuConstant(1))
        pivot = hk.get_parameter("E_c", [], float, init=HaikuConstant(1))

        return jnp.where(energy >= pivot, 1.0 + amplitude * jnp.exp(-factor * energy), 1.0)


class Tbabs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model. This model calculates the cross section for X-ray absorption by the ISM
    as the sum of the cross sections for X-ray absorption due to the gas-phase ISM, the grain-phase ISM,
    and the molecules in the ISM.

    $$
        \mathcal{M}(E) = \exp^{-N_{\text{H}}\sigma(E)}
    $$

    ??? abstract "Parameters"
        * $N_{\text{H}}$ : Equivalent hydrogen column density
            $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    !!! note
        Abundances and cross-sections $\sigma$ can be found in Wilms et al. (2000).

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = Table.read(table_manager.fetch("xsect_tbabs_wilm.fits"))
        self.energy = jnp.asarray(np.array(table["ENERGY"]), dtype=np.float64)
        self.sigma = jnp.asarray(np.array(table["SIGMA"]), dtype=np.float64)

    def continuum(self, energy):
        nh = hk.get_parameter("N_H", [], float, init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return jnp.exp(-nh * sigma)


class Phabs(MultiplicativeComponent):
    r"""
    A photoelectric absorption model.

    ??? abstract "Parameters"
        * $N_{\text{H}}$ : Equivalent hydrogen column density
            $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = Table.read(table_manager.fetch("xsect_phabs_aspl.fits"))
        self.energy = jnp.asarray(np.array(table["ENERGY"]), dtype=np.float64)
        self.sigma = jnp.asarray(np.array(table["SIGMA"]), dtype=np.float64)

    def continuum(self, energy):
        nh = hk.get_parameter("N_H", [], float, init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=jnp.inf, right=0.0)

        return jnp.exp(-nh * sigma)


class Wabs(MultiplicativeComponent):
    r"""
    A photo-electric absorption using Wisconsin (Morrison & McCammon 1983) cross-sections.

    ??? abstract "Parameters"
        * $N_{\text{H}}$ : Equivalent hydrogen column density
            $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = Table.read(table_manager.fetch("xsect_wabs_angr.fits"))
        self.energy = jnp.asarray(np.array(table["ENERGY"]), dtype=np.float64)
        self.sigma = jnp.asarray(np.array(table["SIGMA"]), dtype=np.float64)

    def continuum(self, energy):
        nh = hk.get_parameter("N_H", [], float, init=HaikuConstant(1))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return jnp.exp(-nh * sigma)


class Gabs(MultiplicativeComponent):
    r"""
    A Gaussian absorption model.

    $$
        \mathcal{M}(E) = \exp \left( - \frac{\tau}{\sqrt{2 \pi} \sigma} \exp
        \left( -\frac{\left(E-E_0\right)^2}{2 \sigma^2} \right) \right)
    $$

    ??? abstract "Parameters"
        * $\tau$ : Absorption strength $\left[\text{dimensionless}\right]$
        * $\sigma$ : Absorption width $\left[\text{keV}\right]$
        * $E_0$ : Absorption center $\left[\text{keV}\right]$

    !!! note
        The optical depth at line center is $\tau/(\sqrt{2 \pi} \sigma)$.

    """

    def continuum(self, energy):
        tau = hk.get_parameter("tau", [], float, init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(1))
        center = hk.get_parameter("E_0", [], float, init=HaikuConstant(1))

        return jnp.exp(
            -tau / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(-0.5 * ((energy - center) / sigma) ** 2)
        )


class Highecut(MultiplicativeComponent):
    r"""
    A high-energy cutoff model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp
        \left( \frac{E_c - E}{E_f} \right)& \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $E_c$ : Cutoff energy $\left[\text{keV}\right]$
        * $E_f$ : Folding energy $\left[\text{keV}\right]$
    """

    def continuum(self, energy):
        cutoff = hk.get_parameter("E_c", [], float, init=HaikuConstant(1))
        folding = hk.get_parameter("E_f", [], float, init=HaikuConstant(1))

        return jnp.where(energy <= cutoff, 1.0, jnp.exp((cutoff - energy) / folding))


class Zedge(MultiplicativeComponent):
    r"""
    A redshifted absorption edge model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp \left( -D \left(\frac{E(1+z)}{E_c}\right)^3 \right)
        & \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    ??? abstract "Parameters"
        * $E_c$ : Threshold energy
        * $E_f$ : Absorption depth at the threshold
        * $z$ : Redshift [dimensionless]
    """

    def continuum(self, energy):
        E_c = hk.get_parameter("E_c", [], float, init=HaikuConstant(1))
        D = hk.get_parameter("D", [], float, init=HaikuConstant(1))
        z = hk.get_parameter("z", [], float, init=HaikuConstant(0))

        return jnp.where(energy <= E_c, 1.0, jnp.exp(-D * (energy * (1 + z) / E_c) ** 3))


class Tbpcf(MultiplicativeComponent):
    r"""
    Partial covering model using the Tuebingen-Boulder ISM absorption model (for more details, see `Tbabs`).

    $$
        \mathcal{M}(E) = f \exp^{-N_{\text{H}}\sigma(E)} + (1-f)
    $$

    ??? abstract "Parameters"
        * $N_{\text{H}}$ : Equivalent hydrogen column density
            $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$
        * $f$ : Partial covering fraction, ranges between 0 and 1 [dimensionless]

    !!! note
        Abundances and cross-sections $\sigma$ can be found in Wilms et al. (2000).

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = Table.read(table_manager.fetch("xsect_tbabs_wilm.fits"))
        self.energy = jnp.asarray(np.array(table["ENERGY"]), dtype=np.float64)
        self.sigma = jnp.asarray(np.array(table["SIGMA"]), dtype=np.float64)

    def continuum(self, energy):
        nh = hk.get_parameter("N_H", [], float, init=HaikuConstant(1))
        f = hk.get_parameter("f", [], float, init=HaikuConstant(0.2))
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return f * jnp.exp(-nh * sigma) + (1 - f)


class FDcut(MultiplicativeComponent):
    r"""
    A Fermi-Dirac cutoff model.

    $$
        \mathcal{M}(E) = \left( 1 + \exp \left( \frac{E - E_c}{E_f} \right) \right)^{-1}
    $$

    ??? abstract "Parameters"
        * $E_c$ : Cutoff energy $\left[\text{keV}\right]$
        * $E_f$ : Folding energy $\left[\text{keV}\right]$
    """

    def continuum(self, energy):
        cutoff = hk.get_parameter("E_c", [], init=HaikuConstant(1))
        folding = hk.get_parameter("E_f", [], init=HaikuConstant(1))

        return (1 + jnp.exp((energy - cutoff) / folding)) ** -1
