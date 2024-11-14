from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from astropy.table import Table

from ..util.online_storage import table_manager
from .abc import MultiplicativeComponent


class MultiplicativeConstant(MultiplicativeComponent):
    r"""
    A multiplicative constant

    !!! abstract "Parameters"
        * $K$ (`norm`) $\left[\text{dimensionless}\right]$: The multiplicative constant.

    """

    def __init__(self):
        self.norm = nnx.Param(1.0)

    def factor(self, energy):
        return self.norm


class Expfac(MultiplicativeComponent):
    r"""
    An exponential modification of a spectrum.

    $$
    \mathcal{M}(E) = \begin{cases}1 + A \exp \left(-fE\right) &
    \text{if $E>E_c$}\\1 & \text{if $E<E_c$}\end{cases}
    $$

    !!! abstract "Parameters"
        * $A$ (`A`) $\left[\text{dimensionless}\right]$ : Amplitude of the modification
        * $f$ (`f`) $\left[\text{keV}^{-1}\right]$ : Exponential factor
        * $E_c$ (`E_c`) $\left[\text{keV}\right]$: Start energy of modification

    """

    def __init__(self):
        self.A = nnx.Param(1.0)
        self.f = nnx.Param(1.0)
        self.E_c = nnx.Param(1.0)

    def factor(self, energy):
        return jnp.where(energy >= self.E_c, 1.0 + self.A * jnp.exp(-self.f * energy), 1.0)


class Tbabs(MultiplicativeComponent):
    r"""
    The Tuebingen-Boulder ISM absorption model. This model calculates the cross section for X-ray absorption by the ISM
    as the sum of the cross sections for X-ray absorption due to the gas-phase ISM, the grain-phase ISM,
    and the molecules in the ISM.

    $$
        \mathcal{M}(E) = \exp^{-N_{\text{H}}\sigma(E)}
    $$

    !!! abstract "Parameters"
        * $N_{\text{H}}$ (`nh`) $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$ : Equivalent hydrogen column density


    !!! note
        Abundances and cross-sections $\sigma$ can be found in Wilms et al. (2000).

    """

    def __init__(self):
        table = Table.read(table_manager.fetch("xsect_tbabs_wilm.fits"))
        self.energy = nnx.Variable(np.asarray(table["ENERGY"], dtype=np.float64))
        self.sigma = nnx.Variable(np.asarray(table["SIGMA"], dtype=np.float64))
        self.nh = nnx.Param(1.0)

    def factor(self, energy):
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return jnp.exp(-self.nh * sigma)


class Phabs(MultiplicativeComponent):
    r"""
    A photoelectric absorption model.

    !!! abstract "Parameters"
        * $N_{\text{H}}$ (`nh`) $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$ : Equivalent hydrogen column density


    """

    def __init__(self):
        table = Table.read(table_manager.fetch("xsect_phabs_aspl.fits"))
        self.energy = nnx.Variable(np.asarray(table["ENERGY"], dtype=np.float64))
        self.sigma = nnx.Variable(np.asarray(table["SIGMA"], dtype=np.float64))
        self.nh = nnx.Param(1.0)

    def factor(self, energy):
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return jnp.exp(-self.nh * sigma)


class Wabs(MultiplicativeComponent):
    r"""
    A photo-electric absorption using Wisconsin (Morrison & McCammon 1983) cross-sections.

    ??? abstract "Parameters"
        * $N_{\text{H}}$ (`nh`) $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$ : Equivalent hydrogen column density
    """

    def __init__(self):
        table = Table.read(table_manager.fetch("xsect_wabs_angr.fits"))
        self.energy = nnx.Variable(np.asarray(table["ENERGY"], dtype=np.float64))
        self.sigma = nnx.Variable(np.asarray(table["SIGMA"], dtype=np.float64))
        self.nh = nnx.Param(1.0)

    def factor(self, energy):
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)

        return jnp.exp(-self.nh * sigma)


class Gabs(MultiplicativeComponent):
    r"""
    A Gaussian absorption model.

    $$
        \mathcal{M}(E) = \exp \left( - \frac{\tau}{\sqrt{2 \pi} \sigma} \exp
        \left( -\frac{\left(E-E_0\right)^2}{2 \sigma^2} \right) \right)
    $$

    !!! abstract "Parameters"
        * $\tau$ (`tau`) $\left[\text{dimensionless}\right]$ : Absorption strength
        * $\sigma$ (`sigma`) $\left[\text{keV}\right]$ : Absorption width
        * $E_0$ (`E0`) $\left[\text{keV}\right]$ : Absorption center

    !!! note
        The optical depth at line center is $\tau/(\sqrt{2 \pi} \sigma)$.

    """

    def __init__(self):
        self.tau = nnx.Param(1.0)
        self.sigma = nnx.Param(1.0)
        self.E0 = nnx.Param(1.0)

    def factor(self, energy):
        return jnp.exp(
            -self.tau
            / (jnp.sqrt(2 * jnp.pi) * self.sigma)
            * jnp.exp(-0.5 * ((energy - self.E0) / self.sigma) ** 2)
        )


class Highecut(MultiplicativeComponent):
    r"""
    A high-energy cutoff model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp
        \left( \frac{E_c - E}{E_f} \right)& \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    !!! abstract "Parameters"
        * $E_c$ (`Ec`) $\left[\text{keV}\right]$ : Cutoff energy
        * $E_f$ (`Ef`) $\left[\text{keV}\right]$ : Folding energy
    """

    def __init__(self):
        self.Ec = nnx.Param(1.0)
        self.Ef = nnx.Param(1.0)

    def factor(self, energy):
        return jnp.where(energy <= self.Ec, 1.0, jnp.exp((self.Ec - energy) / self.Ef))


class Zedge(MultiplicativeComponent):
    r"""
    A redshifted absorption edge model.

    $$
        \mathcal{M}(E) = \begin{cases} \exp \left( -D \left(\frac{E(1+z)}{E_c}\right)^3 \right)
        & \text{if $E > E_c$}\\ 1 & \text{if $E < E_c$}\end{cases}
    $$

    !!! abstract "Parameters"
        * $E_c$ (`Ec`) $\left[\text{keV}\right]$ : Threshold energy
        * $D$ (`D`) $\left[\text{dimensionless}\right]$ : Absorption depth at the threshold
        * $z$ (`z`) $\left[\text{dimensionless}\right]$ : Redshift
    """

    def __init__(self):
        self.Ec = nnx.Param(1.0)
        self.D = nnx.Param(1.0)
        self.z = nnx.Param(0.0)

    def factor(self, energy):
        return jnp.where(
            energy <= self.Ec, 1.0, jnp.exp(-self.D * (energy * (1 + self.z) / self.Ec) ** 3)
        )


class Tbpcf(MultiplicativeComponent):
    r"""
    Partial covering model using the Tuebingen-Boulder ISM absorption model (for more details, see `Tbabs`).

    $$
        \mathcal{M}(E) = f \exp^{-N_{\text{H}}\sigma(E)} + (1-f)
    $$

    !!! abstract "Parameters"
        * $N_{\text{H}}$ (`nh`) $\left[\frac{\text{atoms}~10^{22}}{\text{cm}^2}\right]$ : Equivalent hydrogen column density
        * $f$ (`f`) $\left[\text{dimensionless}\right]$ : Partial covering fraction, ranges between 0 and 1

    !!! note
        Abundances and cross-sections $\sigma$ can be found in Wilms et al. (2000).

    """

    def __init__(self):
        table = Table.read(table_manager.fetch("xsect_tbabs_wilm.fits"))
        self.energy = nnx.Variable(np.asarray(table["ENERGY"], dtype=np.float64))
        self.sigma = nnx.Variable(np.asarray(table["SIGMA"], dtype=np.float64))
        self.nh = nnx.Param(1.0)
        self.f = nnx.Param(0.2)

    def continuum(self, energy):
        sigma = jnp.interp(energy, self.energy, self.sigma, left=1e9, right=0.0)
        return self.f * jnp.exp(-self.nh * sigma) + (1 - self.f)


class FDcut(MultiplicativeComponent):
    r"""
    A Fermi-Dirac cutoff model.

    $$
        \mathcal{M}(E) = \left( 1 + \exp \left( \frac{E - E_c}{E_f} \right) \right)^{-1}
    $$

    ??? abstract "Parameters"
        * $E_c$ (`Ec`) $\left[\text{keV}\right]$ : Cutoff energy
        * $E_f$ (`Ef`) $\left[\text{keV}\right]$ : Folding energy
    """

    def __init__(self):
        self.Ec = nnx.Param(1.0)
        self.Ef = nnx.Param(3.0)

    def continuum(self, energy):
        return (1 + jnp.exp((energy - self.Ec) / self.Ef)) ** -1
