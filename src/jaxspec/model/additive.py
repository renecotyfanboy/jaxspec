from __future__ import annotations

from functools import partial

import astropy.constants
import astropy.units as u
import flax.nnx as nnx
import interpax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from astropy.table import Table

from ..util.integrate import integrate_interval
from ..util.online_storage import table_manager
from .abc import AdditiveComponent


class Powerlaw(AdditiveComponent):
    r"""
    A power law model

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha}$$

    !!! abstract "Parameters"
        * $\alpha$ (`alpha`) $\left[\text{dimensionless}\right]$ : Photon index of the power law
        * $E_0$ $\left[ \mathrm{keV}\right]$ : Reference energy fixed at 1 keV
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization at the reference energy (1 keV)
    """

    def __init__(self):
        self.alpha = nnx.Param(1.7)
        self.norm = nnx.Param(1e-4)

    def continuum(self, energy):
        return self.norm * energy ** (-self.alpha)


class Additiveconstant(AdditiveComponent):
    r"""
    A constant model

    $$\mathcal{M}\left( E \right) = K$$

    !!! abstract "Parameters"
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.norm = nnx.Param(1.0)

    def integrated_continuum(self, e_low, e_high):
        return (e_high - e_low) * self.norm


class Lorentz(AdditiveComponent):
    r"""
    A Lorentzian line profile

    $$\mathcal{M}\left( E \right) = K\frac{\frac{\sigma}{2\pi}}{(E-E_L)^2 + \left(\frac{\sigma}{2}\right)^2}$$

    !!! abstract "Parameters"
        - $E_L$ (`E_l`) $\left[\text{keV}\right]$ : Energy of the line
        - $\sigma$ (`sigma`) $\left[\text{keV}\right]$ : FWHM of the line
        - $K$ (`norm`) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.E_l = jnp.asarray(nnx.Param(1.0), dtype=jnp.float64)
        self.sigma = jnp.asarray(nnx.Param(1e-3), dtype=jnp.float64)
        self.norm = jnp.asarray(nnx.Param(1.0), dtype=jnp.float64)

    def continuum(self, energy):
        return (
            self.norm
            * self.sigma
            / (2 * jnp.pi)
            / ((energy - self.E_l) ** 2 + (self.sigma / 2) ** 2)
        )


class Logparabola(AdditiveComponent):
    r"""
    A LogParabola model

    $$
    \mathcal{M}\left( E \right) = K  \left( \frac{E}{E_{\text{Pivot}}} \right)
    ^{-(\alpha - \beta ~ \log (E/E_{\text{Pivot}})) }
    $$

    !!! abstract "Parameters"
        * $a$ (`a`) $\left[\text{dimensionless}\right]$ : Slope of the LogParabola at the pivot energy
        * $b$ (`b`) $\left[\text{dimensionless}\right]$ : Curve parameter of the LogParabola
        * $E_{\text{Pivot}}$ $\left[ \mathrm{keV}\right]$ : Pivot energy fixed at 1 keV
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.a = nnx.Param(1.0)
        self.b = nnx.Param(1.0)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy):
        return self.norm * energy ** (-(self.a - self.b * jnp.log(energy)))


class Blackbody(AdditiveComponent):
    r"""
    A black body model

    $$\mathcal{M}\left( E \right) = \frac{K \times 8.0525 E^{2}}{(k_B T)^{4}\left(\exp(E/k_BT)-1\right)}$$

    !!! abstract "Parameters"
        * $k_B T$ (`kT`) $\left[\text{keV}\right]$ : Temperature
        * $K$ (`norm`) $\left[\text{dimensionless}\right]$ : $L_{39}/D_{10}^{2}$, where $L_{39}$ is the source luminosity [$10^{39} \frac{\text{erg}}{\text{s}}$]
        and $D_{10}$ is the distance to the source [$10 \text{kpc}$]
    """

    # TODO : rewrite constant as a astropy unit stuff

    def __init__(self):
        self.kT = nnx.Param(0.5)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy):
        return self.norm * 8.0525 * energy**2 / ((self.kT**4) * jnp.expm1(energy / self.kT))


class Blackbodyrad(AdditiveComponent):
    r"""
    A black body model in radius normalization

    $$\mathcal{M}\left( E \right) = \frac{K \times 1.0344\times 10^{-3} E^{2}}{\left(\exp (E/k_BT)-1\right)}$$

    !!! abstract "Parameters"
        * $k_B T$ (`kT`) $\left[\text{keV}\right]$ : Temperature
        * $K$ (`norm`) [dimensionless] : $R^2_{km}/D_{10}^{2}$, where $R_{km}$ is the source radius [$\text{km}$]
        and $D_{10}$ is the distance to the source [$10 \text{kpc}$]
    """

    def __init__(self):
        self.kT = nnx.Param(0.5)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy):
        return self.norm * 1.0344e-3 * energy**2 / jnp.expm1(energy / self.kT)


class Gauss(AdditiveComponent):
    r"""
    A Gaussian line profile. If the width is $\leq 0$ then it is treated as a delta function.
    The `Zgauss` variant computes a redshifted Gaussian.

    $$\mathcal{M}\left( E \right) = \frac{K}{\sigma \sqrt{2 \pi}}\exp\left(\frac{-(E-E_L)^2}{2\sigma^2}\right)$$

    !!! abstract "Parameters"
        * $E_L$ (`E_l`) $\left[\text{keV}\right]$ : Energy of the line
        * $\sigma$ (`sigma`) $\left[\text{keV}\right]$ : Width of the line
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.E_l = nnx.Param(2.0)
        self.sigma = nnx.Param(1e-2)
        self.norm = nnx.Param(1.0)

    def integrated_continuum(self, e_low, e_high):
        return self.norm * (
            jsp.stats.norm.cdf(
                e_high,
                loc=jnp.asarray(self.E_l, dtype=jnp.float64),
                scale=jnp.asarray(self.sigma, dtype=jnp.float64),
            )
            - jsp.stats.norm.cdf(
                e_low,
                loc=jnp.asarray(self.E_l, dtype=jnp.float64),
                scale=jnp.asarray(self.sigma, dtype=jnp.float64),
            )
        )


class Cutoffpl(AdditiveComponent):
    r"""
    A power law model with high energy exponential cutoff

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha} \exp(-E/\beta)$$

    !!! abstract "Parameters"
        * $\alpha$ (`alpha`) $\left[\text{dimensionless}\right]$ : Photon index of the power law
        * $\beta$ (`beta`) $\left[\text{keV}\right]$ : Folding energy of the exponential cutoff
        * $E_0$ $\left[ \mathrm{keV}\right]$ : Reference energy fixed at 1 keV
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.alpha = nnx.Param(1.7)
        self.beta = nnx.Param(15.0)
        self.norm = nnx.Param(1e-4)

    def continuum(self, energy):
        return self.norm * energy ** (-self.alpha) * jnp.exp(-energy / self.beta)


class Diskbb(AdditiveComponent):
    r"""
    `Diskpbb` with $p=0.75$

    !!! abstract "Parameters"
        * $T_{\text{in}}$ (`Tin`) $\left[ \mathrm{keV}\right]$: Temperature at inner disk radius
        * $\text{norm}$ (`norm`) $\left[\text{dimensionless}\right]$ : $\cos i(r_{\text{in}}/d)^{2}$,
        where $r_{\text{in}}$ is an apparent inner disk radius $\left[\text{km}\right]$,
        $d$ the distance to the source [$10 \text{kpc}$], $i$ the angle of the disk ($i=0$ is face-on)
    """

    def __init__(self):
        self.Tin = nnx.Param(1.0)
        self.norm = nnx.Param(1e-4)

    def continuum(self, energy):
        p = 0.75
        tout = 0.0

        # Tout is set to 0 as it is evaluated at R=infinity
        def integrand(kT, e, tin, p):
            return e**2 * (kT / tin) ** (-2 / p - 1) / (jnp.exp(e / kT) - 1)

        integral = integrate_interval(integrand)
        return (
            self.norm
            * 2.78e-3
            * (0.75 / p)
            / self.Tin
            * jnp.vectorize(lambda e: integral(tout, self.Tin, e, self.Tin, p))(energy)
        )


class Agauss(AdditiveComponent):
    r"""
    A simple Gaussian line profile in wavelength space.
    If the width is $\leq 0$ then it is treated as a delta function.
    The `Zagauss` variant computes a redshifted Gaussian.

    $$\mathcal{M}\left( \lambda \right) =
    \frac{K}{\sigma \sqrt{2 \pi}} \exp\left(\frac{-(\lambda - \lambda_L)^2}{2 \sigma^2}\right)$$

    !!! abstract "Parameters"
        * $\lambda_L$ (`lambda_l`) $\left[\unicode{x212B}\right]$ : Wavelength of the line in Angström
        * $\sigma$ (`sigma`) $\left[\unicode{x212B}\right]$ : Width of the line width in Angström
        * $K$ (`norm`) $\left[\frac{\unicode{x212B}~\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$: Normalization
    """

    def __init__(self):
        self.lambda_l = nnx.Param(12.0)
        self.sigma = nnx.Param(1e-2)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy) -> (jax.Array, jax.Array):
        hc = (astropy.constants.h * astropy.constants.c).to(u.angstrom * u.keV).value

        return self.norm * jsp.stats.norm.pdf(
            hc / energy,
            loc=jnp.asarray(self.lambda_l, dtype=jnp.float64),
            scale=jnp.asarray(self.sigma, dtype=jnp.float64),
        )


class Zagauss(AdditiveComponent):
    r"""
    A redshifted Gaussian line profile in wavelength space.
    If the width is $\leq 0$ then it is treated as a delta function.

    $$\mathcal{M}\left( \lambda \right) =
    \frac{K (1+z)}{\sigma \sqrt{2 \pi}} \exp\left(\frac{-(\lambda/(1+z) - \lambda_L)^2}{2 \sigma^2}\right)$$

    !!! abstract "Parameters"
        * $\lambda_L$ (`lambda_l`) $\left[\unicode{x212B}\right]$  : Wavelength of the line in Angström
        * $\sigma$ (`sigma`) $\left[\unicode{x212B}\right]$  : Width of the line width in Angström
        * $z$ (`redshift`) $\left[\text{dimensionless}\right]$ : Redshift
        * $K$ (`norm`) $\left[\frac{\unicode{x212B}~\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization
    """

    def __init__(self):
        self.lambda_l = nnx.Param(12.0)
        self.sigma = nnx.Param(1e-2)
        self.redshift = nnx.Param(0.0)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy) -> (jax.Array, jax.Array):
        hc = (astropy.constants.h * astropy.constants.c).to(u.angstrom * u.keV).value

        redshift = self.redshift

        return (
            self.norm
            * (1 + redshift)
            * jsp.stats.norm.pdf(
                (hc / energy) / (1 + redshift),
                loc=jnp.asarray(self.lambda_l, dtype=jnp.float64),
                scale=jnp.asarray(self.sigma, dtype=jnp.float64),
            )
        )


class Zgauss(AdditiveComponent):
    r"""
    A redshifted Gaussian line profile. If the width is $\leq 0$ then it is treated as a delta function.

    $$\mathcal{M}\left( E \right) =
    \frac{K}{(1+z) \sigma \sqrt{2 \pi}}\exp\left(\frac{-(E(1+z)-E_L)^2}{2\sigma^2}\right)$$

    !!! abstract "Parameters"
        * $E_L$ (`E_l`) $\left[\text{keV}\right]$ : Energy of the line
        * $\sigma$ (`sigma`) $\left[\text{keV}\right]$ : Width of the line
        * $K$ (`norm`) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$ : Normalization
        * $z$ (`redshift`) $\left[\text{dimensionless}\right]$ : Redshift
    """

    def __init__(self):
        self.E_l = nnx.Param(2.0)
        self.sigma = nnx.Param(1e-2)
        self.redshift = nnx.Param(0.0)
        self.norm = nnx.Param(1.0)

    def continuum(self, energy) -> (jax.Array, jax.Array):
        return (self.norm / (1 + self.redshift)) * jsp.stats.norm.pdf(
            energy * (1 + self.redshift),
            loc=jnp.asarray(self.E_l, dtype=jnp.float64),
            scale=jnp.asarray(self.sigma, dtype=jnp.float64),
        )


class NSatmos(AdditiveComponent):
    r"""
    A neutron star atmosphere model based on the `NSATMOS` model from `XSPEC`. See [this page](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/node205.html)

    !!! warning
        The boundary case of $R_{\text{NS}} < 1.125 R_{\text{S}}$ is handled with a null flux instead of a constant value as in `XSPEC`.

    !!! abstract "Parameters"
        * $T_{eff}$ (`Tinf`) $\left[\text{Kelvin}\right]$ : Effective temperature at the surface (No redshift applied)
        * $M_{ns}$ (`mass`) $\left[M_{\odot}\right]$ : Mass of the NS
        * $R_∞$  (`radius`) $\left[\text{km}\right]$ : Radius at infinity (modulated by gravitational effects)
        * $D$ (`distance`) $\left[\text{kpc}\right]$ : Distance to the neutron star
        * norm (`norm`) $\left[\text{dimensionless}\right]$ : fraction of the neutron star surface emitting
    """

    def __init__(self):
        self.Tinf = nnx.Param(6.0)
        self.mass = nnx.Param(1.4)
        self.radius = nnx.Param(10.0)
        self.distance = nnx.Param(10.0)
        self.norm = nnx.Param(1.0)

        entry_table = Table.read(table_manager.fetch("nsatmosdata.fits"), 1)

        # Extract the table values. All this code could be summarized in two lines if we reformat the nsatmosdata.fits table
        tab_temperature = np.asarray(entry_table["TEMP"][0], dtype=float)  # Logarithmic value
        tab_gravity = np.asarray(entry_table["GRAVITY"][0], dtype=float)  # Logarithmic value
        tab_mucrit = np.asarray(entry_table["MUCRIT"][0], dtype=float)
        tab_energy = np.asarray(entry_table["ENERGY"][0], dtype=float)
        tab_flux_flat = Table.read(table_manager.fetch("nsatmosdata.fits"), 2)["FLUX"]

        tab_flux = np.empty(
            (
                tab_temperature.size,
                tab_gravity.size,
                tab_mucrit.size,
                tab_energy.size,
            )
        )

        for i in range(len(tab_temperature)):
            for j in range(len(tab_gravity)):
                for k in range(len(tab_mucrit)):
                    tab_flux[i, j, k] = np.array(
                        tab_flux_flat[
                            i * len(tab_gravity) * len(tab_mucrit) + j * len(tab_mucrit) + k
                        ]
                    )

        tab_flux = np.asarray(tab_flux, dtype=float)

        self.tab_temperature = nnx.Variable(tab_temperature)
        self.tab_gravity = nnx.Variable(tab_gravity)
        self.tab_mucrit = nnx.Variable(tab_mucrit)
        self.tab_energy = nnx.Variable(tab_energy)
        self.tab_flux = nnx.Variable(tab_flux)

    def interp_flux_func(self, temperature_log, gravity_log, mu):
        return interpax.interp3d(
            10.0**temperature_log,
            10.0**gravity_log,
            mu,
            10.0**self.tab_temperature,
            10.0**self.tab_gravity,
            self.tab_mucrit,
            self.tab_flux,
            method="linear",
        )

    @partial(jnp.vectorize, excluded=(0,))
    def continuum(self, energy):
        # log10 of temperature in Kelvin
        # 'Tinf': temp_log, # 5 to 6.5
        # 'M': mass, # 0.5 to 3
        #  'Rns': radius, # 5 to 30

        temp_log = self.Tinf
        mass = self.mass
        radius = self.radius
        distance = self.distance
        norm = self.norm

        # Derive parameters usable to retrive value in the flux table
        Rcgs = 1e5 * radius  # Radius in cgs
        r_schwarzschild = 2.95e5 * mass  # Schwarzschild radius in cgs
        r_normalized = Rcgs / r_schwarzschild  # Ratio of the radius to the Schwarzschild radius
        r_over_Dsql = 2 * jnp.log10(
            Rcgs / (3.09e21 * distance)
        )  # Log( (R/D)**2 ), 3.09e21 constant transforms radius in cgs to kpc
        zred1 = 1 / jnp.sqrt(1 - (1 / r_normalized))  # Gravitational redshift
        gravity = (6.67e-8 * 1.99e33 * mass / Rcgs**2) * zred1  # Gravity field g in cgs
        gravity_log = jnp.log10(
            gravity
        )  # Log gravity because this is the format given in the table

        # Not sure about mu yet, but it is linked to causality
        cmu = jnp.where(
            r_normalized < 1.5, jnp.sqrt(1.0 - 6.75 / r_normalized**2 + 6.75 / r_normalized**3), 0.0
        )

        # Interpolate the flux table to get the flux at the surface

        flux = self.interp_flux_func(temp_log, gravity_log, cmu)

        # Rescale the photon energies and fluxes back to the correct local temperature
        Tfactor = 10.0 ** (temp_log - 6.0)
        fluxshift = 3.0 * (temp_log - 6.0)
        E = self.tab_energy * Tfactor
        flux += fluxshift

        # Rescale applying redshift
        fluxshift = -jnp.log10(zred1)
        E = E / zred1
        flux += fluxshift

        # Convert to counts/keV (which corresponds to dividing by1.602e-9*EkeV)
        # Multiply by the area of the star, and calculate the count rate at the observer
        flux += r_over_Dsql
        counts = 10.0 ** (flux - jnp.log10(1.602e-9) - jnp.log10(E))

        true_flux = norm * jnp.exp(
            interpax.interp1d(jnp.log(energy), jnp.log(E), jnp.log(counts), method="linear")
        )

        return jax.lax.select(r_normalized < 1.125, jnp.zeros_like(true_flux), true_flux)


class Band(AdditiveComponent):
    r"""
    A Band function model

    $$
    \mathcal{M}(E) = \begin{cases} K \left( \frac{E}{100 \, \text{keV}}\right)^{\alpha_1}\exp(-\frac{E}{E_c})  &
    \text{if $E < E_c (\alpha_1 - \alpha_2)$} \\
    K \left[ (\alpha_1 - \alpha_2) \frac{E_c}{100 \, \text{keV}} \right]^{\alpha_1-\alpha_2} \left( \frac{E}{100 \, \text{keV}}\right)^{\alpha_2} \exp(-(\alpha_1 - \alpha_2))  & \text{if $E > E_c (\alpha_1 - \alpha_2)$}
    \end{cases}
    $$

    !!! abstract "Parameters"
        * $\alpha_1$ (`alpha1`) $\left[\text{dimensionless}\right]$ : First powerlaw index
        * $\alpha_2$  (`alpha2`) $\left[\text{dimensionless}\right]$ : Second powerlaw index
        * $E_c$  (`Ec`) $\left[\text{keV}\right]$ : Radius at infinity (modulated by gravitational effects)
        * norm (`norm`) $\left[\frac{\text{photons}}{\text{keV}\text{cm}^2\text{s}}\right]$ : Normalization at the reference energy (100 keV)
    """

    def __init__(self):
        self.alpha1 = nnx.Param(-1.0)
        self.alpha2 = nnx.Param(-2.0)
        self.Ec = nnx.Param(200.0)
        self.norm = nnx.Param(1e-4)

    def continuum(self, energy):
        Epivot = 100.0
        alpha_diff = jnp.asarray(self.alpha1) - jnp.asarray(self.alpha2)

        spectrum = jnp.where(
            energy < self.Ec * (alpha_diff),
            (energy / Epivot) ** self.alpha1 * jnp.exp(-energy / self.Ec),
            (alpha_diff * (self.Ec / Epivot)) ** (alpha_diff)
            * (energy / 100) ** self.alpha2
            * jnp.exp(-alpha_diff),
        )

        return self.norm * spectrum


'''
class Diskpbb(AdditiveComponent):
    r"""
    A multiple blackbody disk model where local disk temperature T(r) is proportional to $$r^{(-p)}$$,
    where $$p$$ is a free parameter. The standard disk model, diskbb, is recovered if $$p=0.75$$.
    If radial advection is important then $$p<0.75$$.

    $$\\mathcal{M}\\left( E \right) = \frac{2\\pi(\\cos i)r^{2}_{\text{in}}}{pd^2} \\int_{T_{\text{in}}}^{T_{\text{out}}}
    \\left( \frac{T}{T_{\text{in}}} \right)^{-2/p-1} \text{bbody}(E,T) \frac{dT}{T_{\text{in}}}$$

    ??? abstract "Parameters"
        * $\text{norm}$ : $\\cos i(r_{\text{in}}/d)^{2}$,
        where $r_{\text{in}}$ is "an apparent" inner disk radius $\\left[\text{km}\right]$,
        $d$ the distance to the source in units of $10 \text{kpc}$,
        $i$ the angle of the disk ($i=0$ is face-on)
        * $p$ : Exponent of the radial dependence of the disk temperature $\\left[\text{dimensionless}\right]$
        * $T_{\text{in}}$ : Temperature at inner disk radius $\\left[ \\mathrm{keV}\right]$
    """

    def continuum(self, energy):
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))
        p = hk.get_parameter("p", [], float, init=HaikuConstant(0.75))
        tin = hk.get_parameter("Tin", [], float, init=HaikuConstant(1))

        # Tout is set to 0 as it is evaluated at R=infinity
        def integrand(kT, energy):
            return 2.78e-3 * energy**2 * (kT / tin) ** (-2 / p - 1) / (jnp.exp(energy / kT) - 1)

        func_vmapped = jax.vmap(lambda e: integrate_interval(lambda kT: integrand(kT, e), 0, tin, n=51))

        return norm * (0.75 / p) * func_vmapped(energy)
'''
