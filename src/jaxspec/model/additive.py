from __future__ import annotations

import astropy.constants
import astropy.units as u
import haiku as hk
import interpax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from astropy.table import Table
from haiku.initializers import Constant as HaikuConstant

from ..util.integrate import integrate_interval
from ..util.online_storage import table_manager
from .abc import AdditiveComponent


class Powerlaw(AdditiveComponent):
    r"""
    A power law model

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha}$$

    ??? abstract "Parameters"
        * $\alpha$ : Photon index of the power law $\left[\text{dimensionless}\right]$
        * $E_0$ : Reference energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
        * $K$ : Normalization at the reference energy (1 keV) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        alpha = hk.get_parameter("alpha", [], float, init=HaikuConstant(1.3))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1e-4))

        return norm * energy ** (-alpha)


class Additiveconstant(AdditiveComponent):
    r"""
    A constant model

    $$\mathcal{M}\left( E \right) = K$$

    ??? abstract "Parameters"
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * jnp.ones_like(energy)

    def primitive(self, energy):
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * energy


class Lorentz(AdditiveComponent):
    r"""
    A Lorentzian line profile

    $$\mathcal{M}\left( E \right) = K\frac{\frac{\sigma}{2\pi}}{(E-E_L)^2 + \left(\frac{\sigma}{2}\right)^2}$$

    ??? abstract "Parameters"
        - $E_L$ : Energy of the line $\left[\text{keV}\right]$
        - $\sigma$ : FWHM of the line $\left[\text{keV}\right]$
        - $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        line_energy = hk.get_parameter("E_l", [], float, init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * sigma / (2 * jnp.pi) / ((energy - line_energy) ** 2 + (sigma / 2) ** 2)


class Logparabola(AdditiveComponent):
    r"""
    A LogParabola model

    $$
    \mathcal{M}\left( E \right) = K  \left( \frac{E}{E_{\text{Pivot}}} \right)
    ^{-(\alpha + \beta ~ \log (E/E_{\text{Pivot}})) }
    $$

    ??? abstract "Parameters"
        * $a$ : Slope of the LogParabola at the pivot energy $\left[\text{dimensionless}\right]$
        * $b$ : Curve parameter of the LogParabola $\left[\text{dimensionless}\right]$
        * $E_{\text{Pivot}}$ : Pivot energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
        * $K$ : Normalization at the pivot energy (1keV) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    # TODO : conform with xspec definition
    def continuum(self, energy):
        a = hk.get_parameter("a", [], float, init=HaikuConstant(11 / 3))
        b = hk.get_parameter("b", [], float, init=HaikuConstant(0.2))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * energy ** (-(a + b * jnp.log(energy)))


class Blackbody(AdditiveComponent):
    r"""
    A black body model

    $$\mathcal{M}\left( E \right) = \frac{K \times 8.0525 E^{2}}{(k_B T)^{4}\left(\exp(E/k_BT)-1\right)}$$

    ??? abstract "Parameters"
        * $k_B T$ : Temperature $\left[\text{keV}\right]$
        * $K$ : $L_{39}/D_{10}^{2}$, where $L_{39}$ is the source luminosity in units of $10^{39}$ erg/s
        and $D_{10}$ is the distance to the source in units of 10 kpc
    """

    def continuum(self, energy):
        kT = hk.get_parameter("kT", [], float, init=HaikuConstant(11 / 3))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * 8.0525 * energy**2 / ((kT**4) * (jnp.exp(energy / kT) - 1))


class Blackbodyrad(AdditiveComponent):
    r"""
    A black body model in radius normalization

    $$\mathcal{M}\left( E \right) = \frac{K \times 1.0344\times 10^{-3} E^{2}}{\left(\exp (E/k_BT)-1\right)}$$

    ??? abstract "Parameters"
        * $k_B T$ : Temperature $\left[\text{keV}\right]$
        * $K$ : $R^2_{km}/D_{10}^{2}$, where $R_{km}$ is the source radius in km
        and $D_{10}$ is the distance to the source in units of 10 kpc [dimensionless]
    """

    def continuum(self, energy):
        kT = hk.get_parameter("kT", [], float, init=HaikuConstant(11 / 3))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * 1.0344e-3 * energy**2 / (jnp.exp(energy / kT) - 1)


class Gauss(AdditiveComponent):
    r"""
    A Gaussian line profile. If the width is $$\leq 0$$ then it is treated as a delta function.
    The `Zgauss` variant computes a redshifted Gaussian.

    $$\mathcal{M}\left( E \right) = \frac{K}{\sigma \sqrt{2 \pi}}\exp\left(\frac{-(E-E_L)^2}{2\sigma^2}\right)$$

    ??? abstract "Parameters"
        * $E_L$ : Energy of the line $\left[\text{keV}\right]$
        * $\sigma$ : Width of the line $\left[\text{keV}\right]$
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy) -> (jax.Array, jax.Array):
        line_energy = hk.get_parameter("E_l", [], float, init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * jsp.stats.norm.pdf(energy, loc=line_energy, scale=sigma)


"""
class APEC(AdditiveComponent):
    def __init__(self, name="apec"):
        super(APEC, self).__init__(name=name)

        ref = importlib.resources.files("jaxspec") / "tables/apec_tab.npz"
        with importlib.resources.as_file(ref) as path:
            files = np.load(path)

        self.kT_ref = files["kT_ref"]
        self.e_ref = np.nan_to_num(files["continuum_energy"], nan=1e6)
        self.c_ref = np.nan_to_num(files["continuum_emissivity"])
        self.pe_ref = np.nan_to_num(files["pseudo_energy"], nan=1e6)
        self.pc_ref = np.nan_to_num(files["pseudo_emissivity"])
        self.energy_lines = np.nan_to_num(files["lines_energy"], nan=1e6)  # .astype(np.float32))
        self.epsilon_lines = np.nan_to_num(files["lines_emissivity"])  # .astype(np.float32))
        self.element_lines = np.nan_to_num(files["lines_element"])  # .astype(np.int32))

        del files

        self.trace_elements = jnp.array([3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30]) - 1
        self.metals = np.array([6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28]) - 1  # Element number to python index
        self.metals_one_hot = np.zeros((30,))
        self.metals_one_hot[self.metals] = 1

    def interp_on_cubes(self, energy, energy_cube, continuum_cube):
        # Changer en loginterp
        # Interpoler juste sur les points qui ne sont pas tabulés
        # Ajouter l'info de la taille max des données (à resortir dans la routine qui trie les fichier apec)
        return jnp.vectorize(
            lambda ecube, ccube: jnp.interp(energy, ecube, ccube),
            signature="(k),(k)->()",
        )(energy_cube, continuum_cube)

    def reduction_with_elements(self, Z, energy, energy_cube, continuum_cube):
        return jnp.sum(
            self.interp_on_cubes(energy, energy_cube, continuum_cube) * jnp.where(self.metals_one_hot, Z, 1.0)[None, :],
            axis=-1,
        )

    def mono_fine_structure(self, e_low, e_high) -> (jax.Array, jax.Array):
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))
        kT = hk.get_parameter("kT", [], float, init=HaikuConstant(1))
        Z = hk.get_parameter("Z", [], float, init=HaikuConstant(1))

        idx = jnp.searchsorted(self.kT_ref, kT, side="left") - 1

        energy = jax_slice(self.energy_lines, idx, 2)
        epsilon = jax_slice(self.epsilon_lines, idx, 2)
        element = jax_slice(self.element_lines, idx, 2) - 1

        emissivity_in_bins = jnp.where((e_low < energy) & (energy < e_high), True, False) * epsilon
        flux_at_edges = jnp.nansum(
            jnp.where(jnp.isin(element, self.metals), Z, 1) * emissivity_in_bins,
            axis=-1,
        )  # Coeff for metallicity

        return (
            jnp.interp(kT, jax_slice(self.kT_ref, idx, 2), flux_at_edges) * 1e14 * norm,
            (e_low + e_high) / 2,
        )

    def emission_lines(self, e_low, e_high) -> (jax.Array, jax.Array):
        # Compute the fine structure lines with e_low and e_high as array, mapping the mono_fine_structure function
        # over the various axes of e_low and e_high

        return jnp.vectorize(self.mono_fine_structure)(e_low, e_high)

        # return jnp.zeros_like(e_low), (e_low + e_high)/2

    @partial(jnp.vectorize, excluded=(0,))
    def continuum(self, energy):
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))
        kT = hk.get_parameter("kT", [], float, init=HaikuConstant(1))
        Z = hk.get_parameter("Z", [], float, init=HaikuConstant(1))

        idx = jnp.searchsorted(self.kT_ref, kT, side="left") - 1  # index of left value

        continuum = jnp.interp(
            kT,
            jax_slice(self.kT_ref, idx, 2),
            self.reduction_with_elements(Z, energy, jax_slice(self.e_ref, idx, 2), jax_slice(self.c_ref, idx, 2)),
        )
        pseudo = jnp.interp(
            kT,
            jax_slice(self.kT_ref, idx, 2),
            self.reduction_with_elements(
                Z,
                energy,
                jax_slice(self.pe_ref, idx, 2),
                jax_slice(self.pc_ref, idx, 2),
            ),
        )

        return (continuum + pseudo) * 1e14 * norm
"""


class Cutoffpl(AdditiveComponent):
    r"""
    A power law model with high energy exponential cutoff

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha} \exp(-E/\beta)$$

    ??? abstract "Parameters"
        * $\alpha$ : Photon index of the power law $\left[\text{dimensionless}\right]$
        * $\beta$ : Folding energy of the exponential cutoff $\left[\text{keV}\right]$
        * $E_0$ : Reference energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
        * $K$ : Normalization at the reference energy (1 keV) $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        alpha = hk.get_parameter("alpha", [], float, init=HaikuConstant(1.3))
        beta = hk.get_parameter("beta", [], float, init=HaikuConstant(15))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1e-4))

        return norm * energy ** (-alpha) * jnp.exp(-energy / beta)


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


class Diskbb(AdditiveComponent):
    r"""
    `Diskpbb` with $p=0.75$

    ??? abstract "Parameters"
        * $T_{\text{in}}$ : Temperature at inner disk radius $\left[ \mathrm{keV}\right]$
        * $\text{norm}$ : $\cos i(r_{\text{in}}/d)^{2}$,
        where $r_{\text{in}}$ is "an apparent" inner disk radius $\left[\text{km}\right]$,
        $d$ the distance to the source in units of $10 \text{kpc}$, $i$ the angle of the disk ($i=0$ is face-on)
    """

    def continuum(self, energy):
        p = 0.75
        tout = 0.0
        tin = hk.get_parameter("Tin", [], float, init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        # Tout is set to 0 as it is evaluated at R=infinity
        def integrand(kT, e, tin, p):
            return e**2 * (kT / tin) ** (-2 / p - 1) / (jnp.exp(e / kT) - 1)

        integral = integrate_interval(integrand)
        return (
            norm
            * 2.78e-3
            * (0.75 / p)
            / tin
            * jnp.vectorize(lambda e: integral(tout, tin, e, tin, p))(energy)
        )


class Agauss(AdditiveComponent):
    r"""
    A simple Gaussian line profile in wavelength space.
    If the width is $\leq 0$ then it is treated as a delta function.
    The `Zagauss` variant computes a redshifted Gaussian.

    $$\mathcal{M}\left( \lambda \right) =
    \frac{K}{\sigma \sqrt{2 \pi}} \exp\left(\frac{-(\lambda - \lambda_L)^2}{2 \sigma^2}\right)$$

    ??? abstract "Parameters"
        * $\lambda_L$ : Wavelength of the line in Angström $\left[\unicode{x212B}\right]$
        * $\sigma$ : Width of the line width in Angström  $\left[\unicode{x212B}\right]$
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy) -> (jax.Array, jax.Array):
        hc = (astropy.constants.h * astropy.constants.c).to(u.angstrom * u.keV).value
        line_wavelength = hk.get_parameter("Lambda_l", [], float, init=HaikuConstant(hc))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(0.001))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))

        return norm * jsp.stats.norm.pdf(hc / energy, loc=line_wavelength, scale=sigma)


class Zagauss(AdditiveComponent):
    r"""
    A redshifted Gaussian line profile in wavelength space.
    If the width is $\leq 0$ then it is treated as a delta function.

    $$\mathcal{M}\left( \lambda \right) =
    \frac{K (1+z)}{\sigma \sqrt{2 \pi}} \exp\left(\frac{-(\lambda/(1+z) - \lambda_L)^2}{2 \sigma^2}\right)$$

    ??? abstract "Parameters"
        * $\lambda_L$ : Wavelength of the line in Angström $\left[\text{\AA}\right]$
        * $\sigma$ : Width of the line width in Angström $\left[\text{\AA}\right]$
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
        * $z$ : Redshift [dimensionless]
    """

    def continuum(self, energy) -> (jax.Array, jax.Array):
        hc = (astropy.constants.h * astropy.constants.c).to(u.angstrom * u.keV).value
        line_wavelength = hk.get_parameter("Lambda_l", [], float, init=HaikuConstant(hc))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(0.001))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))
        redshift = hk.get_parameter("redshift", [], float, init=HaikuConstant(0))

        return (
            norm
            * (1 + redshift)
            * jsp.stats.norm.pdf((hc / energy) / (1 + redshift), loc=line_wavelength, scale=sigma)
        )


class Zgauss(AdditiveComponent):
    r"""
    A redshifted Gaussian line profile. If the width is $\leq 0$ then it is treated as a delta function.

    $$\mathcal{M}\left( E \right) =
    \frac{K}{(1+z) \sigma \sqrt{2 \pi}}\exp\left(\frac{-(E(1+z)-E_L)^2}{2\sigma^2}\right)$$

    ??? abstract "Parameters"
        * $E_L$ : Energy of the line $\left[\text{keV}\right]$
        * $\sigma$ : Width of the line $\left[\text{keV}\right]$
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
        * $z$ : Redshift [dimensionless]
    """

    def continuum(self, energy) -> (jax.Array, jax.Array):
        line_energy = hk.get_parameter("E_l", [], float, init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], float, init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1))
        redshift = hk.get_parameter("redshift", [], float, init=HaikuConstant(0))

        return (norm / (1 + redshift)) * jsp.stats.norm.pdf(
            energy * (1 + redshift), loc=line_energy, scale=sigma
        )


class NSatmos(AdditiveComponent):
    r"""
    A neutron star atmosphere model based on the `NSATMOS` model from `XSPEC`. See [this page](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/node205.html)

    !!! warning
        The boundary case of $R_{\text{NS}} < 1.125 R_{\text{S}}$ is handled with a null flux instead of a constant value as in `XSPEC`.

    ??? abstract "Parameters"
        * $T_{eff}$ : Effective temperature at the surface in K (No redshift applied)
        * $M_{ns}$ : Mass of the NS in solar masses
        * $R_∞$ : Radius at infinity (modulated by gravitational effects) in km
        * $D$ : Distance to the neutron star in kpc
        * norm : fraction of the neutron star surface emitting
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        entry_table = Table.read(table_manager.fetch("nsatmosdata.fits"), 1)

        # Extract the table values. All this code could be summarize in two lines if we reformat the nsatmosdata.fits table
        self.tab_temperature = np.asarray(entry_table["TEMP"][0], dtype=float)  # Logarithmic value
        self.tab_gravity = np.asarray(entry_table["GRAVITY"][0], dtype=float)  # Logarithmic value
        self.tab_mucrit = np.asarray(entry_table["MUCRIT"][0], dtype=float)
        self.tab_energy = np.asarray(entry_table["ENERGY"][0], dtype=float)
        self.tab_flux_flat = Table.read(table_manager.fetch("nsatmosdata.fits"), 2)["FLUX"]

        tab_flux = np.empty(
            (
                self.tab_temperature.size,
                self.tab_gravity.size,
                self.tab_mucrit.size,
                self.tab_energy.size,
            )
        )

        for i in range(len(self.tab_temperature)):
            for j in range(len(self.tab_gravity)):
                for k in range(len(self.tab_mucrit)):
                    tab_flux[i, j, k] = np.array(
                        self.tab_flux_flat[
                            i * len(self.tab_gravity) * len(self.tab_mucrit)
                            + j * len(self.tab_mucrit)
                            + k
                        ]
                    )

        self.tab_flux = np.asarray(tab_flux, dtype=float)

    def interp_flux_func(self, temperature_log, gravity_log, mu):
        # Interpolate the tables to get the flux on the tabulated energy grid

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

    def continuum(self, energy):
        temp_log = hk.get_parameter(
            "Tinf", [], float, init=HaikuConstant(6.0)
        )  # log10 of temperature in Kelvin

        #         'Tinf': temp_log, # 5 to 6.5
        #         'M': mass, # 0.5 to 3
        #         'Rns': radius, # 5 to 30

        mass = hk.get_parameter("M", [], float, init=HaikuConstant(1.4))
        radius = hk.get_parameter("Rns", [], float, init=HaikuConstant(10.0))
        distance = hk.get_parameter("dns", [], float, init=HaikuConstant(10.0))
        norm = hk.get_parameter("norm", [], float, init=HaikuConstant(1.0))

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

        flux = jax.jit(self.interp_flux_func)(temp_log, gravity_log, cmu)

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
