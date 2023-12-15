from __future__ import annotations

from abc import ABC

import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import importlib.resources
from jax.lax import dynamic_slice_in_dim as jax_slice
from functools import partial
from .abc import ModelComponent
from haiku.initializers import Constant as HaikuConstant


class AdditiveComponent(ModelComponent, ABC):
    type = "additive"

    def continuum(self, energy):
        """
        Method for computing the continuum associated to the model.
        By default, this is set to 0, which means that the model has no continuum.
        This should be overloaded by the user if the model has a continuum.
        """

        return jnp.zeros_like(energy)

    def emission_lines(self, e_min, e_max) -> (jax.Array, jax.Array):
        """
        Method for computing the fine structure of an additive model between two energies.
        By default, this is set to 0, which means that the model has no emission lines.
        This should be overloaded by the user if the model has a fine structure.
        """

        return jnp.zeros_like(e_min), (e_min + e_max) / 2

    '''
    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on
        double exponential quadrature for finite intervals to compute an approximation
        of the integral of a model.

        references
        ----------
        * $Takahasi and Mori (1974) <https://ems.press/journals/prims/articles/2686>$_
        * $Mori and Sugihara (2001) <https://doi.org/10.1016/S0377-0427(00)00501-X>$_
        * $Tanh-sinh quadrature <https://en.wikipedia.org/wiki/Tanh-sinh_quadrature>$_ from Wikipedia

        """

        t = jnp.linspace(-4, 4, 71) # The number of points used is hardcoded and this is not ideal
        # Quadrature nodes as defined in reference
        phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
        # Change of variable to turn the integral from E_min to E_max into an integral from -1 to 1
        x = (e_max - e_min) / 2 * phi + (e_max + e_min) / 2
        dx = (e_max - e_min) / 2 * dphi

        return jnp.trapz(self(x) * dx, x=t)
    '''


class Powerlaw(AdditiveComponent):
    r"""
    A power law model

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha}$$

    ??? abstract "Parameters"
        * $\alpha$ : Photon index of the power law $\left[\text{dimensionless}\right]$
        * $E_0$ : Reference energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
        * $K$ : Normalization at 1 keV $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        alpha = hk.get_parameter("alpha", [], init=HaikuConstant(1.3))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1e-4))

        return norm * energy ** (-alpha)


class AdditiveConstant(AdditiveComponent):
    r"""
    A constant model

    $$\mathcal{M}\left( E \right) = K$$

    ??? abstract "Parameters"
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

        return norm * jnp.ones_like(energy)

    def primitive(self, energy):
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

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

    def emission_lines(self, e_min, e_max) -> (jax.Array, jax.Array):
        # return the primitive of a lorentzian
        line_energy = hk.get_parameter("E_l", [], init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

        # This is AI generated for tests I should double check this at some point
        def primitive(energy):
            return norm * (sigma / (2 * jnp.pi)) * jnp.arctan((energy - line_energy) / (sigma / 2))

        return primitive(e_max) - primitive(e_min), (e_min + e_max) / 2

    def continuum(self, energy):
        hk.get_parameter("E_l", [], init=HaikuConstant(1))
        hk.get_parameter("sigma", [], init=HaikuConstant(1))
        hk.get_parameter("norm", [], init=HaikuConstant(1))

        return jnp.zeros_like(energy)


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
        * $K$ : Normalization at the pivot energy $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
        * $E_{\text{Pivot}}$ : Pivot energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
    """

    def continuum(self, energy):
        a = hk.get_parameter("a", [], init=HaikuConstant(11 / 3))
        b = hk.get_parameter("b", [], init=HaikuConstant(0.2))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

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
        kT = hk.get_parameter("kT", [], init=HaikuConstant(11 / 3))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

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
        kT = hk.get_parameter("kT", [], init=HaikuConstant(11 / 3))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

        return norm * 1.0344e-3 * energy**2 / (jnp.exp(energy / kT) - 1)


class Gauss(AdditiveComponent):
    r"""
    A Gaussian line profile

    $$\mathcal{M}\left( E \right) = \frac{K}{\sqrt{2\pi\sigma^2}}\exp\left(\frac{-(E-E_L)^2}{2\sigma^2}\right)$$

    The primitive is defined as :

    $$
    \int \mathcal{M}\left( E \right) \text{d}E =
    \frac{K}{2}\left( 1+\text{erf} \frac{(E-E_L)}{\sqrt{2}\sigma} \right)
    $$

    ??? abstract "Parameters"
        * $E_L$ : Energy of the line $\left[\text{keV}\right]$
        * $\sigma$ : Width of the line $\left[\text{keV}\right]$
        * $K$ : Normalization $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy) -> (jax.Array, jax.Array):
        line_energy = hk.get_parameter("E_l", [], init=HaikuConstant(1))
        sigma = hk.get_parameter("sigma", [], init=HaikuConstant(1))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))

        return norm * jsp.stats.norm.pdf(energy, loc=line_energy, scale=sigma)


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
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))
        kT = hk.get_parameter("kT", [], init=HaikuConstant(1))
        Z = hk.get_parameter("Z", [], init=HaikuConstant(1))

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
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1))
        kT = hk.get_parameter("kT", [], init=HaikuConstant(1))
        Z = hk.get_parameter("Z", [], init=HaikuConstant(1))

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


class Cutoffpl(AdditiveComponent):
    r"""
    A power law model with high energy exponential cutoff

    $$\mathcal{M}\left( E \right) = K \left( \frac{E}{E_0} \right)^{-\alpha} \exp(-E/\beta)$$

    ??? abstract "Parameters"
        * $\alpha$ : Photon index of the power law $\left[\text{dimensionless}\right]$
        * $\beta$ : Folding energy of the exponential cutoff $\left[\text{keV}\right]$
        * $E_0$ : Reference energy fixed at 1 keV $\left[ \mathrm{keV}\right]$
        * $K$ : Normalization at 1 keV $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$
    """

    def continuum(self, energy):
        alpha = hk.get_parameter("alpha", [], init=HaikuConstant(1.3))
        beta = hk.get_parameter("beta", [], init=HaikuConstant(15))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1e-4))

        return norm * energy ** (-alpha) * jnp.exp(-energy / beta)
