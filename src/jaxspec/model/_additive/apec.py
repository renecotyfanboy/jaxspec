import warnings

from typing import Literal

import astropy.units as u
import haiku as hk
import jax
import jax.numpy as jnp

from astropy.constants import c, m_p
from haiku.initializers import Constant as HaikuConstant
from jax import lax
from jax.lax import fori_loop, scan
from jax.scipy.stats import norm as gaussian

from ...util.abundance import abundance_table, element_data
from ..abc import AdditiveComponent
from .apec_loaders import get_continuum, get_lines, get_pseudo, get_temperature


@jax.jit
def lerp(x, x0, x1, y0, y1):
    """
    Linear interpolation routine
    Return y(x) =  (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
    """
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)


@jax.jit
def interp_and_integrate(energy_low, energy_high, energy_ref, continuum_ref, end_index):
    """
    This function interpolate & integrate the values of a tabulated reference continuum between two energy limits
    Sorry for the boilerplate here, but be sure that it works !

    Parameters:
        energy_low: lower limit of the integral
        energy_high: upper limit of the integral
        energy_ref: energy grid of the reference continuum
        continuum_ref: continuum values evaluated at energy_ref

    """
    energy_ref = jnp.where(jnp.arange(energy_ref.shape[0]) < end_index, energy_ref, jnp.nan)
    start_index = jnp.searchsorted(energy_ref, energy_low, side="left") - 1
    end_index = jnp.searchsorted(energy_ref, energy_high, side="left") + 1

    def body_func(index, value):
        integrated_flux, previous_energy, previous_continuum = value
        current_energy, current_continuum = energy_ref[index], continuum_ref[index]

        # 5 cases
        # Neither current and previous energies are within the integral limits > nothing is added to the integrated flux
        # The left limit of the integral is between the current and previous energy > previous energy is set to the limit, previous continuum is interpolated, and then added to the integrated flux
        # The right limit of the integral is between the current and previous energy > current energy is set to the limit, current continuum is interpolated, and then added to the integrated flux
        # Both current and previous energies are within the integral limits -> add to the integrated flux
        # Within

        current_energy_is_between = (energy_low <= current_energy) * (current_energy < energy_high)
        previous_energy_is_between = (energy_low <= previous_energy) * (
            previous_energy < energy_high
        )
        energies_within_bins = (previous_energy <= energy_low) * (energy_high < current_energy)

        case = (
            (1 - previous_energy_is_between) * current_energy_is_between * 1
            + previous_energy_is_between * (1 - current_energy_is_between) * 2
            + (previous_energy_is_between * current_energy_is_between) * 3
            + energies_within_bins * 4
        )

        term_to_add = lax.switch(
            case,
            [
                lambda pe, pc, ce, cc, el, er: 0.0,  # 1
                lambda pe, pc, ce, cc, el, er: (cc + lerp(el, pe, ce, pc, cc)) * (ce - el) / 2,  # 2
                lambda pe, pc, ce, cc, el, er: (pc + lerp(er, pe, ce, pc, cc)) * (er - pe) / 2,  # 3
                lambda pe, pc, ce, cc, el, er: (pc + cc) * (ce - pe) / 2,  # 4
                lambda pe, pc, ce, cc, el, er: (lerp(el, pe, ce, pc, cc) + lerp(er, pe, ce, pc, cc))
                * (er - el)
                / 2,
                # 5
            ],
            previous_energy,
            previous_continuum,
            current_energy,
            current_continuum,
            energy_low,
            energy_high,
        )

        return integrated_flux + term_to_add, current_energy, current_continuum

    integrated_flux, _, _ = fori_loop(start_index, end_index, body_func, (0.0, 0.0, 0.0))

    return integrated_flux


@jax.jit
def interp(e_low, e_high, energy_ref, continuum_ref, end_index):
    energy_ref = jnp.where(jnp.arange(energy_ref.shape[0]) < end_index, energy_ref, jnp.nan)

    return (
        jnp.interp(e_high, energy_ref, continuum_ref) - jnp.interp(e_low, energy_ref, continuum_ref)
    ) / (e_high - e_low)


@jax.jit
def interp_flux(energy, energy_ref, continuum_ref, end_index):
    """
    Iterate through an array of shape (energy_ref,) and compute the flux between the bins defined by energy
    """

    def scanned_func(carry, unpack):
        e_low, e_high = unpack
        continuum = interp_and_integrate(e_low, e_high, energy_ref, continuum_ref, end_index)

        return carry, continuum

    _, continuum = scan(scanned_func, 0.0, (energy[:-1], energy[1:]))

    return continuum


@jax.jit
def interp_flux_elements(energy_ref, continuum_ref, end_index, energy, abundances):
    """
    Iterate through an array of shape (abundance, energy_ref) and compute the flux between the bins defined by energy
    and weight the flux depending on the abundance of each element
    """

    def scanned_func(_, unpack):
        energy_ref, continuum_ref, end_idx = unpack
        element_flux = interp_flux(energy, energy_ref, continuum_ref, end_idx)

        return _, element_flux

    _, flux = scan(scanned_func, 0.0, (energy_ref, continuum_ref, end_index))

    return abundances @ flux


@jax.jit
def get_lines_contribution_broadening(
    line_energy, line_element, line_emissivity, end_index, energy, abundances, total_broadening
):
    def body_func(i, flux):
        # Notice the -1 in line element to match the 0-based indexing
        l_energy, l_emissivity, l_element = line_energy[i], line_emissivity[i], line_element[i] - 1
        broadening = l_energy * total_broadening[l_element]
        l_flux = gaussian.cdf(energy[1:], l_energy, broadening) - gaussian.cdf(
            energy[:-1], l_energy, broadening
        )
        l_flux = l_flux * l_emissivity * abundances[l_element]

        return flux + l_flux

    return fori_loop(0, end_index, body_func, jnp.zeros_like(energy[:-1]))


@jax.jit
def continuum_func(energy, kT, abundances):
    idx, kT_low, kT_high = get_temperature(kT)
    continuum_low = interp_flux_elements(*get_continuum(idx), energy, abundances)
    continuum_high = interp_flux_elements(*get_continuum(idx + 1), energy, abundances)

    return lerp(kT, kT_low, kT_high, continuum_low, continuum_high)


@jax.jit
def pseudo_func(energy, kT, abundances):
    idx, kT_low, kT_high = get_temperature(kT)
    continuum_low = interp_flux_elements(*get_pseudo(idx), energy, abundances)
    continuum_high = interp_flux_elements(*get_pseudo(idx + 1), energy, abundances)

    return lerp(kT, kT_low, kT_high, continuum_low, continuum_high)


# @jax.custom_jvp
@jax.jit
def lines_func(energy, kT, abundances, broadening):
    idx, kT_low, kT_high = get_temperature(kT)
    line_low = get_lines_contribution_broadening(*get_lines(idx), energy, abundances, broadening)
    line_high = get_lines_contribution_broadening(
        *get_lines(idx + 1), energy, abundances, broadening
    )

    return lerp(kT, kT_low, kT_high, line_low, line_high)


class APEC(AdditiveComponent):
    """
    APEC model implementation in pure JAX for X-ray spectral fitting.

    !!! warning
        This implementation is optimised for the CPU, it shows poor performance on the GPU.
    """

    def __init__(
        self,
        continuum: bool = True,
        pseudo: bool = True,
        lines: bool = True,
        thermal_broadening: bool = True,
        turbulent_broadening: bool = True,
        variant: Literal["none", "v", "vv"] = "none",
        abundance_table: Literal[
            "angr", "aspl", "feld", "aneb", "grsa", "wilm", "lodd", "lgpp", "lgps"
        ] = "angr",
        trace_abundance: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        warnings.warn("Be aware that this APEC implementation is not meant to be used yet")

        self.atomic_weights = jnp.asarray(element_data["atomic_weight"].to_numpy())

        self.abundance_table = abundance_table
        self.thermal_broadening = thermal_broadening
        self.turbulent_broadening = turbulent_broadening
        self.continuum_to_compute = continuum
        self.pseudo_to_compute = pseudo
        self.lines_to_compute = lines
        self.trace_abundance = trace_abundance
        self.variant = variant

    def get_thermal_broadening(self):
        r"""
        Compute the thermal broadening $\sigma_T$ for each element using :

        $$ \frac{\sigma_T}{E_{\text{line}}} = \frac{1}{c}\sqrt{\frac{k_{B} T}{A m_p}}$$

        where $E_{\text{line}}$ is the energy of the line, $c$ is the speed of light, $k_{B}$ is the Boltzmann constant,
        $T$ is the temperature, $A$ is the atomic weight of the element and $m_p$ is the proton mass.
        """

        if self.thermal_broadening:
            kT = hk.get_parameter("kT", [], init=HaikuConstant(6.5))
            factor = 1 / c * (1 / m_p) ** (1 / 2)
            factor = factor.to(u.keV ** (-1 / 2)).value

            # Multiply this factor by Line_Energy * sqrt(kT/A) to get the broadening for a line
            # This return value must be multiplied by the energy of the line to get actual broadening
            return factor * jnp.sqrt(kT / self.atomic_weights)

        else:
            return jnp.zeros((30,))

    def get_turbulent_broadening(self):
        r"""
        Return the turbulent broadening  using :

        $$\frac{\sigma_\text{turb}}{E_{\text{line}}} = \frac{\sigma_{v ~ ||}}{c}$$

        where $\sigma_{v ~ ||}$ is the velocity dispersion along the line of sight in km/s.
        """
        if self.turbulent_broadening:
            # This return value must be multiplied by the energy of the line to get actual broadening
            return (
                hk.get_parameter("Velocity", [], init=HaikuConstant(100.0)) / c.to(u.km / u.s).value
            )
        else:
            return 0.0

    def get_parameters(self):
        none_elements = ["C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar", "Ca", "Fe", "Ni"]
        v_elements = ["He", "C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar", "Ca", "Fe", "Ni"]
        trace_elements = (
            jnp.asarray([3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30], dtype=int) - 1
        )

        # Set abundances of trace element (will be overwritten in the vv case)
        abund = jnp.ones((30,)).at[trace_elements].multiply(self.trace_abundance)

        if self.variant == "vv":
            for i, element in enumerate(abundance_table["Element"]):
                if element != "H":
                    abund = abund.at[i].set(hk.get_parameter(element, [], init=HaikuConstant(1.0)))

        elif self.variant == "v":
            for i, element in enumerate(abundance_table["Element"]):
                if element != "H" and element in v_elements:
                    abund = abund.at[i].set(hk.get_parameter(element, [], init=HaikuConstant(1.0)))

        else:
            Z = hk.get_parameter("Abundance", [], init=HaikuConstant(1.0))
            for i, element in enumerate(abundance_table["Element"]):
                if element != "H" and element in none_elements:
                    abund = abund.at[i].set(Z)

        if abund != "angr":
            abund = abund * jnp.asarray(
                abundance_table[self.abundance_table] / abundance_table["angr"]
            )

        # Set the temperature, redshift, normalisation
        kT = hk.get_parameter("kT", [], init=HaikuConstant(6.5))
        z = hk.get_parameter("Redshift", [], init=HaikuConstant(0.0))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1.0))

        return kT, z, norm, abund

    def emission_lines(self, e_low, e_high):
        # Get the parameters and extract the relevant data
        energy = jnp.hstack([e_low, e_high[-1]])
        kT, z, norm, abundances = self.get_parameters()
        total_broadening = jnp.hypot(self.get_thermal_broadening(), self.get_turbulent_broadening())
        energy = energy * (1 + z)

        continuum = continuum_func(energy, kT, abundances) if self.continuum_to_compute else 0.0
        pseudo_continuum = pseudo_func(energy, kT, abundances) if self.pseudo_to_compute else 0.0
        lines = (
            lines_func(energy, kT, abundances, total_broadening) if self.lines_to_compute else 0.0
        )

        return (continuum + pseudo_continuum + lines) * norm * 1e14 / (1 + z), (e_low + e_high) / 2
