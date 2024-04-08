import h5netcdf
import jax.numpy as jnp
import jax
import haiku as hk
import importlib.resources
import astropy.units as u
from jax import lax
from jax.lax import scan, fori_loop
from jax.scipy.stats import norm as gaussian
from typing import Literal
from ...util.abundance import abundance_table, element_data
from haiku.initializers import Constant as HaikuConstant
from astropy.constants import c, m_p
from ..abc import AdditiveComponent


@jax.jit
def lerp(x, x0, x1, y0, y1):
    """
    Linear interpolation routine
    Return y(x) =  (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
    """
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)


@jax.jit
def interp_along_array(energy_low, energy_high, energy_ref, continuum_ref, end_index):
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
        previous_energy_is_between = (energy_low <= previous_energy) * (previous_energy < energy_high)
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
                lambda pe, pc, ce, cc, el, er: (lerp(el, pe, ce, pc, cc) + lerp(er, pe, ce, pc, cc)) * (er - el) / 2,
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
def interp_flux(energy, energy_ref, continuum_ref, end_index):
    """
    Iterate through an array of shape (energy_ref,) and compute the flux between the bins defined by energy
    """

    def scanned_func(carry, unpack):
        e_low, e_high = unpack
        continuum = interp_along_array(e_low, e_high, energy_ref, continuum_ref, end_index)

        return carry, continuum

    _, continuum = scan(scanned_func, 0.0, (energy[:-1], energy[1:]))

    return continuum


@jax.jit
def interp_flux_elements(energy, energy_ref, continuum_ref, end_index, abundances):
    """
    Iterate through an array of shape (abundance, energy_ref) and compute the flux between the bins defined by energy
    and weight the flux depending on the abundance of each element
    """

    flux = jnp.zeros_like(energy[:-1])

    def scanned_func(flux, unpack):
        energy_ref, continuum_ref, abun, end_idx = unpack
        element_flux = abun * interp_flux(energy, energy_ref, continuum_ref, end_idx)
        flux = flux + element_flux

        return flux, None

    flux, _ = scan(scanned_func, flux, (energy_ref, continuum_ref, abundances, end_index))

    return flux


@jax.jit
def get_lines_contribution_broadening(
    energy, line_energy, line_emissivity, line_element, abundances, end_index, total_broadening
):
    def body_func(i, flux):
        # Notice the -1 in line element to match the 0-based indexing
        l_energy, l_emissivity, l_element = line_energy[i], line_emissivity[i], line_element[i] - 1
        broadening = l_energy * total_broadening[l_element]
        l_flux = gaussian.cdf(energy[1:], l_energy, broadening) - gaussian.cdf(energy[:-1], l_energy, broadening)
        l_flux = l_flux * l_emissivity * abundances[l_element]

        return flux + l_flux

    return fori_loop(0, end_index, body_func, jnp.zeros_like(energy[:-1]))


@jax.jit
def apec_table_getter():
    with jax.ensure_compile_time_eval():
        ref = importlib.resources.files("jaxspec") / "tables/apec.nc"

        with h5netcdf.File(ref, "r") as f:
            temperature = jnp.asarray(f["/temperature"])
            line_energy_array = jnp.asarray(f["/line_energy"])
            line_element_array = jnp.asarray(f["/line_element"])
            line_emissivity_array = jnp.asarray(f["/line_emissivity"])
            continuum_energy_array = jnp.asarray(f["/continuum_energy"])
            continuum_emissivity_array = jnp.asarray(f["/continuum_emissivity"])
            pseudo_energy_array = jnp.asarray(f["/pseudo_energy"])
            pseudo_emissivity_array = jnp.asarray(f["/pseudo_emissivity"])
            end_index_continuum = jnp.asarray(f["/continuum_end_index"])
            end_index_pseudo = jnp.asarray(f["/pseudo_end_index"])
            end_index_lines = jnp.asarray(f["/line_end_index"])

        metals = jnp.asarray([6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28], dtype=int) - 1
        trace_elements = jnp.asarray([3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30], dtype=int) - 1

        misc_arrays = (temperature, metals, trace_elements)
        end_indexes = (end_index_continuum, end_index_pseudo, end_index_lines)
        line_data = (line_energy_array, line_element_array, line_emissivity_array)
        continuum_data = (continuum_energy_array, continuum_emissivity_array)
        pseudo_data = (pseudo_energy_array, pseudo_emissivity_array)

        return misc_arrays, end_indexes, line_data, continuum_data, pseudo_data


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
        abundance_table: Literal["angr", "aspl", "feld", "aneb", "grsa", "wilm", "lodd", "lgpp", "lgps"] = "angr",
        trace_abundance: float = 1.0,
        **kwargs,
    ):
        super(APEC, self).__init__(**kwargs)

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
            return hk.get_parameter("Velocity", [], init=HaikuConstant(100.0)) / c.to(u.km / u.s).value
        else:
            return 0.0

    def get_parameters(self):
        none_elements = ["C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar", "Ca", "Fe", "Ni"]
        v_elements = ["He", "C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar", "Ca", "Fe", "Ni"]
        trace_elements = jnp.asarray([3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30], dtype=int) - 1
        abund = jnp.ones((30,))

        # Set abundances of trace element (will be overwritten in the vv case)
        abund = abund.at[trace_elements].multiply(self.trace_abundance)

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
            abund = abund * jnp.asarray(abundance_table[self.abundance_table] / abundance_table["angr"])

        # Set the temperature, redshift, normalisation
        T = hk.get_parameter("kT", [], init=HaikuConstant(6.5))
        z = hk.get_parameter("Redshift", [], init=HaikuConstant(0.0))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1.0))

        return T, z, norm, abund

    def emission_lines(self, e_low, e_high):
        # Unpack the data
        misc_arrays, end_indexes, line_data, continuum_data, pseudo_data = apec_table_getter()
        temperature, metals, trace_elements = misc_arrays
        end_index_continuum, end_index_pseudo, end_index_lines = end_indexes
        line_energy_array, line_element_array, line_emissivity_array = line_data
        continuum_energy_array, continuum_emissivity_array = continuum_data
        pseudo_energy_array, pseudo_emissivity_array = pseudo_data

        # Get the parameters and extract the relevant data
        energy = jnp.hstack([e_low, e_high[-1]])
        T, z, norm, abundances = self.get_parameters()
        total_broadening = jnp.hypot(self.get_thermal_broadening(), self.get_turbulent_broadening())
        idx = jnp.searchsorted(temperature, T) - 1
        T_low, T_high = temperature[idx], temperature[idx + 1]
        energy = energy * (1 + z)

        if self.continuum_to_compute:
            continuum_low = interp_flux_elements(
                energy, continuum_energy_array[idx], continuum_emissivity_array[idx], end_index_continuum[idx], abundances
            )

            continuum_high = interp_flux_elements(
                energy,
                continuum_energy_array[idx + 1],
                continuum_emissivity_array[idx + 1],
                end_index_continuum[idx + 1],
                abundances,
            )

        else:
            continuum_low = 0.0
            continuum_high = 0.0

        if self.pseudo_to_compute:
            pcontinuum_low = interp_flux_elements(
                energy, pseudo_energy_array[idx], pseudo_emissivity_array[idx], end_index_pseudo[idx], abundances
            )

            pcontinuum_high = interp_flux_elements(
                energy, pseudo_energy_array[idx + 1], pseudo_emissivity_array[idx + 1], end_index_pseudo[idx + 1], abundances
            )

        else:
            pcontinuum_low = 0.0
            pcontinuum_high = 0.0

        if self.lines_to_compute:
            line_low = get_lines_contribution_broadening(
                energy,
                line_energy_array[idx],
                line_emissivity_array[idx],
                line_element_array[idx],
                abundances,
                end_index_lines[idx],
                total_broadening,
            )

            line_high = get_lines_contribution_broadening(
                energy,
                line_energy_array[idx + 1],
                line_emissivity_array[idx + 1],
                line_element_array[idx + 1],
                abundances,
                end_index_lines[idx + 1],
                total_broadening,
            )

        else:
            line_low = 0.0
            line_high = 0.0

        interp_cont = lerp(
            T, T_low, T_high, continuum_low + pcontinuum_low + line_low, continuum_high + pcontinuum_high + line_high
        )

        return interp_cont * norm * 1e14 / (1 + z), (e_low + e_high) / 2
