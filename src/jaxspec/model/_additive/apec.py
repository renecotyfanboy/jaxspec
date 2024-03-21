import h5netcdf
import jax.numpy as jnp
import jax
import haiku as hk
import importlib.resources
from jax import lax
from jax.lax import scan, fori_loop
from ...util.abundance import abundance_table
from haiku.initializers import Constant as HaikuConstant
from ..abc import AdditiveComponent


@jax.jit
def lerp(x, x0, x1, y0, y1):
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)


@jax.jit
def interp_along_array(energy_low, energy_high, energy_ref, continuum_ref, end_index):
    """
    Sorry for the boilerplate here, but be sure that it works !
    """

    def body_func(index, value):
        integrated_flux, previous_energy, previous_continuum = value
        current_energy, current_continuum = energy_ref[index], continuum_ref[index]

        # 5 cases
        # - Neither current and previous energies are within the integral limits -> nothing is added to the integrated flux
        # - The left limit of the integral is between the current and previous energy -> previous energy is set to the limit, previous continuum is interpolated, and then added to the integrated flux
        # - The right limit of the integral is between the current and previous energy -> current energy is set to the limit, current continuum is interpolated, and then added to the integrated flux
        # - Both current and previous energies are within the integral limits -> add to the integrated flux
        # - Within

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

    integrated_flux, _, _ = fori_loop(0, end_index, body_func, (0.0, 0.0, 0.0))

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
def interp_flux_elements(energy, energy_ref, continuum_ref, abundances, end_indexes):
    """
    Iterate through an array of shape (abundance, energy_ref) and compute the flux between the bins defined by energy and weight the flux depending of the abundance of each element
    """

    flux = jnp.zeros_like(energy[:-1])

    def scanned_func(flux, unpack):
        energy_ref, continuum_ref, abun, end_index = unpack
        element_flux = abun * interp_flux(energy, energy_ref, continuum_ref, end_index)
        flux = flux + element_flux

        return flux, None

    flux, _ = scan(scanned_func, flux, (energy_ref, continuum_ref, abundances, end_indexes))

    return flux


@jax.jit
def get_lines_between_energies(energy_low, energy_high, line_energy, line_emissivity, line_element, abundances, end_index):
    def body_func(index, value):
        energy = line_energy[index]
        emissivity = line_emissivity[index]
        element = line_element[index].astype(int)

        line_in_bin = (energy_low < energy) * (energy < energy_high)

        flux = lax.select(line_in_bin, emissivity * abundances[element], 0.0)

        return value + flux

    return fori_loop(0, end_index, body_func, 0.0)


@jax.jit
def get_lines_contribution(energy, line_energy, line_emissivity, line_element, abundances, end_index):
    def scanned_func(_, unpack):
        energy_low, energy_high = unpack

        flux = get_lines_between_energies(
            energy_low, energy_high, line_energy, line_emissivity, line_element, abundances, end_index
        )

        return _, flux

    _, flux = scan(scanned_func, 0, (energy[:-1], energy[1:]))

    return flux


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

        end_index_continuum = jnp.sum(~jnp.isnan(continuum_energy_array), axis=-1)
        end_index_pseudo = jnp.sum(~jnp.isnan(pseudo_energy_array), axis=-1)
        end_index_lines = jnp.sum(~jnp.isnan(line_energy_array), axis=-1)

        abund = jnp.ones(30)
        metals = jnp.asarray([6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 26, 28], dtype=int) - 1
        trace_elements = jnp.asarray([3, 4, 5, 9, 11, 15, 17, 19, 21, 22, 23, 24, 25, 27, 29, 30], dtype=int) - 1

        misc_arrays = (temperature, metals, trace_elements, abund)
        end_indexes = (end_index_continuum, end_index_pseudo, end_index_lines)
        line_data = (line_energy_array, line_element_array, line_emissivity_array)
        continuum_data = (continuum_energy_array, continuum_emissivity_array)
        pseudo_data = (pseudo_energy_array, pseudo_emissivity_array)

        return misc_arrays, end_indexes, line_data, continuum_data, pseudo_data


class Vvapec(AdditiveComponent):
    def get_parameters(self):
        T = hk.get_parameter("T", [], init=HaikuConstant(6.5))
        abund = jnp.zeros((30,))

        for i, element in enumerate(abundance_table["Element"]):
            abund = abund.at[i].set(hk.get_parameter(element, [], init=HaikuConstant(1.0)))

        z = hk.get_parameter("z", [], init=HaikuConstant(0.0))
        norm = hk.get_parameter("norm", [], init=HaikuConstant(1.0))

        return T, z, norm, abund

    def emission_lines(self, e_low, e_high):
        misc_arrays, end_indexes, line_data, continuum_data, pseudo_data = apec_table_getter()
        temperature, metals, trace_elements, abund = misc_arrays
        end_index_continuum, end_index_pseudo, end_index_lines = end_indexes
        line_energy_array, line_element_array, line_emissivity_array = line_data
        continuum_energy_array, continuum_emissivity_array = continuum_data
        pseudo_energy_array, pseudo_emissivity_array = pseudo_data

        energy = jnp.hstack([e_low, e_high[-1]])
        T, z, norm, abund = self.get_parameters()
        abundances = abund  # self.abund.at[self.metals].set(Z)
        idx = jnp.searchsorted(temperature, T) - 1
        T_low, T_high = temperature[idx], temperature[idx + 1]
        energy = energy / (1 + z)

        continuum_low = interp_flux_elements(
            energy, continuum_energy_array[idx], continuum_emissivity_array[idx], abundances, end_index_continuum[idx]
        )

        continuum_high = interp_flux_elements(
            energy, continuum_energy_array[idx + 1], continuum_emissivity_array[idx + 1], abundances, end_index_continuum[idx + 1]
        )

        pcontinuum_low = interp_flux_elements(
            energy, pseudo_energy_array[idx], pseudo_emissivity_array[idx], abundances, end_index_pseudo[idx]
        )

        pcontinuum_high = interp_flux_elements(
            energy, pseudo_energy_array[idx + 1], pseudo_emissivity_array[idx + 1], abundances, end_index_pseudo[idx + 1]
        )

        line_low = get_lines_contribution(
            energy, line_energy_array[idx], line_emissivity_array[idx], line_element_array[idx], abundances, end_index_lines[idx]
        )

        line_high = get_lines_contribution(
            energy, line_energy_array[idx], line_emissivity_array[idx], line_element_array[idx], abundances, end_index_lines[idx]
        )

        interp_cont = lerp(
            T, T_low, T_high, continuum_low + pcontinuum_low + line_low, continuum_high + pcontinuum_high + line_high
        )

        return interp_cont * norm * 1e14, (e_low + e_high) / 2
