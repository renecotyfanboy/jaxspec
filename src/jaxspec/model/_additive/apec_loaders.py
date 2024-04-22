""" This module contains the functions that load the APEC tables from the HDF5 file. They are implemented as JAX
pure callback to enable reading data from the files without saturating the memory. """

import jax
import jax.numpy as jnp
import numpy as np
import importlib.resources
import xarray as xr


apec_file = xr.open_dataset(importlib.resources.files("jaxspec") / "tables/apec.nc", engine="h5netcdf")


def temperature_table_getter(kT):
    idx = np.searchsorted(apec_file.temperature, kT) - 1

    if idx > len(apec_file.temperature) - 2:
        return idx, 0.0, 0.0
    else:
        return idx, float(apec_file.temperature[idx]), float(apec_file.temperature[idx + 1])


def continuum_table_getter(idx):
    if idx > len(apec_file.temperature) - 2:
        continuum_energy_array = jnp.zeros_like(apec_file.continuum_energy[0])
        continuum_emissivity_array = jnp.zeros_like(apec_file.continuum_emissivity[0])
        end_index_continuum = jnp.zeros_like(apec_file.continuum_end_index[0])

    else:
        continuum_energy_array = jnp.asarray(apec_file.continuum_energy[idx])
        continuum_emissivity_array = jnp.asarray(apec_file.continuum_emissivity[idx])
        end_index_continuum = jnp.asarray(apec_file.continuum_end_index[idx])

    return continuum_energy_array, continuum_emissivity_array, end_index_continuum


def pseudo_table_getter(idx):
    if idx > len(apec_file.temperature) - 2:
        pseudo_energy_array = jnp.zeros_like(apec_file.pseudo_energy[0])
        pseudo_emissivity_array = jnp.zeros_like(apec_file.pseudo_emissivity[0])
        end_index_pseudo = jnp.zeros_like(apec_file.pseudo_end_index[0])

    else:
        pseudo_energy_array = jnp.asarray(apec_file.pseudo_energy[idx])
        pseudo_emissivity_array = jnp.asarray(apec_file.pseudo_emissivity[idx])
        end_index_pseudo = jnp.asarray(apec_file.pseudo_end_index[idx])

    return pseudo_energy_array, pseudo_emissivity_array, end_index_pseudo


def lines_table_getter(idx):
    if idx > len(apec_file.temperature) - 2:
        line_energy_array = jnp.zeros_like(apec_file.line_energy[0])
        line_element_array = jnp.zeros_like(apec_file.line_element[0])
        line_emissivity_array = jnp.zeros_like(apec_file.line_emissivity[0])
        end_index_lines = jnp.zeros_like(apec_file.line_end_index[0])

    else:
        line_energy_array = jnp.asarray(apec_file.line_energy[idx])
        line_element_array = jnp.asarray(apec_file.line_element[idx])
        line_emissivity_array = jnp.asarray(apec_file.line_emissivity[idx])
        end_index_lines = jnp.asarray(apec_file.line_end_index[idx])

    return line_energy_array, line_element_array, line_emissivity_array, end_index_lines


pure_callback_temperature_shape = jax.eval_shape(lambda: jax.tree.map(jnp.asarray, temperature_table_getter(10.0)))
pure_callback_continuum_shape = jax.eval_shape(lambda: jax.tree.map(jnp.asarray, continuum_table_getter(0)))
pure_callback_pseudo_shape = jax.eval_shape(lambda: jax.tree.map(jnp.asarray, pseudo_table_getter(0)))
pure_callback_line_shape = jax.eval_shape(lambda: jax.tree.map(jnp.asarray, lines_table_getter(0)))


@jax.jit
def get_temperature(kT):
    return jax.pure_callback(temperature_table_getter, pure_callback_temperature_shape, kT)


@jax.jit
def get_continuum(idx):
    return jax.pure_callback(continuum_table_getter, pure_callback_continuum_shape, idx)


@jax.jit
def get_pseudo(idx):
    return jax.pure_callback(pseudo_table_getter, pure_callback_pseudo_shape, idx)


@jax.jit
def get_lines(idx):
    return jax.pure_callback(lines_table_getter, pure_callback_line_shape, idx)
