"""This module contains the functions that load the APEC tables from the HDF5 file. They are implemented as JAX
pure callback to enable reading data from the files without saturating the memory."""

import h5netcdf
import jax
import jax.numpy as jnp

from ...util.online_storage import table_manager


@jax.jit
def temperature_table_getter():
    with h5netcdf.File(table_manager.fetch("apec.nc"), "r") as f:
        temperature = jnp.asarray(f["/temperature"])

    return temperature


@jax.jit
def get_temperature(kT):
    temperature = temperature_table_getter()
    idx = jnp.searchsorted(temperature, kT) - 1

    return idx, temperature[idx], temperature[idx + 1]


@jax.jit
def continuum_table_getter():
    with h5netcdf.File(table_manager.fetch("apec.nc"), "r") as f:
        continuum_energy = jnp.asarray(f["/continuum_energy"])
        continuum_emissivity = jnp.asarray(f["/continuum_emissivity"])
        continuum_end_index = jnp.asarray(f["/continuum_end_index"])

    return continuum_energy, continuum_emissivity, continuum_end_index


@jax.jit
def pseudo_table_getter():
    with h5netcdf.File(table_manager.fetch("apec.nc"), "r") as f:
        pseudo_energy = jnp.asarray(f["/pseudo_energy"])
        pseudo_emissivity = jnp.asarray(f["/pseudo_emissivity"])
        pseudo_end_index = jnp.asarray(f["/pseudo_end_index"])

    return pseudo_energy, pseudo_emissivity, pseudo_end_index


@jax.jit
def line_table_getter():
    with h5netcdf.File(table_manager.fetch("apec.nc"), "r") as f:
        line_energy = jnp.asarray(f["/line_energy"])
        line_element = jnp.asarray(f["/line_element"])
        line_emissivity = jnp.asarray(f["/line_emissivity"])
        line_end_index = jnp.asarray(f["/line_end_index"])

    return line_energy, line_element, line_emissivity, line_end_index


@jax.jit
def get_continuum(idx):
    continuum_energy, continuum_emissivity, continuum_end_index = continuum_table_getter()
    return continuum_energy[idx], continuum_emissivity[idx], continuum_end_index[idx]


@jax.jit
def get_pseudo(idx):
    pseudo_energy, pseudo_emissivity, pseudo_end_index = pseudo_table_getter()
    return pseudo_energy[idx], pseudo_emissivity[idx], pseudo_end_index[idx]


@jax.jit
def get_lines(idx):
    line_energy, line_element, line_emissivity, line_end_index = line_table_getter()
    return line_energy[idx], line_element[idx], line_emissivity[idx], line_end_index[idx]
