import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from astropy.table import Table
from flax import nnx
from jax import lax
from jax.scipy.interpolate import RegularGridInterpolator
from jax.typing import ArrayLike
from tqdm.auto import tqdm

from ..model.abc import AdditiveComponent
from .interpolator import RegularGridInterpolatorWithGrad


class TableManager:
    """
    Handler for the tabulated data from `xspec` tabulated models. The table must follow the format speficied in the
    [`atable`](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XSmodelAtable.html) additive component.
    """

    def __init__(self, table_path):
        self._table_path = table_path

        # Read the tables once.
        raw_parameters = Table.read(self._table_path, "PARAMETERS")
        raw_energies = Table.read(self._table_path, "ENERGIES")
        raw_spectra = Table.read(self._table_path, "SPECTRA")

        # Build and wrap the parameter table.
        parameter_table = tuple(np.asarray(row["VALUE"], dtype=float) for row in raw_parameters)
        parameters = []
        for row in raw_parameters:
            parameter_name = row["NAME"].rstrip()
            parameters.append((parameter_name, float(row["INITIAL"])))

        # Process and wrap the energies table.
        energy_low = np.asarray(raw_energies["ENERG_LO"].value, dtype=float)
        energy_high = np.asarray(raw_energies["ENERG_HI"].value, dtype=float)
        energies_table = np.vstack((energy_low, energy_high))

        # Compute the parameter shape
        parameter_shape = tuple()
        for parameter in parameter_table:
            parameter_shape += parameter.shape

        # Build and wrap the spectra table.
        total_shape = parameter_shape + energy_low.shape

        spectra_table = np.empty(total_shape, dtype=float)
        for idx, par_indexes in enumerate(np.ndindex(*parameter_shape)):
            spectra_table[par_indexes] = np.asarray(raw_spectra["INTPSPEC"][idx], dtype=float)

        self.parameters = parameters
        self.energies_table = energies_table
        self.parameter_table = parameter_table
        self.parameter_shape = parameter_shape
        self.spectra_table = spectra_table

    def check_proper_ordering(self):
        """
        Assert that the parameter ordering in the spectra table is consistent with the parameter table.
        """

        spectra_table = Table.read(self._table_path, "SPECTRA")
        pars_mesh = np.meshgrid(*self.parameter_table, indexing="ij", sparse=False)

        for idx, par_indexes in enumerate(
            tqdm(np.ndindex(*self.parameter_shape), total=len(spectra_table))
        ):
            expected = np.asarray([par[par_indexes] for par in pars_mesh])
            obtained = jnp.asarray(spectra_table["PARAMVAL"][idx], dtype=float)
            assert jnp.allclose(expected, obtained), "Parameter ordering mismatch"


class TabulatedModel(nnx.Module):
    """
    See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/ogip_92_009/ogip_92_009.pdf
    """

    def __init__(self, table_path: str):
        table_manager = TableManager(table_path)

        self._parameters = table_manager.parameters
        self._parameter_table = table_manager.parameter_table
        self._spectra_table = table_manager.spectra_table
        self._energies_table = table_manager.energies_table

        # Instantiate the parameters of the nnx.Module
        for parameter_name, value in table_manager.parameters:
            setattr(self, parameter_name, nnx.Param(value))

        del table_manager

    def get_parameter_list(self):
        parameter_list = []
        for parameter_name, _ in self._parameters:
            parameter_list.append(getattr(self, parameter_name))

        return jnp.asarray(parameter_list, dtype=float)


"""
import numpy as np
from scipy.sparse import csr_matrix

def redistribution_matrix(old_e_low, old_e_high, new_e_low, new_e_high):
    new_bins = jnp.stack([new_e_low, new_e_high], axis=1)

    def scan_body(carry, new_bin):
        new_low, new_high = new_bin

        # Compute the overlap between the current new bin and all old bins.
        lower_bounds = jnp.maximum(old_e_low, new_low)
        upper_bounds = jnp.minimum(old_e_high, new_high)
        overlap_fraction = jnp.maximum(0, upper_bounds - lower_bounds) / (old_e_high - old_e_low)

        return carry, overlap_fraction

    _, matrix = lax.scan(scan_body, None, new_bins)
    return matrix.T
"""


def redistribute(
    integrated_spectrum: ArrayLike,
    old_e_low: ArrayLike,
    old_e_high: ArrayLike,
    e_low: ArrayLike,
    e_high: ArrayLike,
) -> ArrayLike:
    """
    Redistribute the integrated spectrum over the new energy bins.

    Parameters:
        integrated_spectrum: Integrated spectrum to redistribute.
        old_e_low: Lower bounds of the old energy bins.
        old_e_high: Upper bounds of the old energy bins.
        e_low: Lower bounds of the new energy bins.
        e_high: Upper bounds of the new energy bins.
    """
    new_bins = jnp.stack([e_low, e_high], axis=1)

    def scan_body(carry, new_bin):
        new_low, new_high = new_bin

        # Compute the overlap between the current new bin and all old bins.
        lower_bounds = jnp.maximum(old_e_low, new_low)
        upper_bounds = jnp.minimum(old_e_high, new_high)
        overlap_fraction = jnp.maximum(0, upper_bounds - lower_bounds) / (old_e_high - old_e_low)

        # Sum over old bins: each old bin contributes its integrated value times
        # the fraction of the new bin that overlaps with it.
        new_intensity = jnp.sum(overlap_fraction * integrated_spectrum)

        return carry, new_intensity

    _, redistributed_values = lax.scan(scan_body, None, new_bins)
    return redistributed_values


class AdditiveTabulated(AdditiveComponent, TabulatedModel):
    """
    Equivalent of the [`atable`](https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XSmodelAtable.html) additive
    component in `xspec`.
    """

    def __init__(self, table_path: str):
        super().__init__(table_path)

        self.norm = nnx.Param(1.0)

        self._interpolator = RegularGridInterpolator(
            self._parameter_table, self._spectra_table, fill_value=0.0
        )

    def _integrate_on_grid(self):
        return self._interpolator(self.get_parameter_list()).squeeze()

    def integrated_continuum(self, e_low, e_high):
        integrated_spectrum = self._integrate_on_grid()
        return jnp.asarray(self.norm) * redistribute(
            integrated_spectrum, *self._energies_table, e_low, e_high
        )


class TabulatedModelXarray(nnx.Module):
    """
    See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/ogip_92_009/ogip_92_009.pdf
    """

    def __init__(self, table_path: str):
        ds = xr.open_dataset(table_path)

        self._parameters = [dim for dim in ds["spectra"].dims if dim != "energy"]
        self._parameter_table = [np.asarray(ds[key]) for key in self._parameters]
        self._spectra_table = ds["spectra"]
        self._energies_table = (np.asarray(ds["energy_low"]), np.asarray(ds["energy_high"]))

        # Instantiate the parameters of the nnx.Module
        for parameter_name, parameter_value in zip(self._parameters, self._parameter_table):
            setattr(
                self,
                parameter_name,
                nnx.Param(np.median(parameter_value) + np.random.uniform(-5e-1, +5e-1)),
            )

    def get_parameter_list(self):
        parameter_list = []
        for parameter_name in self._parameters:
            parameter_list.append(getattr(self, parameter_name))

        return jnp.asarray(parameter_list, dtype=float)


class AdditiveTabulatedXarray(AdditiveComponent, TabulatedModelXarray):
    def __init__(self, table_path: str):
        super().__init__(table_path)

        self.norm = nnx.Param(1.0)

        interpolator = RegularGridInterpolatorWithGrad(
            self._parameter_table, np.asarray(self._spectra_table)
        )

        def callback(pars):
            value, grad = interpolator(pars)
            return np.vstack([value[None, :], grad])

        result = callback(self.get_parameter_list())

        out_type = jax.ShapeDtypeStruct(jnp.shape(result), jnp.result_type(result))

        @jax.custom_jvp
        def spectrum_interpolation(pars):
            result = jax.pure_callback(
                lambda p: callback(p), out_type, pars, vmap_method="legacy_vectorized"
            )
            return result[0, ...]

        @spectrum_interpolation.defjvp
        def spectrum_interpolation_jvp(primals, tangents):
            pars = primals
            pars_dot = tangents

            result = jax.pure_callback(
                lambda p: callback(p), out_type, pars, vmap_method="legacy_vectorized"
            )
            value = result[0, ...]
            grad = result[1:, ...]

            return value, jnp.squeeze(jnp.asarray(pars_dot) @ grad)

        self._spectrum_interpolation = spectrum_interpolation

    def _integrate_on_grid(self):
        return self._spectrum_interpolation(jnp.asarray(self.get_parameter_list()))

    def integrated_continuum(self, e_low, e_high):
        integrated_spectrum = self._integrate_on_grid()
        return jnp.asarray(self.norm) * redistribute(
            integrated_spectrum, *self._energies_table, e_low, e_high
        )
