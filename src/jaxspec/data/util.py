from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from astropy.io import fits
from jax.experimental.sparse import BCOO
from numpyro import handlers

from ..model.abc import SpectralModel
from ..util.online_storage import table_manager
from . import Instrument, ObsConfiguration, Observation

K = TypeVar("K")
V = TypeVar("V")

if TYPE_CHECKING:
    from ..data import ObsConfiguration
    from ..model.abc import SpectralModel


def load_example_pha(
    source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"],
) -> (Observation, list[Observation] | dict[str, Observation]):
    """
    Load some example observations from the package data.

    Parameters:
        source: The source to be loaded. Can be either "NGC7793_ULX4_PN" or "NGC7793_ULX4_ALL".
    """

    if source == "NGC7793_ULX4_PN":
        return Observation.from_pha_file(
            table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits"),
            bkg_path=table_manager.fetch("example_data/NGC7793_ULX4/PNbackground_spectrum.fits"),
        )

    elif source == "NGC7793_ULX4_ALL":
        return {
            "PN": Observation.from_pha_file(
                table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits"),
                bkg_path=table_manager.fetch(
                    "example_data/NGC7793_ULX4/PNbackground_spectrum.fits"
                ),
            ),
            "MOS1": Observation.from_pha_file(
                table_manager.fetch("example_data/NGC7793_ULX4/MOS1_spectrum_grp.fits"),
                bkg_path=table_manager.fetch(
                    "example_data/NGC7793_ULX4/MOS1background_spectrum.fits"
                ),
            ),
            "MOS2": Observation.from_pha_file(
                table_manager.fetch("example_data/NGC7793_ULX4/MOS2_spectrum_grp.fits"),
                bkg_path=table_manager.fetch(
                    "example_data/NGC7793_ULX4/MOS2background_spectrum.fits"
                ),
            ),
        }

    else:
        raise ValueError(f"{source} not recognized.")


def load_example_instruments(source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"]):
    """
    Load some example instruments from the package data.

    Parameters:
        source: The source to be loaded. Can be either "NGC7793_ULX4_PN" or "NGC7793_ULX4_ALL".

    """
    if source == "NGC7793_ULX4_PN":
        return Instrument.from_ogip_file(
            table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf"),
            table_manager.fetch("example_data/NGC7793_ULX4/PN.arf"),
        )

    elif source == "NGC7793_ULX4_ALL":
        return {
            "PN": Instrument.from_ogip_file(
                table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf"),
                table_manager.fetch("example_data/NGC7793_ULX4/PN.arf"),
            ),
            "MOS1": Instrument.from_ogip_file(
                table_manager.fetch("example_data/NGC7793_ULX4/MOS1.rmf"),
                table_manager.fetch("example_data/NGC7793_ULX4/MOS1.arf"),
            ),
            "MOS2": Instrument.from_ogip_file(
                table_manager.fetch("example_data/NGC7793_ULX4/MOS2.rmf"),
                table_manager.fetch("example_data/NGC7793_ULX4/MOS2.arf"),
            ),
        }

    else:
        raise ValueError(f"{source} not recognized.")


def load_example_obsconf(source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"]):
    """
    Load some example ObsConfigurations.

    Parameters:
        source: The source to be loaded. Can be either "NGC7793_ULX4_PN" or "NGC7793_ULX4_ALL".
    """

    if source in "NGC7793_ULX4_PN":
        instrument = load_example_instruments(source)
        observation = load_example_pha(source)

        return ObsConfiguration.from_instrument(
            instrument, observation, low_energy=0.5, high_energy=8.0
        )

    elif source == "NGC7793_ULX4_ALL":
        instruments_dict = load_example_instruments(source)
        observations_dict = load_example_pha(source)

        return {
            key: ObsConfiguration.from_instrument(
                instruments_dict[key], observations_dict[key], low_energy=0.5, high_energy=8.0
            )
            for key in instruments_dict.keys()
        }

    else:
        raise ValueError(f"{source} not recognized.")


def forward_model_with_multiple_inputs(
    model: "SpectralModel",
    parameters,
    obs_configuration: "ObsConfiguration",
    sparse=False,
):
    energies = np.asarray(obs_configuration.in_energies)
    parameter_dims = next(iter(parameters.values())).shape

    def flux_func(p):
        return model.photon_flux(p, *energies)

    for _ in parameter_dims:
        flux_func = jax.vmap(flux_func)

    flux_func = jax.jit(flux_func)

    if sparse:
        # folding.transfer_matrix.data.density > 0.015 is a good criterion to consider sparsify
        transfer_matrix = BCOO.from_scipy_sparse(
            obs_configuration.transfer_matrix.data.to_scipy_sparse().tocsr()
        )

    else:
        transfer_matrix = np.asarray(obs_configuration.transfer_matrix.data.todense())

    expected_counts = jnp.matvec(transfer_matrix, flux_func(parameters))

    # The result is clipped at 1e-6 to avoid 0 round-off and diverging likelihoods
    return jnp.clip(expected_counts, a_min=1e-6)


def fakeit_for_multiple_parameters(
    obsconfs: ObsConfiguration | list[ObsConfiguration],
    model: SpectralModel,
    parameters: Mapping[K, V],
    rng_key: int = 0,
    apply_stat: bool = True,
    sparsify_matrix: bool = False,
):
    """
    Convenience function to simulate multiple spectra from a given model and a set of parameters.
    This is supposed to be somewhat optimized and can handle multiple parameters at once without blowing
    up the memory. The parameters should be passed as a dictionary with the parameter name as the key and
    the parameter values as the values, the value can be a scalar or a nd-array.

    # Example:

    ``` python
    from jaxspec.data.util import fakeit_for_multiple_parameters
    from numpy.random import default_rng

    rng = default_rng(42)
    size = (10, 30)

    parameters = {
        "tbabs_1_nh": rng.uniform(0.1, 0.4, size=size),
        "powerlaw_1_alpha": rng.uniform(1, 3, size=size),
        "powerlaw_1_norm": rng.exponential(10 ** (-0.5), size=size),
        "blackbodyrad_1_kT": rng.uniform(0.1, 3.0, size=size),
        "blackbodyrad_1_norm": rng.exponential(10 ** (-3), size=size)
    }

    spectra = fakeit_for_multiple_parameters(obsconf, model, parameters)
    ```

    Parameters:
        obsconfs: The observational setup(s).
        model: The model to use.
        parameters: The parameters of the model.
        rng_key: The random number generator seed.
        apply_stat: Whether to apply Poisson statistic on the folded spectra or not.
        sparsify_matrix: Whether to sparsify the matrix or not.
    """

    obsconf_list = [obsconfs] if isinstance(obsconfs, ObsConfiguration) else obsconfs
    fakeits = []

    for i, obsconf in enumerate(obsconf_list):
        countrate = forward_model_with_multiple_inputs(
            model, parameters, obsconf, sparse=sparsify_matrix
        )

        if apply_stat:
            with handlers.seed(rng_seed=rng_key):
                spectrum = numpyro.sample(
                    f"likelihood_obs_{i}",
                    numpyro.distributions.Poisson(countrate),
                )

        else:
            spectrum = countrate

        fakeits.append(spectrum)

    return fakeits[0] if len(fakeits) == 1 else fakeits


def data_path_finder(
    pha_path: str, require_arf: bool = True, require_rmf: bool = True, require_bkg: bool = False
) -> tuple[str | None, str | None, str | None]:
    """
    Function which tries its best to find the ARF, RMF and BKG files associated with a given PHA file.

    Parameters:
        pha_path: The PHA file path.
        require_arf: Whether to raise an error if the ARF file is not found.
        require_rmf: Whether to raise an error if the RMF file is not found.
        require_bkg: Whether to raise an error if the BKG file is not found.

    Returns:
        arf_path: The ARF file path.
        rmf_path: The RMF file path.
        bkg_path: The BKG file path.
    """

    def find_path(file_name: str, directory: str, raise_err: bool = True) -> str | None:
        if raise_err:
            if file_name.lower() != "none" and file_name != "":
                return find_file_or_compressed_in_dir(file_name, directory, raise_err)

        return None

    header = fits.getheader(pha_path, "SPECTRUM")
    directory = str(Path(pha_path).parent)

    arf_path = find_path(header.get("ANCRFILE", "none"), directory, require_arf)
    rmf_path = find_path(header.get("RESPFILE", "none"), directory, require_rmf)
    bkg_path = find_path(header.get("BACKFILE", "none"), directory, require_bkg)

    return arf_path, rmf_path, bkg_path


def find_file_or_compressed_in_dir(path: str | Path, directory: str | Path, raise_err: bool) -> str:
    """
    Try to find a file or its .gz compressed version in a given directory and return
    the full path of the file.
    """
    path = Path(path) if isinstance(path, str) else path
    directory = Path(directory) if isinstance(directory, str) else directory

    if directory.joinpath(path).exists():
        return str(directory.joinpath(path))

    matching_files = list(directory.glob(str(path) + "*"))

    if matching_files:
        file = matching_files[0]
        if file.suffix == ".gz":
            return str(file)

    elif raise_err:
        raise FileNotFoundError(f"Can't find {path}(.gz) in {directory}.")
