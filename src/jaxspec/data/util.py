from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeVar

import jax
import numpyro

from astropy.io import fits
from numpyro import handlers

from .._fit._build_model import forward_model
from ..model.abc import SpectralModel
from ..util.online_storage import table_manager
from . import Instrument, ObsConfiguration, Observation

K = TypeVar("K")
V = TypeVar("V")


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


def fakeit_for_multiple_parameters(
    instrument: ObsConfiguration | list[ObsConfiguration],
    model: SpectralModel,
    parameters: Mapping[K, V],
    rng_key: int = 0,
    apply_stat: bool = True,
    sparsify_matrix: bool = False,
):
    """
    Convenience function to simulate multiple spectra from a given model and a set of parameters.


    Parameters:
        instrument: The instrumental setup.
        model: The model to use.
        parameters: The parameters of the model.
        rng_key: The random number generator seed.
        apply_stat: Whether to apply Poisson statistic on the folded spectra or not.
        sparsify_matrix: Whether to sparsify the matrix or not.
    """

    instruments = [instrument] if isinstance(instrument, ObsConfiguration) else instrument
    fakeits = []

    for i, obs in enumerate(instruments):
        countrate = jax.vmap(lambda p: forward_model(model, p, instrument, sparse=sparsify_matrix))(
            parameters
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


def data_path_finder(pha_path: str) -> tuple[str | None, str | None, str | None]:
    """
    Function which tries its best to find the ARF, RMF and BKG files associated with a given PHA file.

    Parameters:
        pha_path: The PHA file path.

    Returns:
        arf_path: The ARF file path.
        rmf_path: The RMF file path.
        bkg_path: The BKG file path.
    """

    def find_path(file_name: str, directory: str) -> str | None:
        if file_name.lower() != "none" and file_name != "":
            return find_file_or_compressed_in_dir(file_name, directory)
        else:
            return None

    header = fits.getheader(pha_path, "SPECTRUM")
    directory = str(Path(pha_path).parent)

    arf_path = find_path(header.get("ANCRFILE", "none"), directory)
    rmf_path = find_path(header.get("RESPFILE", "none"), directory)
    bkg_path = find_path(header.get("BACKFILE", "none"), directory)

    return arf_path, rmf_path, bkg_path


def find_file_or_compressed_in_dir(path: str | Path, directory: str | Path) -> str:
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

    else:
        raise FileNotFoundError(f"Can't find {path}(.gz) in {directory}.")
