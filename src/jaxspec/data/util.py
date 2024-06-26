import importlib.resources

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import haiku as hk
import jax
import numpy as np
import numpyro

from astropy.io import fits
from numpy.typing import ArrayLike
from numpyro import handlers

from ..fit import CountForwardModel
from ..model.abc import SpectralModel
from . import Instrument, ObsConfiguration, Observation

K = TypeVar("K")
V = TypeVar("V")


def load_example_observations():
    """
    Load some example observations from the package data.
    """

    example_observations = {
        "PN": Observation.from_pha_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/PN_spectrum_grp20.fits"),
            low_energy=0.3,
            high_energy=7.5,
        ),
        "MOS1": Observation.from_pha_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS1_spectrum_grp.fits"),
            low_energy=0.3,
            high_energy=7,
        ),
        "MOS2": Observation.from_pha_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS2_spectrum_grp.fits"),
            low_energy=0.3,
            high_energy=7,
        ),
    }

    return example_observations


def load_example_instruments():
    """
    Load some example instruments from the package data.
    """

    example_instruments = {
        "PN": Instrument.from_ogip_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/PN.rmf"),
            str(importlib.resources.files("jaxspec") / "data/example_data/PN.arf"),
        ),
        "MOS1": Instrument.from_ogip_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS1.rmf"),
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS1.arf"),
        ),
        "MOS2": Instrument.from_ogip_file(
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS2.rmf"),
            str(importlib.resources.files("jaxspec") / "data/example_data/MOS2.arf"),
        ),
    }

    return example_instruments


def load_example_foldings():
    """
    Load some example instruments from the package data.
    """

    example_instruments = load_example_instruments()
    example_observations = load_example_observations()

    example_foldings = {
        "PN": ObsConfiguration.from_instrument(
            example_instruments["PN"],
            example_observations["PN"],
            low_energy=0.3,
            high_energy=8.0,
        ),
        "MOS1": ObsConfiguration.from_instrument(
            example_instruments["MOS1"],
            example_observations["MOS1"],
            low_energy=0.3,
            high_energy=7,
        ),
        "MOS2": ObsConfiguration.from_instrument(
            example_instruments["MOS2"],
            example_observations["MOS2"],
            low_energy=0.3,
            high_energy=7,
        ),
    }

    return example_foldings


def fakeit(
    instrument: ObsConfiguration | list[ObsConfiguration],
    model: SpectralModel,
    parameters: Mapping[K, V],
    rng_key: int = 0,
    sparsify_matrix: bool = False,
) -> ArrayLike | list[ArrayLike]:
    """
    Convenience function to simulate a spectrum from a given model and a set of parameters.
    It requires an instrumental setup, and unlike in
    [XSPEC's fakeit](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node72.html), the error on the counts is given
    exclusively by Poisson statistics.

    Parameters
    ----------
        instrument: The instrumental setup.
        model: The model to use.
        parameters: The parameters of the model.
        rng_key: The random number generator seed.
        sparsify_matrix: Whether to sparsify the matrix or not.
    """

    instruments = [instrument] if isinstance(instrument, ObsConfiguration) else instrument
    fakeits = []

    for i, instrument in enumerate(instruments):
        transformed_model = hk.without_apply_rng(
            hk.transform(
                lambda par: CountForwardModel(model, instrument, sparse=sparsify_matrix)(par)
            )
        )

        def obs_model(p):
            return transformed_model.apply(None, p)

        with handlers.seed(rng_seed=rng_key):
            counts = numpyro.sample(
                f"likelihood_obs_{i}",
                numpyro.distributions.Poisson(obs_model(parameters)),
            )

        """
        pha = DataPHA(
            instrument.rmf.channel,
            np.array(counts, dtype=int)*u.ct,
            instrument.exposure,
            grouping=instrument.grouping)

        observation = Observation(
            pha=pha,
            arf=instrument.arf,
            rmf=instrument.rmf,
            low_energy=instrument.low_energy,
            high_energy=instrument.high_energy
        )
        """

        fakeits.append(np.array(counts, dtype=int))

    return fakeits[0] if len(fakeits) == 1 else fakeits


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

    TODO : avoid redundancy, better doc and type hints

    Parameters
    ----------
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
        transformed_model = hk.without_apply_rng(
            hk.transform(lambda par: CountForwardModel(model, obs, sparse=sparsify_matrix)(par))
        )

        @jax.jit
        @jax.vmap
        def obs_model(p):
            return transformed_model.apply(None, p)

        if apply_stat:
            with handlers.seed(rng_seed=rng_key):
                spectrum = numpyro.sample(
                    f"likelihood_obs_{i}",
                    numpyro.distributions.Poisson(obs_model(parameters)),
                )

        else:
            spectrum = obs_model(parameters)

        fakeits.append(spectrum)

    return fakeits[0] if len(fakeits) == 1 else fakeits


def data_path_finder(pha_path: str) -> tuple[str | None, str | None, str | None]:
    """
    Function which tries its best to find the ARF, RMF and BKG files associated with a given PHA file.

    Parameters
    ----------
        pha_path: The PHA file path.

    Returns
    -------
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
