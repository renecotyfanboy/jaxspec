import importlib.resources
import numpyro
import jax
import numpy as np
import haiku as hk
from pathlib import Path
from numpy.typing import ArrayLike
from collections.abc import Mapping
from typing import TypeVar

from .ogip import DataPHA, DataARF, DataRMF
from . import Observation, Instrument, ObsConfiguration
from ..model.abc import SpectralModel
from ..fit import CountForwardModel
from numpyro import handlers

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
            high_energy=7.5,
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
) -> ArrayLike | list[ArrayLike]:
    """
    This function is a convenience function that allows to simulate spectra from a given model and a set of parameters.
    It requires an instrumental setup, and unlike in
    [XSPEC's fakeit](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node72.html), the error on the counts is given
    exclusively by Poisson statistics.

    Parameters:
        instrument: The instrumental setup.
        model: The model to use.
        parameters: The parameters of the model.
        rng_key: The random number generator seed.
    """

    instruments = [instrument] if isinstance(instrument, ObsConfiguration) else instrument
    fakeits = []

    for i, instrument in enumerate(instruments):
        transformed_model = hk.without_apply_rng(hk.transform(lambda par: CountForwardModel(model, instrument)(par)))

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
    apply_stat=True,
):
    """
    This function is a convenience function that allows to simulate spectra multiple spectra from a given model and a
    set of parameters.

    TODO : avoid redundancy, better doc and type hints

    Parameters:
        instrument: The instrumental setup.
        model: The model to use.
        parameters: The parameters of the model.
        rng_key: The random number generator seed.
        apply_stat: Whether to apply Poisson statistic on the folded spectra or not.
    """

    instruments = [instrument] if isinstance(instrument, ObsConfiguration) else instrument
    fakeits = []

    for i, obs in enumerate(instruments):
        transformed_model = hk.without_apply_rng(hk.transform(lambda par: CountForwardModel(model, obs)(par)))

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


def data_loader(pha_path: str, arf_path=None, rmf_path=None, bkg_path=None):
    """
    This function is a convenience function that allows to load PHA, ARF and RMF data
    from a given PHA file, using either the ARF/RMF/BKG filenames in the header or the
    specified filenames overwritten by the user.

    Parameters:
        pha_path: The PHA file path.
        arf_path: The ARF file path.
        rmf_path: The RMF file path.
        bkg_path: The BKG file path.
    """

    pha = DataPHA.from_file(pha_path)
    directory = str(Path(pha_path).parent)

    if arf_path is None:
        if pha.ancrfile != "none" and pha.ancrfile != "":
            arf_path = find_file_or_compressed_in_dir(pha.ancrfile, directory)

    if rmf_path is None:
        if pha.respfile != "none" and pha.respfile != "":
            rmf_path = find_file_or_compressed_in_dir(pha.respfile, directory)

    if bkg_path is None:
        if pha.backfile.lower() != "none" and pha.backfile != "":
            bkg_path = find_file_or_compressed_in_dir(pha.backfile, directory)

    arf = DataARF.from_file(arf_path) if arf_path is not None else None
    rmf = DataRMF.from_file(rmf_path) if rmf_path is not None else None
    bkg = DataPHA.from_file(bkg_path) if bkg_path is not None else None

    metadata = {
        "observation_file": pha_path,
        "background_file": bkg_path,
        "response_matrix_file": rmf_path,
        "ancillary_response_file": arf_path,
    }

    return pha, arf, rmf, bkg, metadata


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
