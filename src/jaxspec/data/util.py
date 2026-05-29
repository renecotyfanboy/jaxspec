from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import jax
import numpyro

from astropy.io import fits
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
    """Evaluate a spectral model for a batch of parameter sets.

    Delegates to :meth:`~jaxspec.fit._forward_model.ForwardModel.evaluate` so
    ``fakeit``, posterior-predictive checks, and the numpyro likelihood share
    one spectral + folding code path. ``jax.vmap`` is applied per parameter
    batch dimension.

    Parameters:
        model: The spectral model.
        parameters: A dict mapping dotted-path parameter names (e.g.
            ``"powerlaw_1.alpha"``) to arrays whose shape encodes
            the batch dimensions. Every model parameter must be supplied —
            there is no default fallback; a missing key raises ``ValueError``.
        obs_configuration: The observation configuration providing the energy
            grid and transfer matrix.
        sparse: Whether to use a sparse BCOO transfer matrix.

    Returns:
        Expected counts with shape ``(*batch_dims, n_channels)``, clipped at
        ``1e-6`` (the ``InstrumentModel.fold`` floor).
    """
    from ..fit._forward_model import ForwardModel
    from ..fit._prior_resolution import _enumerate_leaves

    forward = ForwardModel(model, {"data": obs_configuration}, sparsify_matrix=sparse)

    # fakeit has no prior-style defaults: every model parameter must be supplied.
    # Validate up front so a forgotten key fails with a parameter-centric message
    # rather than the prior-dict KeyError ``evaluate`` would raise lazily inside
    # the JIT trace.
    required = {
        up.removeprefix("spectrum.")
        for up in _enumerate_leaves(forward)
        if up.startswith("spectrum.")
    }
    missing = required - set(parameters)
    if missing:
        raise ValueError(
            f"fakeit requires a value for every model parameter; missing: {sorted(missing)}."
        )

    # Promote user keys ("tbabs_1.nh", "powerlaw_1.alpha") to leaf paths
    # matching the single-obs ForwardModel tree ("spectrum.data.<rest>").
    inputs = {f"spectrum.data.{path}": value for path, value in parameters.items()}
    parameter_dims = next(iter(parameters.values())).shape

    def evaluate(inp):
        return forward.evaluate(inp)["data"]["source"]

    for _ in parameter_dims:
        evaluate = jax.vmap(evaluate)

    return jax.jit(evaluate)(inputs)


def fakeit_for_multiple_parameters(
    obsconfs: ObsConfiguration | list[ObsConfiguration],
    model: SpectralModel,
    parameters: Mapping[K, V],
    rng_key: int = 0,
    apply_stat: bool = True,
    sparsify_matrix: bool = False,
):
    """Simulate multiple spectra from a spectral model and a batch of parameters.

    Handles batched parameter arrays efficiently via ``jax.vmap`` and optionally
    applies Poisson noise.

    Example:
        from jaxspec.data.util import fakeit_for_multiple_parameters
        from numpy.random import default_rng

        rng = default_rng(42)
        size = (10, 30)

        parameters = {
            "tbabs_1.nh": rng.uniform(0.1, 0.4, size=size),
            "powerlaw_1.alpha": rng.uniform(1, 3, size=size),
            "powerlaw_1.norm": rng.exponential(10 ** (-0.5), size=size),
            "blackbodyrad_1.kT": rng.uniform(0.1, 3.0, size=size),
            "blackbodyrad_1.norm": rng.exponential(10 ** (-3), size=size),
        }

        spectra = fakeit_for_multiple_parameters(obsconf, model, parameters)

    Parameters:
        obsconfs: One or more observation configurations.
        model: The spectral model to evaluate.
        parameters: Dict mapping dotted-path parameter names to arrays whose
            shape encodes the batch dimensions.
        rng_key: Random number generator seed for Poisson sampling.
        apply_stat: Whether to apply Poisson noise to the folded spectra.
        sparsify_matrix: Whether to use sparse transfer matrices.

    Returns:
        A single array (one obs) or a list of arrays (multiple obs), each with
        shape ``(*batch_dims, n_channels)``.
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
