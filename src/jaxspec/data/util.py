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


_EXAMPLE_DIRECTORY = "example_data/NGC7793_ULX4"
_EXAMPLE_PHA_FILES = {
    "PN": "PN_spectrum_grp20.fits",
    "MOS1": "MOS1_spectrum_grp.fits",
    "MOS2": "MOS2_spectrum_grp.fits",
}


def _example_detectors(source):
    """Select the supported detector set in the same order for spectra and responses."""
    if source == "NGC7793_ULX4_PN":
        return ("PN",)
    if source == "NGC7793_ULX4_ALL":
        return tuple(_EXAMPLE_PHA_FILES)
    raise ValueError(f"{source} not recognized.")


def load_example_pha(
    source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"],
) -> Observation | dict[str, Observation]:
    """Load background-matched example spectra for testing or demonstrating a fit.

    ``NGC7793_ULX4_PN`` returns one Observation. ``NGC7793_ULX4_ALL`` returns
    an ordered mapping for PN, MOS1 and MOS2, matching ``load_example_instruments``.
    """
    observations = {
        detector: Observation.from_pha_file(
            table_manager.fetch(f"{_EXAMPLE_DIRECTORY}/{_EXAMPLE_PHA_FILES[detector]}"),
            bkg_path=table_manager.fetch(
                f"{_EXAMPLE_DIRECTORY}/{detector}background_spectrum.fits"
            ),
        )
        for detector in _example_detectors(source)
    }
    return observations["PN"] if source == "NGC7793_ULX4_PN" else observations


def load_example_instruments(
    source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"],
) -> Instrument | dict[str, Instrument]:
    """Load the response pairs matched to ``load_example_pha``.

    ``NGC7793_ULX4_PN`` returns one Instrument. ``NGC7793_ULX4_ALL`` returns
    an ordered mapping for PN, MOS1 and MOS2.
    """
    instruments = {
        detector: Instrument.from_ogip_file(
            table_manager.fetch(f"{_EXAMPLE_DIRECTORY}/{detector}.rmf"),
            table_manager.fetch(f"{_EXAMPLE_DIRECTORY}/{detector}.arf"),
        )
        for detector in _example_detectors(source)
    }
    return instruments["PN"] if source == "NGC7793_ULX4_PN" else instruments


def load_example_obsconf(
    source: Literal["NGC7793_ULX4_PN", "NGC7793_ULX4_ALL"],
) -> ObsConfiguration | dict[str, ObsConfiguration]:
    """Build ready-to-fit example observations over the 0.5--8 keV detector band.

    ``NGC7793_ULX4_PN`` returns one configuration. ``NGC7793_ULX4_ALL`` returns
    an ordered mapping for PN, MOS1 and MOS2 with their matched spectra and responses.
    """
    instruments = load_example_instruments(source)
    observations = load_example_pha(source)
    if source == "NGC7793_ULX4_PN":
        return ObsConfiguration.from_instrument(
            instruments, observations, low_energy=0.5, high_energy=8.0
        )
    return {
        detector: ObsConfiguration.from_instrument(
            instruments[detector], observations[detector], low_energy=0.5, high_energy=8.0
        )
        for detector in instruments
    }


def forward_model_with_multiple_inputs(
    model: "SpectralModel",
    parameters,
    obs_configuration: "ObsConfiguration",
    sparse=False,
):
    """Evaluate a spectral model for a batch of parameter sets.

    Delegates to [`ForwardModel.evaluate`][jaxspec.fit._forward_model.ForwardModel.evaluate] so
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

    # A shared handler splits the key independently for each observation.
    with handlers.seed(rng_seed=rng_key):
        for i, obsconf in enumerate(obsconf_list):
            countrate = forward_model_with_multiple_inputs(
                model, parameters, obsconf, sparse=sparsify_matrix
            )

            if apply_stat:
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
    Resolve the ARF, RMF and background files named in a PHA header.

    Look beside the PHA file for each relative link, accepting the exact name
    or its gzip counterpart. An absent, blank or ``none`` header value means
    no associated file and returns None, regardless of the corresponding
    ``require_*`` flag. A named file is loaded when present even if optional.

    Parameters:
        pha_path: The PHA file path.
        require_arf: Whether to raise if a named ARF file cannot be found.
        require_rmf: Whether to raise if a named RMF file cannot be found.
        require_bkg: Whether to raise if a named background file cannot be found.

    Returns:
        arf_path: The resolved ARF path, or None when no file is associated or optional and missing.
        rmf_path: The resolved RMF path, or None when no file is associated or optional and missing.
        bkg_path: The resolved background path, or None when no file is associated or optional and missing.
    """

    def find_path(file_name: str, directory: str, raise_err: bool = True) -> str | None:
        """Look up every named link; optionality controls only a missing-file error."""
        file_name = "" if file_name is None else str(file_name).strip()
        if not file_name or file_name.lower() == "none":
            return None
        return find_file_or_compressed_in_dir(file_name, directory, raise_err)

    header = fits.getheader(pha_path, "SPECTRUM")
    directory = str(Path(pha_path).parent)

    arf_path = find_path(header.get("ANCRFILE", "none"), directory, require_arf)
    rmf_path = find_path(header.get("RESPFILE", "none"), directory, require_rmf)
    bkg_path = find_path(header.get("BACKFILE", "none"), directory, require_bkg)

    return arf_path, rmf_path, bkg_path


def find_file_or_compressed_in_dir(
    path: str | Path, directory: str | Path, raise_err: bool
) -> str | None:
    """Resolve the exact named calibration/background file or its gzip equivalent.

    Relative links are interpreted beside the PHA file; absolute paths retain
    their meaning. A similarly prefixed backup is not a calibration match.
    Optional missing files return None, while required named files raise an
    error that identifies the expected path.
    """
    path = Path(path) if isinstance(path, str) else path
    directory = Path(directory) if isinstance(directory, str) else directory

    candidate = directory.joinpath(path)
    candidates = [candidate]
    if candidate.suffix.lower() != ".gz":
        candidates.append(candidate.with_name(candidate.name + ".gz"))
    for file in candidates:
        if file.is_file():
            return str(file)
    if raise_err:
        tried = ", ".join(str(file) for file in candidates)
        raise FileNotFoundError(f"Can't find {path} (tried: {tried}).")
    return None
