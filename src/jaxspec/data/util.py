import importlib.resources
import numpyro
import jax
import numpy as np
import haiku as hk
from numpy.typing import ArrayLike
from collections.abc import Mapping
from typing import TypeVar
from .observation import Observation
from .instrument import Instrument
from ..model.abc import SpectralModel
from ..fit import CountForwardModel
from numpyro import handlers

K = TypeVar("K")
V = TypeVar("V")

example_observations = {
    "PN": Observation.from_pha_file(
        importlib.resources.files("jaxspec") / "data/example_data/PN.pha",
        low_energy=0.3,
        high_energy=12,
    ),
    "MOS1": Observation.from_pha_file(
        importlib.resources.files("jaxspec") / "data/example_data/MOS1.pha",
        low_energy=0.3,
        high_energy=7,
    ),
    "MOS2": Observation.from_pha_file(
        importlib.resources.files("jaxspec") / "data/example_data/MOS2.pha",
        low_energy=0.3,
        high_energy=7,
    ),
}


def fakeit(
    instrument: Instrument | list[Instrument],
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

    instruments = [instrument] if isinstance(instrument, Instrument) else instrument
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
    instrument: Instrument | list[Instrument],
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
        apply_stat: Whether to make a Poisson realisation of the folded spectra or not.
    """

    instruments = [instrument] if isinstance(instrument, Instrument) else instrument
    fakeits = []

    for i, obs in enumerate(instruments):
        transformed_model = hk.without_apply_rng(hk.transform(lambda par: CountForwardModel(model, obs)(par)))

        def obs_model(p):
            return transformed_model.apply(None, p)

        if apply_stat:
            with handlers.seed(rng_seed=rng_key):
                spectrum = numpyro.sample(
                    f"likelihood_obs_{i}",
                    numpyro.distributions.Poisson(jax.vmap(obs_model)(parameters)),
                )

        else:
            spectrum = jax.vmap(obs_model)(parameters)

        fakeits.append(spectrum)

    return fakeits[0] if len(fakeits) == 1 else fakeits
