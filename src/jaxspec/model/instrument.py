from abc import ABC, abstractmethod
from collections.abc import Callable

import numpyro

from flax import nnx
from numpyro.distributions import Distribution


class GainModel(ABC, nnx.Module):
    @abstractmethod
    def numpyro_model(self, observation_name: str):
        pass


class ConstantGain(GainModel):
    def __init__(self, prior_distribution: Distribution):
        self.prior_distribution = prior_distribution

    def numpyro_model(self, observation_name: str):
        factor = numpyro.sample(f"ins/~/gain_{observation_name}", self.prior_distribution)

        def gain(energy):
            return factor

        return gain


class ShiftModel(ABC, nnx.Module):
    @abstractmethod
    def numpyro_model(self, observation_name: str):
        pass


class ConstantShift(ShiftModel):
    def __init__(self, prior_distribution: Distribution):
        self.prior_distribution = prior_distribution

    def numpyro_model(self, observation_name: str):
        shift_offset = numpyro.sample(f"ins/~/shift_{observation_name}", self.prior_distribution)

        def shift(energy):
            return energy + shift_offset

        return shift


class InstrumentModel(nnx.Module):
    def __init__(
        self,
        reference_observation_name: str,
        gain_model: GainModel | None = None,
        shift_model: ShiftModel | None = None,
    ):
        self.reference = reference_observation_name
        self.gain_model = gain_model
        self.shift_model = shift_model

    def get_gain_and_shift_model(
        self, observation_name: str
    ) -> tuple[Callable | None, Callable | None]:
        """
        Return the gain and shift models for the given observation. It should be called within a numpyro model.
        """

        if observation_name == self.reference:
            return None, None

        else:
            gain = (
                self.gain_model.numpyro_model(observation_name)
                if self.gain_model is not None
                else None
            )
            shift = (
                self.shift_model.numpyro_model(observation_name)
                if self.shift_model is not None
                else None
            )

            return gain, shift
