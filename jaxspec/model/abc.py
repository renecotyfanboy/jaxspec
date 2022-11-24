import haiku as hk
from abc import ABC, abstractmethod


class AdditiveComponent(hk.Module, ABC):

    @abstractmethod
    def __call__(self, E):
        """
        Return the model evaluated at a given energy
        """
        pass


class MultiplicativeComponent(hk.Module, ABC):

    @abstractmethod
    def __call__(self, E):
        """
        Return the model evaluated at a given energy
        """
        pass
