import haiku as hk
from abc import ABC, abstractmethod


class ModelComponent(hk.Module, ABC):

    @abstractmethod
    def __call__(self, E):
        """
        Return the model evaluated at a given energy
        """
        pass


class AdditiveComponent(ModelComponent, ABC):

    pass


class MultiplicativeComponent(ModelComponent, ABC):

    pass
