import haiku as hk
from abc import ABC, abstractmethod


class ModelComponent(ABC, hk.Module):

    def __len__(self):
        return 1


class AdditiveComponent(ModelComponent):

    @abstractmethod
    def __call__(self, E):
        """
        Return the model evaluated at a given energy
        """
        pass


class MultiplicativeComponent(ModelComponent):

    @abstractmethod
    def __call__(self, E):
        """
        Return the model evaluated at a given energy
        """
        pass
