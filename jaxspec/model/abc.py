import haiku as hk
from abc import ABC, abstractmethod


class ModelComponent(ABC, hk.Module):
    """Abstract class that allows to build model components from haiku modules"""

    @abstractmethod
    def __call__(self, energy):
        """
        Return the model evaluated at a given energy
        """
        pass


class AdditiveComponent(ABC, ModelComponent):
    """Abstract class for additive model components"""
    pass


class MultiplicativeComponent(ABC, ModelComponent):
    """Abstract class for multiplicative model components"""
    pass
