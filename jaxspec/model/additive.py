import haiku as hk
import jax.numpy as jnp
from .abc import AdditiveComponent
from haiku.initializers import Constant


class PowerLaw(AdditiveComponent):

    def __call__(self, energy):

        alpha = hk.get_parameter('alpha', [], init=Constant(11/3))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*energy**(-alpha)
