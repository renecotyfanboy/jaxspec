import haiku as hk
import jax.numpy as jnp
from .abc import MultiplicativeComponent
from haiku.initializers import Constant


class ExpFac(MultiplicativeComponent):

    def __call__(self, energy):

        amplitude = hk.get_parameter('A', [], init=Constant(1))
        factor = hk.get_parameter('f', [], init=Constant(1))
        pivot = hk.get_parameter('E_c', [], init=Constant(1))

        return jnp.where(energy >= pivot, 1. + amplitude*jnp.exp(-factor*energy), 1.)
