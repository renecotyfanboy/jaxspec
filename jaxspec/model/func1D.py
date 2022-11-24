import haiku as hk
import jax.numpy as jnp
from .abc import ModelComponent


class PowerLaw(ModelComponent):

    def __call__(self, energy):

        alpha = hk.get_parameter('alpha')
        norm = hk.get_parameter('norm')

        return norm*energy**(-alpha)
