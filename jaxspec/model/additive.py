import haiku as hk
import jax.numpy as jnp
from .abc import AdditiveComponent
from haiku.initializers import Constant


class Powerlaw(AdditiveComponent):

    def __call__(self, energy):

        alpha = hk.get_parameter('alpha', [], init=Constant(11/3))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*energy**(-alpha)


class Lorentz(AdditiveComponent):

    def __call__(self, energy):

        line_energy = hk.get_parameter('E_l', [], init=Constant(1))
        sigma = hk.get_parameter('sigma', [], init=Constant(1))
        norm = hk.get_parameter('norm', [], init=Constant(1))

        return norm*(sigma/(2*jnp.pi))/((energy-line_energy)**2 + (sigma/2)**2)
