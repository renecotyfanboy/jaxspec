import jax.numpy as jnp
import pytest

from jaxspec.model.additive import Additiveconstant
from jaxspec.model.list import additive_components, multiplicative_components
from jaxspec.model.multiplicative import MultiplicativeConstant


@pytest.mark.parametrize("test_input", list(additive_components.keys()))
def test_additive_components(test_input):
    energy = jnp.geomspace(0.5, 10, 1000)
    e_low = energy[:-1]
    e_high = energy[1:]

    spectral_model = (
        MultiplicativeConstant()
        * MultiplicativeConstant()
        * (Additiveconstant() + additive_components[test_input]())
    )
    out = spectral_model.turbo_flux(e_low, e_high)
    assert out.shape == e_low.shape


@pytest.mark.parametrize("test_input", list(multiplicative_components.keys()))
def test_multiplicative_components(test_input):
    energy = jnp.geomspace(0.5, 10, 1000)
    e_low = energy[:-1]
    e_high = energy[1:]

    spectral_model = (
        MultiplicativeConstant()
        * multiplicative_components[test_input]()
        * (Additiveconstant() + Additiveconstant())
    )
    out = spectral_model.turbo_flux(e_low, e_high)
    assert out.shape == e_low.shape
