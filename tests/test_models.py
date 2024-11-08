import base64

from io import BytesIO

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
import requests

from jaxspec.model.additive import Additiveconstant, Blackbodyrad, Powerlaw
from jaxspec.model.list import additive_components, multiplicative_components
from jaxspec.model.multiplicative import MultiplicativeConstant, Tbabs
from PIL import Image


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


def test_mermaid_representation():
    spectral_model = Tbabs() * (Powerlaw() + Blackbodyrad())

    def mm(graph):
        graphbytes = graph.encode("utf8")
        base64_bytes = base64.urlsafe_b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        response = requests.get("https://mermaid.ink/img/" + base64_string)
        return Image.open(BytesIO(response.content))

    img = mm(spectral_model.to_mermaid())
    plt.imshow(img)
    plt.suptitle("Tbabs() * (Powerlaw() + Blackbodyrad())")
    plt.axis("off")
    plt.show()
