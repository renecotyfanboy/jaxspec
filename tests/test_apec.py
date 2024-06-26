import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest


def test_apec_model():
    APEC = pytest.importorskip("jaxspec.model.additive.APEC")
    model = APEC(variant="none")
    func = jax.jit(model.photon_flux)
    energy = jnp.linspace(0.2, 10, 2901)
    plt.plot(energy[:-1], func(model.params, energy[:-1], energy[1:]))
    plt.loglog()
    plt.show()
