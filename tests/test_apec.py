import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxspec.model.additive import APEC


def test_apec_model():
    model = APEC(variant="none")
    func = jax.jit(model.photon_flux)
    energy = jnp.linspace(0.2, 10, 29001)
    plt.plot(energy[:-1], func(model.params, energy[:-1], energy[1:]))
    plt.loglog()
    plt.show()
