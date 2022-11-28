import chex
import jax.numpy as jnp
import haiku as hk
from jaxspec.model import _modules


class TestModules(chex.TestCase):

    @chex.all_variants
    def test_all_modules(self):
        """
        Test to evaluate energies with every model components
        """

        energy = jnp.geomspace(0.1, 100, 50)

        for name, module in _modules.items():

            @hk.testing.transform_and_run(jax_transform=self.variant)
            def f(inputs):
                return module()(inputs)

            try:
                out = f(energy)
            except Exception:
                self.fail(f'{name} component failed')

            self.assertEqual(out.shape, energy.shape)
