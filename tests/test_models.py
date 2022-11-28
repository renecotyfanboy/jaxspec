import chex
from jax import grad, vmap
import jax.numpy as jnp
import haiku as hk

chex.set_n_cpu_devices(n=2)

class TestModules(chex.TestCase):

    def setUp(self):
        from jaxspec.model import _modules
        self.module_dict = _modules.items()
        self.energy = jnp.geomspace(0.1, 100, 50)

    @chex.all_variants
    def test_all_modules(self):
        """
        Test to evaluate energies with every model components
        """

        for name, module in self.module_dict:
            @hk.testing.transform_and_run(jax_transform=self.variant)
            def f(inputs): return module()(inputs)
            out = f(self.energy)

            self.assertEqual(out.shape, self.energy.shape, f'{name} changes input shape')

    @chex.all_variants
    def test_all_modules_grad(self):
        """
        Test to evaluate gradient of every model component
        """

        for name, module in self.module_dict:
            @hk.testing.transform_and_run(jax_transform=self.variant)
            def f(inputs): return hk.value_and_grad(module())(inputs)

            out, grad = vmap(f)(self.energy)

            self.assertEqual(jnp.sum(jnp.isnan(grad)), 0, f'{name} gradient returns NaNs')
