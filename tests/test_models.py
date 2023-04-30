import os
import sys
import chex
chex.set_n_cpu_devices(n=2)

import jax.numpy as jnp
import haiku as hk
from jax import vmap
from jaxspec.model.additive import Logparabola, Powerlaw
from jaxspec.model.multiplicative import Tbabs

#Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestModules(chex.TestCase):

    def setUp(self):
        from jaxspec.model import model_components
        self.module_dict = model_components.items()
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
