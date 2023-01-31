import os
import sys
import chex
from jax import grad, vmap
import jax.numpy as jnp

#Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)
chex.set_n_cpu_devices(n=2)


class TestFromString(chex.TestCase):

    def setUp(self):
        from jaxspec.model.from_string import build_model
        from jaxspec.model import model_components
        self.module_dict = model_components.items()
        self.model_string = 'expfac*lorentz'
        self.energy = jnp.geomspace(0.1, 100, 50)
        self.build = build_model

    @chex.all_variants
    def test_build_model_str(self):
        """
        Test building a model from fixed string and evaluating it
        """

        model = self.build(model_string=self.model_string)

        @self.variant
        def f(inputs): return model.apply(model.init(None, self.energy), inputs)

        out = f(self.energy)

        self.assertEqual(out.shape, self.energy.shape, f"'{self.model_string}' changes input shape")

    @chex.all_variants
    def test_grad_model_str(self):
        """
        Test building a model from fixed string and evaluating its gradient
        """

        model = self.build(model_string=self.model_string)

        @self.variant
        def f(inputs):
            init = model.init(None, self.energy)
            func = lambda e : model.apply(init, e)
            return grad(func)(inputs)

        grad_value = vmap(f)(self.energy)

        self.assertEqual(jnp.sum(jnp.isnan(grad_value)), 0, f'{self.model_string} gradient returns NaNs')
