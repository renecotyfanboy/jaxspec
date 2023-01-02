import os
import sys
import chex
import jax.numpy as jnp
from jax import vmap, grad
from jaxspec.model.from_string import build_model

#Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)

chex.set_n_cpu_devices(n=2)


class TestFromString(chex.TestCase):

    def setUp(self):
        from jaxspec.model import _modules
        self.module_dict = _modules.items()
        self.model_string = 'expfac*lorentz'
        self.energy = jnp.geomspace(0.1, 100, 50)

    @chex.all_variants
    def test_build_model_str(self):
        """
        Test building a model from fixed string and evaluating it
        """

        model = build_model(model_string=self.model_string)

        @self.variant
        def f(inputs): return model.apply(model.init(None, self.energy), inputs)

        out = f(self.energy)

        self.assertEqual(out.shape, self.energy.shape, f"'{self.model_string}' changes input shape")

    @chex.all_variants
    def test_grad_model_str(self):
        """
        Test building a model from fixed string and evaluating its gradient
        """

        model = build_model(model_string=self.model_string)

        @self.variant
        def f(inputs):
            init = model.init(None, self.energy)
            func = lambda e : model.apply(init, e)
            return grad(func)(inputs)

        grad_value = vmap(f)(self.energy)

        self.assertEqual(jnp.sum(jnp.isnan(grad_value)), 0, f'{self.model_string} gradient returns NaNs')
