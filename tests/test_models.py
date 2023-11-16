import os
import sys
import chex
import jax.numpy as jnp
import haiku as hk

chex.set_n_cpu_devices(n=4)


# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestModules(chex.TestCase):
    def setUp(self):
        from jaxspec.model.list import (
            model_components,
            additive_components,
        )

        self.module_dict = model_components.items()
        self.additive_dict = additive_components.items()
        self.energy = jnp.geomspace(0.1, 100, 50)

    @chex.all_variants
    def test_all_continuum(self):
        """
        Test to evaluate energies with every component's continuum
        """

        for name, module in self.module_dict:

            @hk.testing.transform_and_run(jax_transform=self.variant)
            def f(inputs):
                return module().continuum(inputs)

            out = f(self.energy)
            self.assertEqual(out.shape, self.energy.shape, f"{name} continuum changes input shape")

    @chex.all_variants
    def test_all_lines(self):
        """
        Test to evaluate energies with every component's emission lines
        """

        for name, module in self.additive_dict:

            @hk.testing.transform_and_run(jax_transform=self.variant)
            def f(input_low, input_high):
                return module().emission_lines(input_low, input_high)

            out = f(self.energy[0:-1], self.energy[1:])
            self.assertEqual(
                out[0].shape,
                self.energy[0:-1].shape,
                f"{name} emission_lines change shape with flux",
            )
            self.assertEqual(
                out[1].shape,
                self.energy[0:-1].shape,
                f"{name} emission_lines change shape with energies",
            )
