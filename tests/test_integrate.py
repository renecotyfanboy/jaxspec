import chex
import jax.numpy as jnp
import haiku as hk

chex.set_n_cpu_devices(n=2)


class TestIntegrationMethods(chex.TestCase):

    def setUp(self):
        from jaxspec.model.integrate import IntegrateABC
        self.methods = IntegrateABC.__subclasses__()

        @hk.to_module
        def integrand(x):
            # Dummy sin module for testing proper embedding of methods with haiku
            return jnp.sin(x)

        self.integrand = integrand

    @chex.all_variants
    def test_all_methods(self):
        """
        Test all integration methods work properly and ensure 1e-5 relative precision with default settings
        """

        for Method in self.methods:

            @hk.testing.transform_and_run(jax_transform=self.variant)
            def relative_precision():
                """
                Relative precision of the integration of a sinus between pi and 2 pi
                """

                f = Method(self.integrand())
                return jnp.abs(f(jnp.pi, 2*jnp.pi) + 2)/2

            out = relative_precision()
            msg = f'{Method.__name__} does not guarantee a relative precision of 1e-5 with default settings, increase default n_points'
            self.assertAlmostEqual(out, jnp.zeros_like(out), delta=1e-5, msg=msg)
