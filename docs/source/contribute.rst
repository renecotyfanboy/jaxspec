Contribute
==========

Making your own model component
-------------------------------

If you want to make your own model component, you can implement it by extending the :class:`AdditiveComponent <jaxspec.model.abc.AdditiveComponent>` or
:class:`MultiplicativeComponent <jaxspec.model.abc.MultiplicativeComponent>` class. Imagine you want to implement some fancy axion model stuff with the following spectrum:

.. math::
    \mathcal{M}(E) = K \exp \left( -fE \right) ~ \sin ~\omega E

It can be done easily as follows :

.. code-block::

    class Fancystuff(AdditiveComponent):

        def __call__(self, energy):

            # These lines tell haiku that omega, f and norm should be considered as external and fittable parameters
            omega = hk.get_parameter("omega", [], init=hk.initializers.Constant(1))
            f = hk.get_parameter('f', [], init=Constant(1))
            norm = hk.get_parameter('norm', [], init=Constant(1))

            # The spectrum is computed using only functions from jax.numpy (jnp)
            spectrum = norm*jnp.exp(-f*energy)*jnp.sin(omega*energy)

            return spectrum