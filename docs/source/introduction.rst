Introduction
============

**jaxspec** is built to work on top of three core libraries

* `JAX <https://github.com/google/jax#what-is-jax>`_ : a
  NumPy-like library for automatic differentiation and accelerated
  numerical computing on CPUs and GPUs.
* `haiku <https://github.com/deepmind/dm-haiku#what-is-haiku>`_ : a lightweight library for neural network modules and stateful
  transformations.
* `numpyro <https://github.com/pyro-ppl/numpyro#what-is-numpyro>`_ : a JAX-based library for Bayesian inference.

Basically, the use of *JAX* as backend allows our models to be differentiable and computable on accelerators, *haiku* allows for easy parameters tracking when assembling complex models from the various components we propose, and *numpyro* gives access to appropriate samplers such as the No U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).