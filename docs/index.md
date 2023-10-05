# Welcome to JAXspec's documentation!

**JAXspec** is a pure Python library for statistical inference on X-ray spectra. It allows to simply build spectral model 
by combining components, and fit it to one or multiple observed spectra. Various inference method are proposed, either 
frequentist or bayesian approaches. As it is written using **[JAX](https://github.com/google/jax#what-is-jax)**, every 
inference problem are just-in-time compiled and can be run on CPU or GPU.

## How does it work?

**JAXspec** is built to work on top of three core libraries

* **[JAX](https://github.com/google/jax#what-is-jax)** : a
  NumPy-like library for automatic differentiation and accelerated
  numerical computing on CPUs and GPUs.
* **[haiku](https://github.com/deepmind/dm-haiku#what-is-haiku)** : a lightweight library for neural network modules and stateful
  transformations.
* **[numpyro](https://github.com/pyro-ppl/numpyro#what-is-numpyro)** : a JAX-based library for Bayesian inference.


Basically, the use of **JAX** as backend allows our models to be differentiable and computable on accelerators, **haiku** 
allows for easy parameters tracking when assembling complex models from the various components we propose, and **numpyro** 
gives access to appropriate samplers such as the No U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).

