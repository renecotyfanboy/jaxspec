# Welcome to jaxspec's documentation!

`jaxspec` is a pure Python library for statistical inference on X-ray spectra. It allows to simply build spectral model
by combining components, and fit it to one or multiple observed spectra using Bayesian approaches. Since `jaxspec` is written using
[`JAX`](https://github.com/google/jax#what-is-jax), all inference problems are just-in-time compiled and can be run on CPU or GPU.


!!! warning
    `jaxspec` is in an early state of its existence, expect **bugs**, **bad documentation** and **breaking changes**. But this is also
    a very exciting time to learn how to use it and contribute to its future shape. You are very welcome to contribute and raise any issues you find.

## Getting started

<div class="grid">

  <a href="frontpage/installation/" class="card" style="font-size: 1.2em;">🛠️ Installation</a>
  <a href="examples/fitting_example/" class="card" style="font-size: 1.2em;">🚀 Quickstart</a>
  <a href="examples/" class="card" style="font-size: 1.2em;">📚 Examples</a>
  <a href="contribute/" class="card" style="font-size: 1.2em;">🤝 Contribute</a>

</div>

## How does it work?

`jaxspec` is built to work on top of two core libraries

* [`JAX`](https://github.com/google/jax#what-is-jax) : a
  NumPy-like library for automatic differentiation and accelerated
  numerical computing on CPUs and GPUs.
* [`numpyro`](https://github.com/pyro-ppl/numpyro#what-is-numpyro) : a JAX-based library for Bayesian inference.


Basically, the use of `JAX` as backend allows our models to be differentiable and computable on accelerators, and `numpyro`
gives access to appropriate samplers such as the No U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).

## References

* [The No-U-Turn Sampler](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) by Matthew D. Hoffman & Andrew Gelman
* [Statistical Aspects of X-ray Spectral Analysis](https://ui.adsabs.harvard.edu/abs/2023arXiv230905705B/abstract) by Johannes Buchner & Peter Boorman
* [Bayesian Modeling and Computation](https://bayesiancomputationbook.com/welcome.html) by Osvaldo A Martin , Ravin Kumar & Junpeng Lao

