# Welcome to jaxspec's documentation!

`jaxspec` is a pure Python library for statistical inference on X-ray spectra. It allows to simply build spectral model
by combining components, and fit it to one or multiple observed spectra using Bayesian approaches. Since `jaxspec` is written using
[`JAX`](https://github.com/google/jax#what-is-jax), all inference problems are just-in-time compiled and can be run on CPU or GPU.


!!! warning
    `jaxspec` is in an early state of its existence, expect **bugs**, **bad documentation** and **breaking changes**. But this is also
    a very exciting time to learn how to use it and contribute to its future shape. You are very welcome to contribute and raise any issues you find.

## Getting started


<div class="grid cards" markdown>

-   <a class="card-link" href="frontpage/installation/" target="_blank" rel="noreferrer">
        <span class="card-title center">🛠️ Installation</span>
    </a>
-   <a class="card-link" href="examples/fitting_example/" target="_blank" rel="noreferrer">
    <span class="card-title center">🚀 Quickstart</span>
    </a>
-   <a class="card-link" href="examples/" target="_blank" rel="noreferrer">
    <span class="card-title center">📚 Examples</span>
    </a>
-   <a class="card-link" href="contribute/" target="_blank" rel="noreferrer">
    <span class="card-title center">🤝 Contribute</span>
    </a>
</div>

## How does it work?

`jaxspec` is built to work on top of two core libraries

* [`JAX`](https://github.com/google/jax#what-is-jax) : a
  NumPy-like library for automatic differentiation and accelerated
  numerical computing on CPUs and GPUs.
* [`numpyro`](https://github.com/pyro-ppl/numpyro#what-is-numpyro) : a JAX-based library for Bayesian inference.


Basically, the use of `JAX` as backend allows our models to be differentiable and computable on accelerators, and `numpyro`
gives access to appropriate samplers such as the No U-Turn Sampler (NUTS) and Hamiltonian Monte Carlo (HMC).

## Citation

If you use `jaxspec` in your research, please consider citing the following article

```
@ARTICLE{2024A&A...690A.317D,
       author = {{Dupourqu{\'e}}, S. and {Barret}, D. and {Diez}, C.~M. and {Guillot}, S. and {Quintin}, E.},
        title = "{jaxspec: A fast and robust Python library for X-ray spectral fitting}",
      journal = {\aap},
     keywords = {methods: data analysis, methods: statistical, X-rays: general},
         year = 2024,
        month = oct,
       volume = {690},
          eid = {A317},
        pages = {A317},
          doi = {10.1051/0004-6361/202451736},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...690A.317D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## References

* [The No-U-Turn Sampler](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) by Matthew D. Hoffman & Andrew Gelman
* [Statistical Aspects of X-ray Spectral Analysis](https://ui.adsabs.harvard.edu/abs/2023arXiv230905705B/abstract) by Johannes Buchner & Peter Boorman
* [Bayesian Modeling and Computation](https://bayesiancomputationbook.com/welcome.html) by Osvaldo A Martin , Ravin Kumar & Junpeng Lao

