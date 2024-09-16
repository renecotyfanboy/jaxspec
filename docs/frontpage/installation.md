# Installation

`jaxspec` requires `Python > 3.9` to be installed. We recommend to start from a fresh `conda` environment using either
[`miniforge`](https://github.com/conda-forge/miniforge) (community driven) or [`miniconda`](https://docs.anaconda.com/miniconda/)
(built by Anaconda).

```
conda create -n jaxspec python=3.11
conda activate jaxspec
```

## Using `pip`

Once you are in an environment with a proper `Python` version, simply use pip to install `jaxspec`

```
pip install jaxspec
```

## From source

If you want to install the development version of `jaxspec`, you can clone the repository and install it from source.

```
git clone https://github.com/renecotyfanboy/jaxspec
cd jaxspec
pip install -e .
```

If you do not need to edit it, you can install it with `pip` directly

```
pip install git+https://github.com/renecotyfanboy/jaxspec --upgrade
```

## GPU support

The default `JAX` dependency in `jaxspec` does not include GPU support. However, it can easily be enabled by reinstalling
`JAX` with the proper wheels after installing `jaxspec`. In general, you should run

```
pip install "jax[cuda12]"
```

More details can be found in the [official `JAX` installation doc](https://jax.readthedocs.io/en/latest/installation.html#).