<p align="center">
  <img src="https://raw.githubusercontent.com/renecotyfanboy/jaxspec/main/docs/logo/logo_small.svg" alt="Logo" width="100" height="100">
</p>

<h1 align="center">
  jaxspec
</h1>


[![PyPI - Version](https://img.shields.io/pypi/v/jaxspec?style=for-the-badge&logo=pypi&color=rgb(37%2C%20150%2C%20190))](https://pypi.org/project/jaxspec/)
[![Python package](https://img.shields.io/pypi/pyversions/jaxspec?style=for-the-badge)](https://pypi.org/project/jaxspec/)
[![Read the Docs](https://img.shields.io/readthedocs/jaxspec?style=for-the-badge)](https://jaxspec.readthedocs.io/en/latest/)
[![Codecov](https://img.shields.io/codecov/c/github/renecotyfanboy/jaxspec?style=for-the-badge)](https://app.codecov.io/gh/renecotyfanboy/jaxspec)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/jaxspec/shared_invite/zt-2cuxkdl2f-t0EEAKP~HBEHKvIUZJL2sg)

> :warning: **jaxspec is still in early release**: expect bugs, breaking API changes, undocumented features and lack of functionalities

`jaxspec` is an X-ray spectral fitting library built in pure Python. It can currently load an X-ray spectrum (in the OGIP standard), define a spectral model from the implemented components, and calculate the best parameters using state-of-the-art Bayesian approaches. It is built on top of JAX to provide just-in-time compilation and automatic differentiation of the spectral models, enabling the use of sampling algorithm such as NUTS.

`jaxspec` is written in pure Python, and has no dependancy to [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/), and can be installed directly using the `pip` command.

Documentation : https://jaxspec.readthedocs.io/en/latest/

## Installation

We recommend the users to start from a fresh Python 3.10 [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```
conda create -n jaxspec python=3.10
conda activate jaxspec
```

Once the environment is set up, you can install jaxspec directly from pypi

```
pip install jaxspec --upgrade
```
