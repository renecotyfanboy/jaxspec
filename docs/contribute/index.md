## Install `jaxspec` repository locally

Adding content to the `jaxspec`'s source code requires you to clone the repository locally. This is done by running the
following command in your terminal:

``` bash
git clone https://github.com/renecotyfanboy/jaxspec
cd jaxspec
```

## Set up a clean environment

`jaxspec` uses [Poetry](https://python-poetry.org/) to manage its dependencies. We recommand to start from a fresh
Python environment, and install Poetry. If you use conda, you can create a new environment with

``` bash
conda create -n jaxspec python=3.10
conda activate jaxspec
```

To install Poetry, run the following in the (`jaxspec`) environment:

``` bash
pip install poetry
```

Then, to install `jaxspec`'s dependencies, run the following command in the directory where you cloned the repository:

``` bash
poetry install
```

# Code quality

We use [ruff](https://docs.astral.sh/ruff/) to enforce code quality standards, which proposes both a linter and a
formatter. They are set up with the pre-commit hooks (see below).

## Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run the linter and formatter automatically before each commit.
All the hooks are defined in `.pre-commit-config.yaml` and can be run manually with

``` bash
poetry run pre-commit run --all-files
```

If you want to install the pre-commit hooks so they run automatically, use

``` bash
poetry run pre-commit install
```