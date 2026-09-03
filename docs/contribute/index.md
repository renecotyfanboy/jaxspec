# Contributing to jaxspec

## Install `jaxspec` repository locally

Adding content to the `jaxspec`'s source code requires you to clone the repository locally. This is done by running the
following command in your terminal:

```bash
git clone https://github.com/renecotyfanboy/jaxspec
cd jaxspec
```

## Set up a clean environment

`jaxspec` uses [uv](https://docs.astral.sh/uv/) to manage its dependencies and lock file.
We recommend starting from a fresh Python environment. `jaxspec` supports Python 3.11 and
3.12; if you use conda, create the environment with

```bash
conda create -n jaxspec python=3.12
conda activate jaxspec
```

To install uv, run the following in the (`jaxspec`) environment:

```bash
pip install uv
```

Then install `jaxspec` and every dependency group, from the directory where you cloned the
repository:

```bash
uv sync --all-groups --frozen
```

`--frozen` installs exactly the versions recorded in the tracked `uv.lock`, which is what
CI does — so your environment reproduces CI's. Drop the flag (and commit the updated lock)
only when you intend to change a dependency.

## Running the tests

```bash
uv run pytest -m "not slow"     # fast subset, about a minute
uv run pytest                   # everything, including the multi-minute inference suites
```

## Code quality

We use [ruff](https://docs.astral.sh/ruff/) to enforce code quality standards, which proposes both a linter and a
formatter. They are set up with the pre-commit hooks (see below).

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run the linter and formatter automatically before each commit.
All the hooks are defined in `.pre-commit-config.yaml` and can be run manually with

```bash
uv run pre-commit run --all-files
```

If you want to install the pre-commit hooks so they run automatically, use

```bash
uv run pre-commit install
```

## Building the documentation

```bash
uv run mkdocs serve
```
