# Table models (`atable`, `mtable`, `etable`)

Some models are too expensive to evaluate on the fly and ship as a grid of
precomputed spectra. XSPEC reads those through `atable{file}`, `mtable{file}`
and `etable{file}`; `jaxspec` reads the same [OGIP 92-009](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_92_009/ogip_92_009.html)
FITS files with [`ATable`][jaxspec.model.tabulated.ATable],
[`MTable`][jaxspec.model.tabulated.MTable] and
[`ETable`][jaxspec.model.tabulated.ETable]. The interpolation is written in
JAX, so a table model is differentiable and vmappable like any other component.

## Loading a table

Table models need a file path, so they are not part of the component registry:

```python
from jaxspec.model.tabulated import ATable

reflionx = ATable("/path/to/reflionx.mod")
```

The fittable parameters come from the file's `PARAMETERS` extension. FITS names
are free-form (`Fe/solar`, `log T`) while prior keys are dotted paths, so names
are normalised to valid path segments:

```python
reflionx.table_parameters
# {'Fe_solar': 'Fe/solar', 'Gamma': 'Gamma', 'Xi': 'Xi'}
```

Keys are the prior names, values the original FITS names. `ATable` adds `norm`,
`z` and `escale` appear when the header declares `REDSHIFT` or `ESCALE`.

## Fitting with a table model

A table component composes with `*` and `+` like any other, and every
parameter needs a prior entry, including the ones read from the file:

```python
import numpyro.distributions as dist
from jaxspec.fit import MCMCFitter
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model = Tbabs() * (Powerlaw() + ATable("/path/to/reflionx.mod"))

prior = {
    "spectrum.tbabs_1.nh":       dist.Uniform(0, 1),
    "spectrum.powerlaw_1.alpha": dist.Uniform(1, 3),
    "spectrum.powerlaw_1.norm":  dist.LogUniform(1e-5, 1e-2),
    # from the table file
    "spectrum.atable_1.Fe_solar": dist.Uniform(0.5, 5.0),
    "spectrum.atable_1.Gamma":    dist.Uniform(1.5, 3.0),
    "spectrum.atable_1.Xi":       dist.LogUniform(30.0, 5000.0),
    "spectrum.atable_1.z":        0.032,          # a fixed value freezes it
    "spectrum.atable_1.norm":     dist.LogUniform(1e-8, 1e-4),
}

fitter = MCMCFitter(model, prior, observations)
```

!!! tip "Keep priors inside the tabulated grid"
    Values outside the grid are clipped to the nearest node, which flattens the
    likelihood there. Bound each prior by the parameter's own grid, as above
    for reflionx's $\xi \in [10, 10^4]$ and $\Gamma \in [1.4, 3.3]$.

!!! tip "Give the component a meaningful name"
    Components are named after their class, so every table shows up as
    `atable_1`, `mtable_1`, ... Subclass the table class to pick the name
    yourself, and bake the file path into `__init__` so the table behaves like
    any built-in component:

    ```python
    from jaxspec.model.tabulated import MTable

    class Reflection(MTable):
        def __init__(self):
            super().__init__("/path/to/reflection.mod")

    model = Reflection() * Powerlaw()
    # parameters live under "spectrum.reflection_1.<name>"
    ```

## Redshift and energy scale

With `REDSHIFT`, the `z` parameter reads the table at $E(1+z)$ and an additive
table also carries the $1/(1+z)$ time-dilation factor. With `ESCALE`, the
`escale` parameter stretches the tabulated energies as $E/E_\text{scale}$, with
no flux factor. Both compose as $E(1+z)/E_\text{scale}$, matching XSPEC.

## Trimming a table to your band

Tables often span far more energy than an analysis uses (reflionx covers 0.1 to
1000 keV). `energy_band` crops the tabulated bins at load time, which shrinks
the constant embedded in the compiled program:

```python
atable = ATable("/path/to/reflionx.mod", energy_band=(0.3, 12.0))
```

Cropped copies are cached and shared between instances. Leave margin when `z` or
`escale` are free: bins falling outside the retained range contribute zero flux.

## Multiplicative tables

`MTable` reads a table of dimensionless factors. `ETable` reads the same thing
as an optical depth and applies $e^{-\tau}$.

```python
from jaxspec.model.tabulated import ETable

model = ETable("/path/to/absorption.mod") * Powerlaw()
```

Neither takes a `norm`. An additive table's bins are redistributed
proportionally to overlap; a multiplicative table's are averaged over the bin
width. Bins outside the tabulated energies use the file's `LOELIMIT` /
`HIELIMIT` when present, 1.0 otherwise.

## Differences from XSPEC

Two behaviours deliberately depart from XSPEC 12.15.1 to keep a sampler alive:

- **Out-of-grid parameters are clipped** to the nearest node, where XSPEC
  raises a hard error.
- **Bins partially covering the low edge** of the table keep their covered
  fraction, where XSPEC discards them entirely (but keeps the high-edge part).

Both are only visible at the edges of a table's coverage.
