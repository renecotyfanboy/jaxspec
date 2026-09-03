# Table models (`atable`, `mtable`, `etable`)

Some models are too expensive to evaluate on the fly, so they ship as a grid of
precomputed spectra: a radiative-transfer calculation, an MHD simulation, an
atmosphere code. XSPEC reads those through `atable{file}`, `mtable{file}` and
`etable{file}`, and `jaxspec` reads the same [OGIP 92-009](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/general/ogip_92_009/ogip_92_009.html)
FITS files with [`ATable`][jaxspec.model.tabulated.ATable],
[`MTable`][jaxspec.model.tabulated.MTable] and
[`ETable`][jaxspec.model.tabulated.ETable].

The interpolation is written in JAX, so a table model is differentiable and
vmappable like every other component: gradient-based samplers, `fakeit`, and
posterior-predictive checks all work without special-casing.

## Loading a table

Table models are the only components that need a constructor argument, so they
are not importable from the component registry — point them at a file:

```python
from jaxspec.model.tabulated import ATable

reflionx = ATable("/path/to/reflionx.mod")
```

The fittable parameters come from the file's `PARAMETERS` extension. FITS
parameter names are free-form (`Fe/solar`, `log T`) while prior keys are dotted
paths, so names are normalised to valid path segments. Ask the component what
it ended up with:

```python
reflionx.table_parameters
# {'Fe_solar': 'Fe/solar', 'Gamma': 'Gamma', 'Xi': 'Xi'}
```

The keys are the names you use in a prior dict; the values are the original FITS
names. On top of those, `ATable` always adds `norm`, and both `z` and `escale`
appear when the file's header declares `REDSHIFT` or `ESCALE`.

## Fitting with a table model

A table component composes with `*` and `+` like any other, and every one of its
parameters needs a prior entry — including the ones that came from the file:

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
    A table only knows what it was computed for. Values outside the grid are
    clipped to the nearest node, which flattens the likelihood there and lets a
    sampler wander into a region carrying no information. Bound each prior by
    the parameter's own grid — the ranges above match reflionx's
    $\xi \in [10, 10^4]$ and $\Gamma \in [1.4, 3.3]$.

## Redshift and energy scale

When the file declares `REDSHIFT`, the extra `z` parameter shifts the table:
the model reads the table at $E(1+z)$, and an additive table additionally
carries the $1/(1+z)$ time-dilation factor. `ESCALE` adds an `escale`
parameter that stretches the tabulated energies the other way ($E/E_\text{scale}$,
no flux factor), for tables whose energy calibration is itself uncertain. Both
compose as $E(1+z)/E_\text{scale}$, matching XSPEC.

## Trimming a table to your band

Tables often span far more energy than any one analysis uses — reflionx covers
0.1 to 1000 keV. `energy_band` crops the tabulated bins at load time, which
shrinks the constant embedded in the compiled program:

```python
atable = ATable("/path/to/reflionx.mod", energy_band=(0.3, 12.0))
```

Cropped copies are cached and shared between instances, so building the same
band twice costs nothing. Leave margin when `z` or `escale` are free: the model
reads the table at $E(1+z)/E_\text{scale}$, and anything falling outside the
retained bins contributes zero flux rather than raising.

## Multiplicative and exponential tables

`MTable` reads a table of dimensionless factors, and `ETable` reads the same
thing as an optical depth, applying $e^{-\tau}$. Both refuse a file whose
header does not match, so a mismatched class fails at construction rather than
silently fitting the wrong algebra:

```python
from jaxspec.model.tabulated import ETable

model = ETable("/path/to/absorption.mod") * Powerlaw()
```

Neither takes a `norm`. Where an additive table's bins are redistributed
proportionally to overlap, a multiplicative table's are averaged over the bin
width — the two OGIP rebinning rules — and bins falling outside the tabulated
energies use the file's `LOELIMIT` / `HIELIMIT` when present, 1.0 otherwise.

## Building a table from your own grid

Any grid of precomputed spectra can be written as an OGIP table and fitted
through `ATable`. The essentials: one `PARAMETERS` row per parameter, an
`ENERGIES` extension of contiguous bins, and a `SPECTRA` extension whose rows
enumerate the parameter grid in C order (last parameter varying fastest), each
holding the spectrum **integrated over each energy bin** in photons/cm²/s.

```python
import numpy as np
from astropy.io import fits

edges = np.geomspace(0.1, 20.0, 501)             # 500 contiguous bins
slopes = np.linspace(1.2, 3.0, 19)               # one parameter, 19 grid points

# One row per grid point, in the same order as the grid. Each row holds the flux
# INTEGRATED over every bin, not the flux density sampled at a bin edge — here the
# exact integral of E**-s. Sampling instead of integrating biases the model by
# roughly half a bin width, which no amount of care downstream recovers.
spectra = np.stack(
    [(edges[1:] ** (1 - s) - edges[:-1] ** (1 - s)) / (1 - s) for s in slopes]
)

primary = fits.PrimaryHDU()
primary.header["MODLNAME"] = "mymodel"
primary.header["MODLUNIT"] = "photons/cm^2/s"
primary.header["ADDMODEL"] = True                 # additive -> ATable
primary.header["REDSHIFT"] = False
primary.header["HDUCLASS"] = "OGIP"
primary.header["HDUCLAS1"] = "XSPEC TABLE MODEL"
primary.header["HDUVERS1"] = "1.0.0"

parameters = fits.BinTableHDU.from_columns(
    [
        fits.Column(name="NAME", format="12A", array=np.array(["slope"])),
        fits.Column(name="METHOD", format="J", array=np.array([0])),      # 0 linear, 1 log
        fits.Column(name="INITIAL", format="E", array=np.array([2.0])),
        fits.Column(name="DELTA", format="E", array=np.array([0.01])),
        fits.Column(name="MINIMUM", format="E", array=np.array([slopes[0]])),
        fits.Column(name="BOTTOM", format="E", array=np.array([slopes[0]])),
        fits.Column(name="TOP", format="E", array=np.array([slopes[-1]])),
        fits.Column(name="MAXIMUM", format="E", array=np.array([slopes[-1]])),
        fits.Column(name="NUMBVALS", format="J", array=np.array([slopes.size])),
        fits.Column(name="VALUE", format=f"{slopes.size}E", array=slopes[None, :]),
    ],
    name="PARAMETERS",
)
parameters.header["NINTPARM"] = 1
parameters.header["NADDPARM"] = 0

energies = fits.BinTableHDU.from_columns(
    [
        fits.Column(name="ENERG_LO", format="E", array=edges[:-1]),
        fits.Column(name="ENERG_HI", format="E", array=edges[1:]),
    ],
    name="ENERGIES",
)

spectra_hdu = fits.BinTableHDU.from_columns(
    [
        fits.Column(name="PARAMVAL", format="1E", array=slopes[:, None]),
        fits.Column(name="INTPSPEC", format=f"{edges.size - 1}E", array=spectra),
    ],
    name="SPECTRA",
)

fits.HDUList([primary, parameters, energies, spectra_hdu]).writeto("mymodel.mod", overwrite=True)
```

Set `ADDMODEL = False` for a multiplicative or exponential table, and store
dimensionless factors (or optical depths, for `ETable`) instead of fluxes. The
loader validates the file on construction — mismatched row counts, non-monotone
grids and non-contiguous energy bins all raise with a message naming the
offending extension, rather than producing a quietly wrong model.

Two resolutions govern how faithful the result is, and both are yours to choose
when writing the file. Rebinning splits a tabulated bin proportionally, which
assumes the flux is uniform inside it, so tabulate on bins comfortably narrower
than your instrument's — the 0.5%-wide bins above reproduce a power law to
$10^{-3}$ on a response three times coarser. The parameter grid matters the same
way: interpolation between nodes is linear, so sample any direction the spectrum
responds to sharply densely enough that a straight line between neighbours is a
fair description.

## Precision and performance

The implementation was validated against XSPEC 12.15.1. Agreement is at
float64 round-off — a median relative deviation of $10^{-12}$ per bin, and
$10^{-15}$ on the band-integrated flux. It degrades to a few $10^{-7}$ in one
situation only: when a requested parameter falls within a float32 epsilon of a
tabulated node, because the `VALUE` grid is stored in single precision. That is
the precision floor of the file format, not of the interpolation.

Per-spectrum evaluation runs at parity with XSPEC's C++ (≈34 µs on a
1000-bin grid); the advantage is elsewhere:

| | `jaxspec` | XSPEC |
|---|---|---|
| One spectrum | 34 µs | 33 µs |
| Gradient, all parameters | 91 µs, independent of parameter count | $(n+1)$ evaluations |
| 512 parameter sets (vectorised) | 4–16 µs each | 30–42 µs each |

Because the gradient is computed by automatic differentiation, its cost does
not grow with the number of free parameters, whereas finite differences need
one extra evaluation each: parity around three parameters, roughly 7× faster at
twenty. This is what makes gradient-based samplers practical on table models.

!!! warning "Enable float64 for steep tables"
    Rebinning integrates the tabulated spectrum cumulatively, and in single
    precision that sum saturates when the per-bin fluxes span many decades,
    corrupting the faint tail. Enable float64 before fitting:

    ```python
    import numpyro
    numpyro.enable_x64()
    ```

## Differences from XSPEC

Two behaviours deliberately depart from XSPEC 12.15.1, both in the direction of
keeping a sampler alive rather than reproducing a quirk:

- **Out-of-grid parameters are clipped** to the nearest tabulated node. XSPEC
  raises a hard error, which would abort a fit whenever a proposal stepped
  outside the grid.
- **Bins partially covering the low edge** of the table keep their covered
  fraction. XSPEC discards them entirely while keeping the covered part at the
  high edge, an asymmetry that shows up as a small flux deficit at the soft end.

Both are only visible at the edges of a table's coverage; priors bounded by the
grid, and a band comfortably inside the tabulated energies, avoid them
entirely.
