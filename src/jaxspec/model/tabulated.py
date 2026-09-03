"""XSPEC-style additive and multiplicative OGIP table models.

[`ATable`][jaxspec.model.tabulated.ATable], [`MTable`][jaxspec.model.tabulated.MTable],
and [`ETable`][jaxspec.model.tabulated.ETable] load the OGIP 92-009 FITS format and
interpolate it over model parameters and requested energy bins. They remain
differentiable with respect to model parameters and bin edges. Import them explicitly;
they require a table path and therefore are not included in ``jaxspec.model.list``.
"""

from __future__ import annotations

import keyword
import os
import re

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from .abc import AdditiveComponent, MultiplicativeComponent
from .instrument import redistribute


@dataclass(frozen=True, eq=False)
class _OGIPTable:
    """Static, cached arrays loaded from one OGIP 92-009 table-model file."""

    name: str
    additive: bool
    redshift: bool
    escale: bool
    lo_limit: float | None
    hi_limit: float | None
    int_names: tuple[str, ...]  # raw FITS names; sanitized per component in _init_table
    add_names: tuple[str, ...]
    initial: tuple[float, ...]
    methods: tuple[int, ...]
    grids: tuple[np.ndarray, ...]
    e_edges: np.ndarray
    spectra: np.ndarray  # (1 + n_add, *grid_shape, n_energy_bins)


def _sanitize_parameter_name(raw: str, taken: set[str]) -> str:
    """Turn a FITS parameter name into a usable nnx attribute / prior-key segment.

    ``.`` breaks the dotted-path grammar, ``[``/``]`` are parsed as an observation
    scope, and a leading underscore marks non-parameter state — so everything
    non-alphanumeric collapses to ``_``. Collisions (with reserved names such as
    ``norm``/``z``/``escale``, or between sanitized names) get a numeric suffix.
    """
    name = re.sub(r"\W+", "_", raw.strip()).strip("_")
    if not name:
        name = "param"
    if name[0].isdigit():
        name = f"p{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    base, k = name, 2
    while name in taken:
        name = f"{base}_{k}"
        k += 1
    taken.add(name)
    return name


def _file_signature(path: str) -> tuple[int, int]:
    """(mtime_ns, size) of the file — part of the cache key, so a table regenerated at
    the same path in a long-lived session is reloaded instead of served stale."""
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _load_ogip_table(path: str, signature: tuple[int, int]) -> _OGIPTable:
    """Read and validate an OGIP 92-009 table-model FITS file, once per process."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        primary = hdul[0].header
        if primary.get("HDUCLAS1", "").strip() != "XSPEC TABLE MODEL":
            raise ValueError(
                f"{path} does not declare HDUCLAS1='XSPEC TABLE MODEL' — not an OGIP "
                f"92-009 table-model file."
            )
        if primary.get("NXFLTEXP", 1) > 1:
            raise NotImplementedError(
                f"{path} tabulates NXFLTEXP={primary['NXFLTEXP']} spectra per grid "
                f"point (XFLT-selected); only single-spectrum tables are supported."
            )
        if "NNPTFILE" in primary:
            raise NotImplementedError(
                f"{path} points at a neural-network emulator (NNPTFILE); only "
                f"interpolated tables are supported."
            )

        additive = bool(primary.get("ADDMODEL", False))
        redshift = bool(primary.get("REDSHIFT", False))
        escale = bool(primary.get("ESCALE", False))
        lo_limit = float(primary["LOELIMIT"]) if "LOELIMIT" in primary else None
        hi_limit = float(primary["HIELIMIT"]) if "HIELIMIT" in primary else None
        model_name = str(primary.get("MODLNAME", Path(path).stem)).strip()

        parameters = hdul["PARAMETERS"]
        n_int = int(parameters.header["NINTPARM"])
        n_add = int(parameters.header.get("NADDPARM", 0))
        rows = parameters.data
        if len(rows) != n_int + n_add:
            raise ValueError(
                f"{path}: PARAMETERS has {len(rows)} rows but NINTPARM+NADDPARM={n_int + n_add}."
            )

        names = tuple(str(row["NAME"]).strip() for row in rows)
        initial = tuple(float(row["INITIAL"]) for row in rows)

        methods, grids = [], []
        for row in rows[:n_int]:
            n_values = int(row["NUMBVALS"])
            grid = np.asarray(np.atleast_1d(row["VALUE"])[:n_values], dtype=np.float64)
            if grid.size != n_values or n_values < 1:
                raise ValueError(
                    f"{path}: parameter {row['NAME']!r} declares NUMBVALS={n_values} "
                    f"but tabulates {grid.size} values."
                )
            if np.any(np.diff(grid) <= 0):
                raise ValueError(
                    f"{path}: VALUE grid of parameter {row['NAME']!r} is not strictly increasing."
                )
            method = int(row["METHOD"])
            if method == 1 and grid[0] <= 0:
                raise ValueError(
                    f"{path}: parameter {row['NAME']!r} requests logarithmic "
                    f"interpolation (METHOD=1) over a non-positive grid."
                )
            methods.append(method)
            grids.append(grid)

        energies = hdul["ENERGIES"].data
        e_lo = np.asarray(energies["ENERG_LO"], dtype=np.float64)
        e_hi = np.asarray(energies["ENERG_HI"], dtype=np.float64)
        if not np.allclose(e_lo[1:], e_hi[:-1], rtol=1e-4):
            raise ValueError(f"{path}: ENERGIES bins are not contiguous.")
        e_edges = np.ascontiguousarray(np.concatenate([e_lo, e_hi[-1:]]))
        if np.any(np.diff(e_edges) <= 0):
            raise ValueError(f"{path}: ENERGIES bins are not strictly increasing.")

        spectra_hdu = hdul["SPECTRA"]
        grid_shape = tuple(grid.size for grid in grids)
        n_rows_expected = int(np.prod(grid_shape))
        if len(spectra_hdu.data) != n_rows_expected:
            raise ValueError(
                f"{path}: SPECTRA has {len(spectra_hdu.data)} rows, expected "
                f"{n_rows_expected} (product of NUMBVALS)."
            )
        # FITS rows use C-order Cartesian parameter indexing.
        columns = ["INTPSPEC"] + [f"ADDSP{i + 1:03d}" for i in range(n_add)]
        spectra = np.stack(
            [
                np.asarray(spectra_hdu.data[column], dtype=np.float64).reshape(
                    *grid_shape, e_lo.size
                )
                for column in columns
            ]
        )

    return _OGIPTable(
        name=model_name,
        additive=additive,
        redshift=redshift,
        escale=escale,
        lo_limit=lo_limit,
        hi_limit=hi_limit,
        int_names=names[:n_int],
        add_names=names[n_int:],
        initial=initial,
        methods=tuple(methods),
        grids=tuple(grids),
        e_edges=e_edges,
        spectra=np.ascontiguousarray(spectra),
    )


@lru_cache(maxsize=16)
def _restricted_ogip_table(
    path: str, signature: tuple[int, int], e_min: float, e_max: float
) -> _OGIPTable:
    """Energy-cropped view of a table, cached so same-band instances share arrays."""
    table = _load_ogip_table(path, signature)
    edges = table.e_edges
    keep = (edges[1:] > e_min) & (edges[:-1] < e_max)
    if not np.any(keep):
        raise ValueError(
            f"energy_band=({e_min}, {e_max}) keV does not overlap the tabulated "
            f"energies [{edges[0]:.4g}, {edges[-1]:.4g}] keV of {path}."
        )
    (indices,) = np.nonzero(keep)
    i0, i1 = int(indices[0]), int(indices[-1]) + 1
    return _OGIPTable(
        name=table.name,
        additive=table.additive,
        redshift=table.redshift,
        escale=table.escale,
        lo_limit=table.lo_limit,
        hi_limit=table.hi_limit,
        int_names=table.int_names,
        add_names=table.add_names,
        initial=table.initial,
        methods=table.methods,
        grids=table.grids,
        e_edges=np.ascontiguousarray(table.e_edges[i0 : i1 + 1]),
        spectra=np.ascontiguousarray(table.spectra[..., i0:i1]),
    )


class _TableComponent:
    """Shared machinery of the three table-model components.

    Mixin only — the concrete classes pick their algebra by also subclassing
    ``AdditiveComponent`` or ``MultiplicativeComponent``.
    """

    _table: _OGIPTable
    _int_names: tuple[str, ...]
    _add_names: tuple[str, ...]

    def _init_table(self, path: str | Path, energy_band: tuple[float, float] | None):
        path = str(Path(path).expanduser().resolve())
        signature = _file_signature(path)
        if energy_band is None:
            table = _load_ogip_table(path, signature)
        else:
            e_min, e_max = map(float, energy_band)
            table = _restricted_ogip_table(path, signature, e_min, e_max)

        expect_additive = isinstance(self, AdditiveComponent)
        if table.additive != expect_additive:
            kind = "an additive (atable)" if table.additive else "a multiplicative (mtable)"
            wanted = "ATable" if table.additive else "MTable/ETable"
            raise ValueError(
                f"{path} is {kind} table model (ADDMODEL={table.additive}); "
                f"load it with {wanted} instead."
            )

        self._table = table
        # Prevent FITS parameter names from shadowing the component API.
        taken = {attr for attr in dir(type(self)) if not attr.startswith("_")}
        taken |= {"norm", "z", "escale"}
        self._int_names = tuple(_sanitize_parameter_name(raw, taken) for raw in table.int_names)
        self._add_names = tuple(_sanitize_parameter_name(raw, taken) for raw in table.add_names)
        for name, value in zip(self._int_names + self._add_names, table.initial):
            setattr(self, name, nnx.Param(float(value)))
        if table.escale:
            self.escale = nnx.Param(1.0)
        if table.redshift:
            self.z = nnx.Param(0.0)

    @property
    def table_parameters(self) -> dict[str, str]:
        """Mapping from nnx attribute name (= prior-key segment) to the raw FITS name."""
        table = self._table
        return dict(zip(self._int_names + self._add_names, table.int_names + table.add_names))

    def _energy_scale(self):
        """Return ``(edge_scale, one_plus_z)``: table energies are read at
        ``E * (1 + z) / escale`` and additive fluxes carry a ``1/(1+z)`` factor."""
        table = self._table
        one_plus_z = 1.0 + jnp.asarray(self.z) if table.redshift else jnp.asarray(1.0)
        scale = one_plus_z / jnp.asarray(self.escale) if table.escale else one_plus_z
        return scale, one_plus_z

    def _interpolated_table_values(self):
        """Multilinear interpolation of the tabulated spectra at the current parameters.

        Returns the combined ``INTPSPEC + sum_i q_i * ADDSP_i`` array over the table's
        native energy bins. Parameters outside the tabulated grid are clipped to the
        edge (XSPEC raises a hard error there; clipping keeps MCMC alive and mirrors
        what APEC does with kT). METHOD=1 parameters interpolate with logarithmic
        weights in the parameter value; the spectra are always combined linearly.
        """
        table = self._table
        spectra = jnp.asarray(table.spectra)

        indices, weights = [], []
        for name, method, grid in zip(self._int_names, table.methods, table.grids):
            value = jnp.clip(jnp.asarray(getattr(self, name)), grid[0], grid[-1])
            if grid.size == 1:
                indices.append((jnp.zeros((), dtype=int), jnp.zeros((), dtype=int)))
                weights.append(jnp.zeros(()))
                continue
            nodes = jnp.asarray(np.log(grid) if method == 1 else grid)
            x = jnp.log(value) if method == 1 else value
            i = jnp.clip(jnp.searchsorted(nodes, x, side="right") - 1, 0, grid.size - 2)
            w = jnp.clip((x - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)
            indices.append((i, i + 1))
            weights.append(w)

        interpolated = 0.0
        for corner in product((0, 1), repeat=len(weights)):
            corner_weight = 1.0
            corner_index = []
            for on_high, (i_lo, i_hi), w in zip(corner, indices, weights):
                corner_weight = corner_weight * (w if on_high else 1.0 - w)
                corner_index.append(i_hi if on_high else i_lo)
            interpolated = interpolated + corner_weight * spectra[(slice(None), *corner_index)]

        combined = interpolated[0]
        if self._add_names:
            q = jnp.stack([jnp.asarray(getattr(self, name)) for name in self._add_names])
            combined = combined + q @ interpolated[1:]
        return combined


class ATable(_TableComponent, AdditiveComponent):
    r"""Additive OGIP table model — the equivalent of XSPEC's `atable{file}`.

    The tabulated spectra are per-bin integrated photon fluxes
    $\left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$; they are interpolated
    multilinearly over the parameter grid and redistributed onto the requested bins
    proportionally to overlap (integral preserving):

    $$ \mathcal{M}(E_\min, E_\max) = \frac{\text{norm}}{1+z}
    \int_{E_\min (1+z)/E_{\text{scale}}}^{E_\max (1+z)/E_{\text{scale}}}
    \text{d}E ~ S_{\vec{p}}(E) $$

    !!! abstract "Constructor arguments"
        * `path`: OGIP 92-009 table-model FITS file.
        * `energy_band`: optional `(e_min, e_max)` in keV cropping the tabulated energy
          bins at load time (cropped copies are cached and shared between instances).
          When fitting a redshift or energy scale, widen the band so
          $E \times (1+z)/E_{\text{scale}}$ stays covered, or flux is silently lost.

    !!! abstract "Parameters"
        Read from the file's `PARAMETERS` extension: one fittable parameter per row,
        renamed to valid dotted-path segments (see `table_parameters` for the mapping),
        plus `escale` when the file sets `ESCALE`, `z` when it sets `REDSHIFT`, and
        `norm` (normalisation of the spectrum). Every parameter needs a prior entry
        (a fixed value is enough to freeze it).

    !!! warning
        Parameter values outside the tabulated grid are clipped to the nearest grid
        node (XSPEC raises a hard error instead) — keep priors inside the grid.
        Bins outside the tabulated energy range get zero flux (XSPEC ignores
        `LOELIMIT`/`HIELIMIT` for additive tables, and so does jaxspec). Energy flux
        uses a mid-bin approximation, like every purely bin-integrated component.
        The table is baked into the compiled program as a constant: instances built
        from the same file share one in-memory copy, but very large tables cost their
        full size once per compiled fit.
        Enable float64 (`jax.config.update("jax_enable_x64", True)`) when the per-bin
        fluxes span many decades: the rebinning cumulative sum saturates single
        precision and silently corrupts the faint tail otherwise.
    """

    def __init__(self, path: str | Path, energy_band: tuple[float, float] | None = None):
        self._init_table(path, energy_band)
        self.norm = nnx.Param(1.0)

    def integrated_continuum(self, e_low, e_high):
        table = self._table
        spectrum = self._interpolated_table_values()
        scale, one_plus_z = self._energy_scale()
        flux = redistribute(
            spectrum,
            jnp.asarray(table.e_edges[:-1]),
            jnp.asarray(table.e_edges[1:]),
            e_low * scale,
            e_high * scale,
        )
        return jnp.asarray(self.norm) * flux / one_plus_z


class MTable(_TableComponent, MultiplicativeComponent):
    r"""Multiplicative OGIP table model — the equivalent of XSPEC's `mtable{file}`.

    The tabulated values are dimensionless multiplicative factors; the factor applied
    to an output bin is their width-weighted average over the covered table bins:

    $$ \mathcal{M}(E_\min, E_\max) =
    \frac{1}{E_\max - E_\min}\int_{E_\min}^{E_\max} \text{d}E ~ S_{\vec{p}}(E) $$

    evaluated at $E \times (1+z)/E_{\text{scale}}$ when the file declares redshift or
    energy-scale parameters (no $1/(1+z)$ flux factor for multiplicative tables).

    !!! abstract "Constructor arguments"
        * `path`: OGIP 92-009 table-model FITS file.
        * `energy_band`: optional `(e_min, e_max)` in keV cropping the tabulated energy
          bins at load time (cropped copies are cached and shared between instances).
          When fitting a redshift or energy scale, widen the band so
          $E \times (1+z)/E_{\text{scale}}$ stays covered, or the out-of-range factor
          silently applies instead.

    !!! abstract "Parameters"
        Read from the file's `PARAMETERS` extension: one fittable parameter per row,
        renamed to valid dotted-path segments (see `table_parameters` for the mapping),
        plus `escale` when the file sets `ESCALE` and `z` when it sets `REDSHIFT`.
        Every parameter needs a prior entry (a fixed value is enough to freeze it).

    !!! warning
        Parameter values outside the tabulated grid are clipped to the nearest grid
        node (XSPEC raises a hard error instead) — keep priors inside the grid.
        Bins fully outside the tabulated energy range use `LOELIMIT`/`HIELIMIT` from
        the file when present, and 1.0 otherwise. A bin partially covering the low end
        of the table averages over its covered fraction; XSPEC 12.15.1 drops such bins
        entirely (asymmetrically with the high end) — jaxspec keeps the symmetric,
        integral-consistent behaviour. The table is baked into the compiled program as
        a constant: instances built from the same file share one in-memory copy.
        Enable float64 (`jax.config.update("jax_enable_x64", True)`) when the factors
        span many decades: the rebinning cumulative sum saturates single precision.
    """

    def __init__(self, path: str | Path, energy_band: tuple[float, float] | None = None):
        self._init_table(path, energy_band)

    def _transform(self, values):
        """Hook mapping combined table values to the multiplicative factor."""
        return values

    def _out_of_range_factor(self, below, above):
        table = self._table
        lo = 1.0 if table.lo_limit is None else table.lo_limit
        hi = 1.0 if table.hi_limit is None else table.hi_limit
        return jnp.where(below, lo, jnp.where(above, hi, 1.0))

    def _factor(self, e_low, e_high, n_points=2):
        # Average over covered table widths rather than point-sampling the factor.
        table = self._table
        values = self._interpolated_table_values()
        table_lo = jnp.asarray(table.e_edges[:-1])
        table_hi = jnp.asarray(table.e_edges[1:])
        widths = jnp.asarray(np.diff(table.e_edges))

        scale, _ = self._energy_scale()
        lo, hi = e_low * scale, e_high * scale
        covered_integral = redistribute(values * widths, table_lo, table_hi, lo, hi)
        covered_width = redistribute(widths, table_lo, table_hi, lo, hi)
        in_range = covered_width > 0
        average = self._transform(covered_integral / jnp.where(in_range, covered_width, 1.0))
        # Limits are final factors, not optical depths for ETable.
        out_value = self._out_of_range_factor(hi <= table_lo[0], lo >= table_hi[-1])
        factor = jnp.where(in_range, average, out_value)
        # A clipped shift can collapse both bin edges.
        return jnp.where(e_high > e_low, factor, 0.0)

    def factor(self, energy):
        """Pointwise factor (step function over the table bins) for direct calls;
        the fit hot path goes through the exact bin-averaged ``_factor`` instead."""
        table = self._table
        values = self._interpolated_table_values()
        edges = jnp.asarray(table.e_edges)
        scaled = energy * self._energy_scale()[0]
        i = jnp.clip(jnp.searchsorted(edges, scaled, side="right") - 1, 0, values.shape[-1] - 1)
        out_value = self._out_of_range_factor(scaled < edges[0], scaled >= edges[-1])
        return jnp.where(
            (scaled >= edges[0]) & (scaled < edges[-1]), self._transform(values[i]), out_value
        )


class ETable(MTable):
    r"""Exponential multiplicative OGIP table model — XSPEC's `etable{file}`.

    Identical to [`MTable`][jaxspec.model.tabulated.MTable] except the combined table
    value (additional parameters included) is mapped through

    $$ \mathcal{M}(E) = e^{-S_{\vec{p}}(E)} $$

    so the tabulated quantity is an optical depth (additional parameters combine
    linearly *before* the exponential, XSPEC's design point for column densities).
    `LOELIMIT`/`HIELIMIT` are final factors and are **not** exponentiated, matching
    XSPEC. See [`MTable`][jaxspec.model.tabulated.MTable] for the constructor
    arguments, parameter discovery and edge-behaviour caveats.
    """

    def _transform(self, values):
        return jnp.exp(-values)
