"""APEC plasma emission from precomputed AtomDB CIE tables.

Lines are integrated analytically over output bins. The continuum is broadened by FFT on a
log-energy grid and conservatively rebinned. The runtime supports JIT, batching and autodiff.
"""

from __future__ import annotations

import warnings

from functools import lru_cache
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from scipy.fft import next_fast_len

from ...util.online_storage import table_manager
from ..abc import AdditiveComponent

C_KMS = 299792.458
MU_C2_KEV = 931494.0  # atomic mass unit * c^2 [keV]
NORM_CONST = 1.0e14  # maps table emissivities to the XSPEC norm convention
TABLE_RESOURCE = "apec_cie_v3.1.3.npz"

# Thresholds for checkpointed line batches.
LINE_GRID_MIN_BATCHED = 2_000_000
LINE_GRID_PER_BATCH = 600_000
# Window used by ``k_window="auto"`` when the energy grid is traced before any concrete call.
DEFAULT_K_WINDOW = 128

# fmt: off
ELEMENT_SYMBOLS = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                   "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                   "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")  # index = Z - 1
# fmt: on
VAPEC_SYMBOLS = ("He", "C", "N", "O", "Ne", "Mg", "Al", "Si", "S", "Ar", "Ca", "Fe", "Ni")
VAPEC_IDX = np.array([ELEMENT_SYMBOLS.index(s) for s in VAPEC_SYMBOLS])
# The scalar APEC abundance scales the VAPEC metals except helium.
METAL_MASK = np.isin(
    np.arange(len(ELEMENT_SYMBOLS)),
    [ELEMENT_SYMBOLS.index(s) for s in VAPEC_SYMBOLS if s != "He"],
)


def table_is_available() -> bool:
    """Return whether the AtomDB table is usable: already cached, or fetchable."""
    if (Path(table_manager.abspath) / TABLE_RESOURCE).is_file():
        return True

    try:
        return table_manager.is_available(TABLE_RESOURCE)
    except Exception:  # no network, proxy, DNS failure...
        return False


@lru_cache(maxsize=1)
def load_apec_table() -> dict[str, np.ndarray]:
    """Load and cache the AtomDB table as NumPy arrays, downloading it on first use."""
    path = table_manager.fetch(TABLE_RESOURCE, progressbar=True)

    with np.load(path) as data:
        return {
            "E0": np.asarray(data["E0"], np.float64),
            "Zidx": np.asarray(data["Zidx"], np.int32),
            "Amass": np.asarray(data["Amass"], np.float64),
            "g": np.asarray(data["g"], np.float32),  # (L, nT), float32 halves memory
            "logTg": np.asarray(data["logTg"], np.float64),
            "cont": np.asarray(data["cont"], np.float32),  # (30, n_ebins_c, nTc)
            "logTc": np.asarray(data["logTc"], np.float64),
            "E_ref_edges": np.asarray(data["E_ref_edges"], np.float64),
            "elem_amass": np.asarray(data["elem_amass"], np.float64),
        }


def restrict_band(
    table: dict[str, np.ndarray],
    e_lo_band: float,
    e_hi_band: float,
    z_max: float = 0.3,
    margin_frac: float = 0.05,
) -> dict[str, np.ndarray]:
    """Restrict a table to the rest-frame energies needed for an observed band.

    ``margin_frac`` retains broadened wings. Flux outside the requested band is unavailable.
    """
    e_min = e_lo_band * (1.0 - margin_frac)
    e_max = e_hi_band * (1.0 + margin_frac) * (1.0 + z_max)
    out = dict(table)

    mask = (table["E0"] >= e_min) & (table["E0"] <= e_max)
    for key in ("E0", "Zidx", "Amass", "g"):
        out[key] = np.ascontiguousarray(table[key][mask])

    edges = table["E_ref_edges"]
    i0 = max(int(np.searchsorted(edges, e_min, side="right")) - 1, 0)
    i1 = min(int(np.searchsorted(edges, e_max, side="left")), edges.size - 1)
    i1 = max(i1, i0 + 1)
    out["E_ref_edges"] = np.ascontiguousarray(edges[i0 : i1 + 1])
    out["cont"] = np.ascontiguousarray(table["cont"][:, i0:i1, :])

    return out


def _prepare_runtime_table(table: dict[str, np.ndarray]) -> dict[str, jax.Array | int]:
    """Return the runtime form of a table: device arrays plus the static FFT length.

    Continuum bins are replaced by their zero-padded spectra for linear FFT convolution. The
    arrays are placed on the default device once, so every component and every compiled
    function built from the same table shares one buffer.
    """
    n_cont_bins = table["cont"].shape[1]
    n_fft = next_fast_len(n_cont_bins + 256)
    spectra = np.fft.rfft(table["cont"].astype(np.float64), n=n_fft, axis=1)
    arrays = {key: value for key, value in table.items() if key != "cont"}
    arrays["cont_F"] = np.ascontiguousarray(spectra.transpose(2, 0, 1).astype(np.complex64))
    out: dict[str, jax.Array | int] = {key: jax.device_put(value) for key, value in arrays.items()}
    out["n_fft"] = int(n_fft)
    return out


@lru_cache(maxsize=1)
def runtime_table() -> dict[str, jax.Array | int]:
    """Return the cached full table in runtime form."""
    return _prepare_runtime_table(load_apec_table())


@lru_cache(maxsize=8)
def restricted_table(
    e_lo_band: float,
    e_hi_band: float,
    z_max: float = 0.3,
    margin_frac: float = 0.05,
) -> dict[str, jax.Array | int]:
    """Return a cached, band-restricted table in runtime form."""
    return _prepare_runtime_table(
        restrict_band(load_apec_table(), e_lo_band, e_hi_band, z_max, margin_frac)
    )


class _ApecTables:
    """Hold the runtime table outside the NNX tree.

    A ``jax.Array`` attribute on a module becomes NNX state, which prior binding and
    ``nnx.clone`` would traverse for every replica and every trace. This plain container is a
    static attribute instead: replicas share it by identity and it never enters ``nnx.split``.
    """

    __slots__ = ("arrays",)

    def __init__(self, arrays: dict[str, jax.Array | int]):
        self.arrays = arrays


def _temperature_weights(logT_nodes, kT):
    """Return the bracket and left weight for clipped, linear-in-temperature interpolation."""
    logT_nodes = jnp.asarray(logT_nodes)  # Support traced indices.
    logT = jnp.log10(kT)
    i = jnp.clip(jnp.searchsorted(logT_nodes, logT) - 1, 0, logT_nodes.shape[0] - 2)
    T0, T1 = 10.0 ** logT_nodes[i], 10.0 ** logT_nodes[i + 1]
    w = jnp.clip((T1 - kT) / (T1 - T0), 0.0, 1.0)
    return i, w


def _interp_T(logT_nodes, table_lastaxis, kT):
    """2-node temperature interpolation along the last axis."""
    i, w = _temperature_weights(logT_nodes, kT)
    lo = jnp.take(table_lastaxis, i, axis=-1)
    hi = jnp.take(table_lastaxis, i + 1, axis=-1)
    return w * lo + (1.0 - w) * hi


def _deposit_lines(acc, e_low, e_high, A, E_obs, s, M, K):
    """Integrate a line batch over nearby bins and add it to ``acc``."""
    j0 = jnp.clip(jnp.searchsorted(e_low, E_obs, side="right") - 1, 0, M - 1).astype(jnp.int32)
    offs = jnp.arange(-(K // 2), K - (K // 2), dtype=jnp.int32)  # (K,)
    jj = j0[:, None] + offs[None, :]  # (L, K)
    valid = (jj >= 0) & (jj < M)
    idx = jnp.clip(jj, 0, M - 1)
    ce_hi = jax.lax.erf(((e_high[idx] - E_obs[:, None]) / s[:, None]).astype(jnp.float32))
    ce_lo = jax.lax.erf(((e_low[idx] - E_obs[:, None]) / s[:, None]).astype(jnp.float32))
    d = jnp.where(valid, 0.5 * A.astype(jnp.float32)[:, None] * (ce_hi - ce_lo), 0.0)
    return acc.at[idx.ravel()].add(d.ravel())


def suggest_line_chunks(n_lines: int, k_window: int) -> int:
    """Return the automatic batch count for an ``(n_lines, k_window)`` grid."""
    grid = int(n_lines) * int(k_window)
    if grid <= LINE_GRID_MIN_BATCHED:
        return 1
    return -(-grid // LINE_GRID_PER_BATCH)


def _line_flux(e_low, e_high, A, E_obs, s, M, K, n_chunks):
    """Accumulate line flux in one operation or a checkpointed scan."""
    if n_chunks is None:
        n_chunks = suggest_line_chunks(A.shape[0], K)
    zeros = jnp.zeros(M, dtype=jnp.float32)

    if n_chunks <= 1:
        return _deposit_lines(zeros, e_low, e_high, A, E_obs, s, M, K)

    L = A.shape[0]
    size = -(-L // n_chunks)
    pad = size * n_chunks - L
    if pad:
        # Keep every scan batch the same shape.
        A = jnp.pad(A, (0, pad))
        E_obs = jnp.pad(E_obs, (0, pad), constant_values=1.0)
        s = jnp.pad(s, (0, pad), constant_values=1.0)

    @jax.checkpoint  # Recompute the erf grid during the backward pass.
    def body(acc, batch):
        A_c, E_c, s_c = batch
        return _deposit_lines(acc, e_low, e_high, A_c, E_c, s_c, M, K), None

    batches = tuple(x.reshape(n_chunks, size) for x in (A, E_obs, s))
    line_flux, _ = jax.lax.scan(body, zeros, batches)
    return line_flux


def apec_flux(
    e_low,
    e_high,
    kT,
    abund_vec,
    sigma_v,
    z,
    norm,
    *,
    E0,
    Zidx,
    Amass,
    g,
    logTg,
    cont_F,
    logTc,
    E_ref_edges,
    elem_amass,
    n_fft,
    k_window,
    line_chunks=None,
):
    """Return integrated line and continuum photon flux per output bin.

    ``abund_vec`` contains 30 abundances relative to AG89 solar. ``k_window``, ``n_fft`` and
    ``line_chunks`` are static; ``line_chunks=None`` selects the batch count automatically.
    Bins may be non-contiguous.
    """
    e_low = jnp.asarray(e_low)
    e_high = jnp.asarray(e_high)
    M = e_low.shape[0]
    K = int(k_window)

    # Keep flux and gradients finite at kT = v = 0.
    g_now = _interp_T(logTg, g, kT)
    A = norm * NORM_CONST * abund_vec[Zidx] * g_now
    E_obs = E0 / (1.0 + z)
    turb2 = (sigma_v / C_KMS) ** 2 + 1e-24
    s = jnp.sqrt(2.0) * E_obs * jnp.sqrt(kT / (Amass * MU_C2_KEV) + turb2)
    line_flux = _line_flux(e_low, e_high, A, E_obs, s, M, K, line_chunks)

    # Gaussian broadening in log-energy is a Fourier-space multiplication.
    edges = jnp.asarray(E_ref_edges)
    Nc = edges.shape[0] - 1
    du_c = jnp.log(edges[1]) - jnp.log(edges[0])
    i_T, w_T = _temperature_weights(logTc, kT)
    w32 = w_T.astype(jnp.float32)
    F_now = w32 * jnp.take(cont_F, i_T, axis=0) + (1 - w32) * jnp.take(cont_F, i_T + 1, axis=0)
    sig_elem = jnp.sqrt(kT / (elem_amass * MU_C2_KEV) + turb2) / du_c
    mf = jnp.arange(F_now.shape[1])
    G = jnp.exp(-2.0 * (jnp.pi * sig_elem[:, None] * mf[None, :] / n_fft) ** 2)
    spec = jnp.sum(abund_vec.astype(jnp.float32)[:, None] * F_now * G.astype(jnp.float32), axis=0)
    cont_sum = jnp.fft.irfft(spec, n=int(n_fft))[:Nc]
    # Preserve faint bins when differencing the cumulative flux.
    Ccum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(cont_sum.astype(jnp.float64))])
    Clo = jnp.interp(e_low * (1.0 + z), edges, Ccum, left=Ccum[0], right=Ccum[-1])
    Chi = jnp.interp(e_high * (1.0 + z), edges, Ccum, left=Ccum[0], right=Ccum[-1])
    cont_flux = norm * NORM_CONST * (Chi - Clo)

    # Account for cosmological time dilation.
    return (line_flux.astype(cont_flux.dtype) + cont_flux) / (1.0 + z)


# The component evaluates through this wrapper: the tables are arguments of the compiled
# function rather than constants embedded in it (~7x faster compilation, one executable for
# every replica). Inside an enclosing ``jax.jit`` — the fitters' numpyro models — JAX 0.9 still
# captures them as constants of the outer computation. Window, batching and FFT length are static.
_apec_flux_jit = jax.jit(apec_flux, static_argnames=("n_fft", "k_window", "line_chunks"))


def suggest_k_window(
    e_low,
    e_high,
    kT_max: float = 15.0,
    v_max: float = 500.0,
    n_sigma: float = 4.5,
    table: dict[str, np.ndarray] | None = None,
    a_min: float | None = None,
) -> int:
    """Return the smallest 32-bin multiple covering ``n_sigma`` of every line.

    This NumPy-only helper sizes each line at its local bin width. ``table`` supplies actual
    line energies and masses; otherwise bin centers and ``a_min`` (hydrogen by default) are
    used. Changing ``k_window`` recompiles the model.
    """
    e_low = np.asarray(e_low)
    e_high = np.asarray(e_high)
    width = e_high - e_low
    turb2 = (v_max / C_KMS) ** 2
    if table is not None:
        energy = np.asarray(table["E0"])
        mass = np.asarray(table["Amass"])
        in_grid = (energy >= e_low[0]) & (energy <= e_high[-1])
        if not in_grid.any():
            return 32
        energy, mass = energy[in_grid], mass[in_grid]
        frac = np.sqrt(kT_max / (mass * MU_C2_KEV) + turb2)
    else:
        energy = np.sqrt(e_low * e_high)
        frac = np.sqrt(kT_max / ((a_min or 1.008) * MU_C2_KEV) + turb2)
    j = np.clip(np.searchsorted(e_low, energy, side="right") - 1, 0, e_low.size - 1)
    K = int(np.ceil(float(np.max(2.0 * n_sigma * energy * frac / width[j]))))
    return max(32, int(np.ceil(K / 32.0)) * 32)


class APEC(AdditiveComponent):
    r"""Collisionally ionized plasma emission from AtomDB 3.1.3.

    This component represents the XSPEC `apec`, `bapec`, `vapec`, `bvapec`, `vvapec` and
    `bvvapec` variants. Lines and the pseudo-continuum are thermally broadened; setting
    ``broadening=True`` adds turbulent velocity broadening.

    !!! abstract "Parameters"
        * $kT$ (`kT`) $\left[\text{keV}\right]$ : Plasma temperature
        * $A$ (`abund`) $\left[\text{dimensionless}\right]$ : Abundance of the 12 vapec metals
          relative to solar; available when ``abundances="fixed"``
        * $A_Z$ (`He`, `C`, `N`, `O`, `Ne`, `Mg`, `Al`, `Si`, `S`, `Ar`, `Ca`, `Fe`, `Ni`)
          $\left[\text{dimensionless}\right]$ : Individual abundances relative to solar;
          available when ``abundances="free"``
        * $A_Z$ (`H`, `He`, ..., `Zn`) $\left[\text{dimensionless}\right]$ : All 30 individual
          abundances relative to solar; available when ``abundances="all"``
        * $v$ (`velocity`) $\left[\text{km s}^{-1}\right]$ : Gaussian turbulent velocity
          broadening; available when ``broadening=True``
        * $z$ (`redshift`) $\left[\text{dimensionless}\right]$ : Redshift
        * $K$ (`norm`) : Normalization
          $\frac{10^{-14}}{4\pi \left[D_A (1+z)\right]^2} \int n_e n_H dV$ (XSPEC convention)

    !!! abstract "Constructor arguments"
        * `broadening` : add the ``velocity`` parameter
        * `abundances` : ``"fixed"`` for one metal abundance, ``"free"`` for the 13 vapec
          abundances, or ``"all"`` for all 30 elements
        * `abundance_table` : solar abundance scale from
          [`abundance_table`][jaxspec.util.abundance.abundance_table]
        * `energy_band` : optional observed ``(e_min, e_max)`` band in keV; flux outside this
          band is unavailable
        * `k_window` : number of nearby output bins receiving each line. ``"auto"`` (default)
          sizes it from the energy grid at the first evaluation, for kT ≤ 15 keV and
          v ≤ 500 km/s; pass an integer for broader lines. A grid that is only ever traced
          (for instance shifted by an instrument model) falls back to 128 until a concrete grid
          has been seen
        * `line_chunks` : static line batch count; ``None`` selects it automatically
        * `z_max` : largest redshift covered by ``energy_band``

    !!! warning
        Constructor arguments are static; changing them requires a new component and JIT
        compilation. The evaluation is compiled once per energy grid shape and window; the
        tables are device arrays shared by every replica and passed as arguments of that
        compiled function. Enable JAX float64 for full table precision.

    !!! warning "Table coverage"
        The table covers 0.1–15 keV in the rest frame. Emissivities clip to its temperature
        range, 0.0813–38.5 keV.
    """

    def __init__(
        self,
        broadening: bool = False,
        abundances: Literal["fixed", "free", "all"] = "fixed",
        abundance_table: str = "angr",
        energy_band: tuple[float, float] | None = None,
        k_window: int | Literal["auto"] = "auto",
        line_chunks: int | None = None,
        z_max: float = 0.3,
    ):
        self.kT = nnx.Param(1.0)
        self.redshift = nnx.Param(0.0)
        self.norm = nnx.Param(1.0)

        if broadening:
            self.velocity = nnx.Param(0.0)

        if abundances == "fixed":
            self.abund = nnx.Param(1.0)
        elif abundances == "free":
            for symbol in VAPEC_SYMBOLS:
                setattr(self, symbol, nnx.Param(1.0))
        elif abundances == "all":
            for symbol in ELEMENT_SYMBOLS:
                setattr(self, symbol, nnx.Param(1.0))
        else:
            raise ValueError(f"abundances must be 'fixed', 'free' or 'all', got {abundances!r}")

        if isinstance(k_window, str):
            if k_window != "auto":
                raise ValueError(f"k_window must be a positive integer or 'auto', got {k_window!r}")
            self._k_window = None
        else:
            self._k_window = int(k_window)
            if self._k_window < 1:
                raise ValueError(f"k_window must be a positive integer or 'auto', got {k_window!r}")

        self._broadening = bool(broadening)
        self._abundances = abundances
        self._line_chunks = None if line_chunks is None else max(int(line_chunks), 1)
        self._k_window_checked = False
        # Grid of the last automatic window sizing (bins, first edge, last edge) and its result.
        self._auto_n_bins = -1
        self._auto_e_min = 0.0
        self._auto_e_max = 0.0
        self._auto_k = DEFAULT_K_WINDOW

        self._tables = _ApecTables(
            restricted_table(*energy_band, z_max=z_max)
            if energy_band is not None
            else runtime_table()
        )

        if abundance_table == "angr":
            self._ab_ratio = np.ones(len(ELEMENT_SYMBOLS))
        else:
            from ...util.abundance import abundance_table as abundance_df

            valid_tables = [column for column in abundance_df.columns if column != "Element"]
            if abundance_table not in valid_tables:
                raise ValueError(
                    f"Unknown abundance table {abundance_table!r}, expected one of {valid_tables}"
                )
            self._ab_ratio = np.asarray(abundance_df[abundance_table], dtype=float) / np.asarray(
                abundance_df["angr"], dtype=float
            )

    def _suggest_k_window(self, e_low, e_high, **kwargs) -> int:
        """Suggest a safe `k_window` for this component and energy grid."""
        arrays = self._tables.arrays
        kwargs.setdefault(
            "table", {"E0": np.asarray(arrays["E0"]), "Amass": np.asarray(arrays["Amass"])}
        )
        return suggest_k_window(e_low, e_high, **kwargs)

    def _resolve_k_window(self, e_low, e_high) -> int:
        """Return the static window for this grid: the explicit value, or one sized from it."""
        traced = isinstance(e_low, jax.core.Tracer)

        if self._k_window is not None:
            if not traced and not self._k_window_checked:
                self._k_window_checked = True
                needed = self._suggest_k_window(np.asarray(e_low), np.asarray(e_high))
                if needed > self._k_window:
                    warnings.warn(
                        f"APEC k_window={self._k_window} is smaller than the {needed} suggested "
                        "for this energy grid (sized for kT <= 15 keV, v <= 500 km/s): line "
                        "wings will be silently truncated. Pass k_window=... at construction.",
                        stacklevel=3,
                    )
            return self._k_window

        n_bins = int(e_low.shape[0])
        if traced:
            if self._auto_n_bins != n_bins:
                warnings.warn(
                    f"APEC k_window='auto' met a traced energy grid of {n_bins} bins before any "
                    f"concrete one; using k_window={DEFAULT_K_WINDOW}. Evaluate the component "
                    "once on concrete energies first, or pass k_window=... at construction.",
                    stacklevel=3,
                )
                return DEFAULT_K_WINDOW
            return self._auto_k

        # Host copies first: indexing a concrete array inside a trace would yield a tracer.
        lo, hi = np.asarray(e_low), np.asarray(e_high)
        e_min, e_max = float(lo[0]), float(hi[-1])
        if (n_bins, e_min, e_max) != (self._auto_n_bins, self._auto_e_min, self._auto_e_max):
            self._auto_k = self._suggest_k_window(lo, hi)
            self._auto_n_bins, self._auto_e_min, self._auto_e_max = n_bins, e_min, e_max
        return self._auto_k

    def _abund_vector(self):
        """Return per-element abundances relative to AG89 solar."""
        if self._abundances == "fixed":
            vector = jnp.where(METAL_MASK, jnp.asarray(self.abund), 1.0)
        elif self._abundances == "free":
            values = jnp.stack([jnp.asarray(getattr(self, symbol)) for symbol in VAPEC_SYMBOLS])
            vector = jnp.ones(len(ELEMENT_SYMBOLS)).at[VAPEC_IDX].set(values)
        else:
            vector = jnp.stack([jnp.asarray(getattr(self, symbol)) for symbol in ELEMENT_SYMBOLS])
        return vector * self._ab_ratio

    def integrated_continuum(self, e_low, e_high):
        sigma_v = jnp.asarray(self.velocity) if self._broadening else 0.0
        return _apec_flux_jit(
            e_low,
            e_high,
            jnp.asarray(self.kT),
            self._abund_vector(),
            sigma_v,
            jnp.asarray(self.redshift),
            jnp.asarray(self.norm),
            k_window=self._resolve_k_window(e_low, e_high),
            line_chunks=self._line_chunks,
            **self._tables.arrays,
        )
