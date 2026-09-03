"""Shared infrastructure for live XSPEC-vs-jaxspec model comparisons.

Every comparison is computed live: XSPEC is evaluated through PyXSPEC at test
time and jaxspec in the same process — nothing is precomputed, stored or cached.
The only static data below are input parameter values and tolerances.

This module never imports ``xspec`` at module level so it stays importable in
environments without HEASOFT (the test modules gate execution with
``pytest.importorskip("xspec")``; the evaluation helpers import it lazily).

Error metric: per-bin all-close assertions are brittle for spectral models —
Wien tails, saturated absorption troughs and line wings underflow to ~1e-20
where the relative error is dominated by XSPEC's single-precision internals,
and a single bin straddling a discontinuity (``highecut``, ``zedge``,
``expfac`` breakpoints) fails the whole test while carrying no flux. Instead we
mask bins below ``max(flux) * mask_rel`` and check three quantities: the median
relative error (typical agreement), the 99th percentile (shape errors), and the
relative error on the total band flux (pure normalization biases that
bin-wise quantiles can half-hide).
"""

import os

from collections.abc import Callable
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import numpy as np

from jaxspec.model._additive.apec import ELEMENT_SYMBOLS, VAPEC_SYMBOLS
from jaxspec.model._additive.disk import DEFAULT_FLUX_BAND
from jaxspec.model.additive import (
    APEC,
    Agauss,
    Band,
    Blackbody,
    Blackbodyrad,
    Cutoffpl,
    Diskbb,
    Diskpbb,
    Gauss,
    Logparabola,
    Lorentz,
    NSatmos,
    Powerlaw,
    Zagauss,
    Zgauss,
)
from jaxspec.model.multiplicative import (
    Expfac,
    Gabs,
    Highecut,
    Phabs,
    Tbabs,
    Tbpcf,
    Wabs,
    Zedge,
    zTbabs,
)

DEFAULT_MEDIAN_TOL = 1e-4
DEFAULT_P99_TOL = 1e-3
DEFAULT_TOTAL_TOL = 1e-4

KEV_TO_ERG = 1.602176634e-9  # (1 keV).to(erg), CODATA exact


@dataclass(frozen=True)
class XspecSettings:
    """Global XSPEC state applied before every evaluation.

    ``abund`` and ``xsect`` are always (re)assigned for determinism — they are
    process-global in XSPEC and would otherwise leak between cases. ``abund``
    must match the provenance of the jaxspec cross-section/emissivity tables.
    """

    abund: str = "angr"
    xsect: str = "vern"
    model_strings: dict[str, str] = field(default_factory=dict)


ANGR = XspecSettings()
WILM = XspecSettings(abund="wilm")  # Tbabs family tables are xsect_tbabs_wilm.fits
ASPL = XspecSettings(abund="aspl")  # Phabs table is xsect_phabs_aspl.fits
APEC_SETTINGS = XspecSettings(
    abund="angr",  # packaged APEC table is expressed in AG89 solar abundances
    model_strings={
        "APECROOT": "3.1.3",  # match the packaged AtomDB version
        "APECTHERMAL": "yes",
        "APECBROADPSEUDO": "yes",
    },
)


@dataclass(frozen=True)
class XspecComparison:
    """One jaxspec-vs-XSPEC comparison: a model pair, parameter sets, tolerances.

    ``param_map`` maps XSPEC ``"component.Parameter"`` keys to jaxspec dotted
    parameter paths, optionally with a value converter:
    ``{"logpar.beta": ("logparabola_1.b", lambda beta: -beta / np.log(10))}``.
    ``notes`` is required whenever a tolerance is looser than the defaults and
    must cite the XSPEC-side quirk that justifies it.

    ``param_hook`` handles parameters that depend on more than their own XSPEC value
    (see ``norm_from_xspec_band_flux``): it receives the case, the XSPEC parameter set
    and the mapped jaxspec parameters, and returns the final ones. It may call XSPEC.
    """

    name: str
    xspec_expression: str
    jaxspec_factory: Callable[[], Any]
    param_map: dict[str, str | tuple[str, Callable[[float], float]]]
    param_sets: tuple[dict[str, float], ...]
    param_hook: Callable[["XspecComparison", dict, dict], dict] | None = None
    grid: tuple[float, float, int, str] = (0.2, 20.0, 2000, "log")
    settings: XspecSettings = field(default_factory=XspecSettings)
    n_points: int = 30
    median_tol: float = DEFAULT_MEDIAN_TOL
    p99_tol: float = DEFAULT_P99_TOL
    total_tol: float = DEFAULT_TOTAL_TOL
    mask_rel: float = 1e-6
    flux_photon_tol: float = 1e-4  # used by the calcFlux comparison only
    flux_energy_tol: float = 1e-3
    notes: str = ""

    def jaxspec_params(self, pset: dict[str, float]) -> dict[str, float]:
        """Translate an XSPEC parameter set into jaxspec dotted parameters."""
        params = {}
        for key, value in pset.items():
            target = self.param_map[key]
            if isinstance(target, tuple):
                target, convert = target
                value = convert(value)
            params[target] = value
        if self.param_hook is not None:
            params = self.param_hook(self, pset, params)
        return params


@dataclass
class ErrorReport:
    median: float
    p99: float
    worst: float
    total_rel: float
    worst_energy: float
    n_used: int
    n_masked: int


def _model_data_dirs() -> list[str]:
    """Candidate XSPEC ``modelData`` directories for the active HEASOFT install.

    Both layouts are probed: ``$HEADAS/spectral/modelData`` (conda-forge / HEASARC
    conda packages) and ``$HEADAS/../spectral/modelData`` (source builds, where
    ``$HEADAS`` is the architecture subdirectory).
    """
    headas = os.environ.get("HEADAS")
    if not headas:
        return []
    candidates = [
        os.path.join(headas, "spectral", "modelData"),
        os.path.join(headas, "..", "spectral", "modelData"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def missing_apec_tables(apecroot: str) -> list[str]:
    """AtomDB files XSPEC would need for ``APECROOT = apecroot`` but cannot find.

    XSPEC does not fail on a missing AtomDB version: it warns and returns an
    all-zero spectrum, which at ``chatter = 0`` is invisible and turns a
    comparison into an empty-mask error far from the cause. The comparison is
    only meaningful against the AtomDB version the jaxspec table was built from,
    so a mismatch must skip rather than fall back to whatever is installed.
    """
    dirs = _model_data_dirs()
    if not dirs:
        return []  # cannot locate the install; let the comparison speak for itself
    return [
        name
        for name in (f"apec_v{apecroot}_coco.fits", f"apec_v{apecroot}_line.fits")
        if not any(glob(os.path.join(d, name)) for d in dirs)
    ]


def require_model_data(settings: XspecSettings) -> None:
    """Skip the calling test when the XSPEC install lacks the pinned AtomDB tables."""
    apecroot = settings.model_strings.get("APECROOT")
    if apecroot is None:
        return
    missing = missing_apec_tables(apecroot)
    if missing:
        import pytest

        pytest.skip(
            f"XSPEC install has no AtomDB {apecroot} tables ({', '.join(missing)}); "
            "the packaged jaxspec table is only comparable against that version."
        )


def apply_xspec_settings(settings: XspecSettings) -> None:
    import xspec

    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.AllModels.clear()
    xspec.AllData.clear()
    xspec.Xset.abund = settings.abund
    xspec.Xset.xsect = settings.xsect
    for key, value in settings.model_strings.items():
        xspec.Xset.addModelString(key, value)


def xspec_model(case: XspecComparison, pset: dict[str, float]):
    """Build the XSPEC model for a case on its energy grid, with params applied."""
    import xspec

    require_model_data(case.settings)
    apply_xspec_settings(case.settings)
    e_min, e_max, n_bins, scale = case.grid
    xspec.AllModels.setEnergies(f"{e_min} {e_max} {n_bins} {scale}")
    model = xspec.Model(case.xspec_expression)
    for dotted, value in pset.items():
        component, parameter = dotted.split(".")
        setattr(getattr(model, component), parameter, value)
    return model


def xspec_photon_flux(
    case: XspecComparison, pset: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Live-evaluate XSPEC: energy bin edges and per-bin integrated photon flux."""
    model = xspec_model(case, pset)
    edges = np.asarray(model.energies(0), dtype=np.float64)
    return edges, np.asarray(model.values(0), dtype=np.float64)


def jaxspec_photon_flux(
    case: XspecComparison, pset: dict[str, float], edges: np.ndarray
) -> np.ndarray:
    model = case.jaxspec_factory()
    return np.asarray(
        model.photon_flux(
            edges[:-1], edges[1:], params=case.jaxspec_params(pset), n_points=case.n_points
        )
    )


def error_report(edges, expected, got, mask_rel=1e-6) -> ErrorReport:
    expected = np.asarray(expected, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    mask = expected > expected.max() * mask_rel
    rel = np.abs(got[mask] - expected[mask]) / expected[mask]
    rel = np.where(np.isfinite(rel), rel, np.inf)  # NaNs must fail, not slip through
    if rel.size == 0:
        raise AssertionError(
            "reference spectrum carries no flux (all bins are zero or non-finite): "
            "XSPEC evaluated the model but returned nothing to compare against — "
            "usually a missing model-data table, which XSPEC only reports at chatter > 0."
        )
    centers = 0.5 * (edges[:-1] + edges[1:])[mask]
    i_worst = int(np.argmax(rel))
    total = got.sum() / expected.sum() - 1.0
    return ErrorReport(
        median=float(np.median(rel)),
        p99=float(np.percentile(rel, 99)),
        worst=float(rel[i_worst]),
        total_rel=float(total) if np.isfinite(total) else float("inf"),
        worst_energy=float(centers[i_worst]),
        n_used=int(mask.sum()),
        n_masked=int((~mask).sum()),
    )


def evaluate_case(case: XspecComparison, pset: dict[str, float]) -> ErrorReport:
    """Run one live XSPEC + jaxspec evaluation and return the error report."""
    edges, expected = xspec_photon_flux(case, pset)
    got = jaxspec_photon_flux(case, pset, edges)
    return error_report(edges, expected, got, mask_rel=case.mask_rel)


def assert_close_to_xspec(case: XspecComparison, pset: dict[str, float]) -> None:
    report = evaluate_case(case, pset)
    failures = []
    if report.median > case.median_tol:
        failures.append(f"median {report.median:.2e} > {case.median_tol:.0e}")
    if report.p99 > case.p99_tol:
        failures.append(f"p99 {report.p99:.2e} > {case.p99_tol:.0e}")
    if abs(report.total_rel) > case.total_tol:
        failures.append(f"total {report.total_rel:+.2e} > {case.total_tol:.0e}")
    assert not failures, (
        f"{case.name} vs XSPEC '{case.xspec_expression}' with {pset}: {', '.join(failures)}; "
        f"worst {report.worst:.2e} @ {report.worst_energy:.3f} keV "
        f"({report.n_used}/{report.n_used + report.n_masked} bins used)."
        + (f" Note: {case.notes}" if case.notes else "")
    )


def as_pytest_params(cases):
    """Flatten cases into pytest.param(case, pset) entries with readable ids."""
    import pytest

    return [
        pytest.param(case, pset, id=f"{case.name}-{i}")
        for case in cases
        for i, pset in enumerate(case.param_sets)
    ]


def xspec_band_flux(
    case: XspecComparison, pset: dict[str, float], band: tuple[float, float]
) -> tuple[float, float]:
    """Live XSPEC calcFlux: (photon flux [ph/cm2/s], energy flux [erg/cm2/s])."""
    import xspec

    model = xspec_model(case, pset)
    xspec.AllModels.calcFlux(f"{band[0]} {band[1]}")
    return model.flux[3], model.flux[0]


def jaxspec_band_flux(
    case: XspecComparison, pset: dict[str, float], band: tuple[float, float], n_grid: int = 2000
) -> tuple[float, float]:
    """Band-integrated photon and energy flux, summed over a fine sub-grid.

    A single-bin evaluation would be wrong for product models (the bin-averaged
    absorption factor times the band-integrated continuum is not the integral
    of their product), so integrate on a fine grid like XSPEC does.
    """
    model = case.jaxspec_factory()
    params = case.jaxspec_params(pset)
    grid = np.geomspace(band[0], band[1], n_grid + 1)
    photon = float(
        np.sum(model.photon_flux(grid[:-1], grid[1:], params=params, n_points=case.n_points))
    )
    energy = (
        float(np.sum(model.energy_flux(grid[:-1], grid[1:], params=params, n_points=case.n_points)))
        * KEV_TO_ERG
    )
    return photon, energy


def norm_from_xspec_band_flux(band: tuple[float, float], norm_path: str) -> Callable:
    """Build a ``param_hook`` setting a jaxspec norm to XSPEC's own band photon flux.

    ``calcFlux`` includes the XSPEC norm, so the hook replaces the mapped value. It
    rebuilds the process-global XSPEC model, so callers must not hold a live
    ``xspec.Model`` across a ``jaxspec_params`` call.
    """

    def hook(case: XspecComparison, pset: dict, params: dict) -> dict:
        photon_flux, _ = xspec_band_flux(case, pset, band)
        return {**params, norm_path: photon_flux}

    return hook


def _flat_powerlaw(psets: tuple[dict[str, float], ...]) -> tuple[dict[str, float], ...]:
    """Append the frozen flat power law used to probe multiplicative factors.

    With ``PhoIndex=0, norm=1`` the additive part is exact and identical on
    both sides (per-bin flux = bin width), so every deviation of the product is
    attributable to the multiplicative factor — unlike dividing XSPEC output by
    a near-zero continuum, which is ill-defined in saturated troughs.
    """
    return tuple({**pset, "powerlaw.PhoIndex": 0.0, "powerlaw.norm": 1.0} for pset in psets)


_PL_MAP = {"powerlaw.PhoIndex": "powerlaw_1.alpha", "powerlaw.norm": "powerlaw_1.norm"}


# --- Disk model cases -------------------------------------------------------------
#
# Run by tests/test_diskbb.py as its own xspec test: these two components' norm is
# not XSPEC's, so the equivalence is pinned there. Spliced into ADDITIVE_CASES below.

DISK_CASES = [
    XspecComparison(
        name="diskbb",
        xspec_expression="diskbb",
        jaxspec_factory=Diskbb,
        param_map={"diskbb.Tin": "diskbb_1.Tin", "diskbb.norm": "diskbb_1.norm"},
        param_hook=norm_from_xspec_band_flux(DEFAULT_FLUX_BAND, "diskbb_1.norm"),
        param_sets=(
            {"diskbb.Tin": 0.3, "diskbb.norm": 1.0},
            {"diskbb.Tin": 1.0, "diskbb.norm": 1.0},
            {"diskbb.Tin": 3.0, "diskbb.norm": 1.0},
        ),
        grid=(0.1, 20.0, 2000, "log"),
        notes="Norm matched through XSPEC calcFlux. The default tolerance covers "
        "XSPEC's numerical precision after the band flux fixes the normalization.",
    ),
    XspecComparison(
        name="diskpbb",
        xspec_expression="diskpbb",
        jaxspec_factory=Diskpbb,
        param_map={
            "diskpbb.Tin": "diskpbb_1.Tin",
            "diskpbb.p": "diskpbb_1.p",
            "diskpbb.norm": "diskpbb_1.norm",
        },
        param_hook=norm_from_xspec_band_flux(DEFAULT_FLUX_BAND, "diskpbb_1.norm"),
        param_sets=(
            {"diskpbb.Tin": 1.0, "diskpbb.p": 0.75, "diskpbb.norm": 1.0},
            {"diskpbb.Tin": 1.5, "diskpbb.p": 0.6, "diskpbb.norm": 1.0},
            {"diskpbb.Tin": 2.0, "diskpbb.p": 0.9, "diskpbb.norm": 1.0},
        ),
        grid=(0.1, 20.0, 2000, "log"),
        median_tol=1e-3,
        p99_tol=2e-2,
        total_tol=2e-3,
        notes="Norm matched through XSPEC calcFlux. The tolerance covers XSPEC's "
        "low-energy quadrature residual for shallow radial exponents outside the "
        "0.5-10 keV normalization band.",
    ),
]


# --- Additive model cases --------------------------------------------------------

ADDITIVE_CASES = [
    XspecComparison(
        name="powerlaw",
        xspec_expression="powerlaw",
        jaxspec_factory=Powerlaw,
        param_map=_PL_MAP,
        # PhoIndex=1.0 probes the analytic pole in integrated_continuum
        param_sets=(
            {"powerlaw.PhoIndex": 0.5, "powerlaw.norm": 1.0},
            {"powerlaw.PhoIndex": 1.7, "powerlaw.norm": 1e-2},
            {"powerlaw.PhoIndex": 3.0, "powerlaw.norm": 1.0},
            {"powerlaw.PhoIndex": 1.0, "powerlaw.norm": 1.0},
        ),
    ),
    XspecComparison(
        name="bbody",
        xspec_expression="bbody",
        jaxspec_factory=Blackbody,
        param_map={"bbody.kT": "blackbody_1.kT", "bbody.norm": "blackbody_1.norm"},
        param_sets=(
            {"bbody.kT": 0.3, "bbody.norm": 1.0},
            {"bbody.kT": 1.0, "bbody.norm": 1.0},
            {"bbody.kT": 3.0, "bbody.norm": 1.0},
        ),
    ),
    XspecComparison(
        name="bbodyrad",
        xspec_expression="bbodyrad",
        jaxspec_factory=Blackbodyrad,
        param_map={"bbodyrad.kT": "blackbodyrad_1.kT", "bbodyrad.norm": "blackbodyrad_1.norm"},
        param_sets=(
            {"bbodyrad.kT": 0.3, "bbodyrad.norm": 1.0},
            {"bbodyrad.kT": 1.0, "bbodyrad.norm": 1.0},
            {"bbodyrad.kT": 3.0, "bbodyrad.norm": 1.0},
        ),
    ),
    XspecComparison(
        name="lorentz",
        xspec_expression="lorentz",
        jaxspec_factory=Lorentz,
        param_map={
            "lorentz.LineE": "lorentz_1.E_l",
            "lorentz.Width": "lorentz_1.sigma",
            "lorentz.norm": "lorentz_1.norm",
        },
        param_sets=(
            {"lorentz.LineE": 6.4, "lorentz.Width": 0.5, "lorentz.norm": 1.0},
            {"lorentz.LineE": 3.0, "lorentz.Width": 1.0, "lorentz.norm": 1.0},
            {"lorentz.LineE": 6.4, "lorentz.Width": 0.05, "lorentz.norm": 1.0},
        ),
        grid=(0.5, 15.0, 3000, "lin"),
        total_tol=5e-3,
        notes="per-bin agreement is exact (XSPEC lorentz renormalizes on [0, inf) like "
        "jaxspec, measured median ~1e-7) but XSPEC stops evaluating far Lorentzian "
        "wings once the summed line flux converges, so for narrow lines the in-band "
        "totals differ by the dropped tail mass (~0.4% at Width=0.05 on 0.5-15 keV).",
    ),
    XspecComparison(
        name="gauss",
        xspec_expression="gauss",
        jaxspec_factory=Gauss,
        param_map={
            "gaussian.LineE": "gauss_1.El",
            "gaussian.Sigma": "gauss_1.sigma",
            "gaussian.norm": "gauss_1.norm",
        },
        # the (1.0, 0.5) set probes the [0, inf) truncation renormalization
        param_sets=(
            {"gaussian.LineE": 6.4, "gaussian.Sigma": 0.1, "gaussian.norm": 1.0},
            {"gaussian.LineE": 3.0, "gaussian.Sigma": 1.0, "gaussian.norm": 1.0},
            {"gaussian.LineE": 1.0, "gaussian.Sigma": 0.5, "gaussian.norm": 1.0},
        ),
        grid=(0.1, 15.0, 2000, "log"),
    ),
    XspecComparison(
        name="zgauss",
        xspec_expression="zgauss",
        jaxspec_factory=Zgauss,
        param_map={
            "zgauss.LineE": "zgauss_1.El",
            "zgauss.Sigma": "zgauss_1.sigma",
            "zgauss.Redshift": "zgauss_1.redshift",
            "zgauss.norm": "zgauss_1.norm",
        },
        param_sets=(
            {"zgauss.LineE": 6.4, "zgauss.Sigma": 0.2, "zgauss.Redshift": 0.1, "zgauss.norm": 1.0},
            {
                "zgauss.LineE": 2.0,
                "zgauss.Sigma": 0.05,
                "zgauss.Redshift": 0.5,
                "zgauss.norm": 1.0,
            },
        ),
        grid=(0.1, 15.0, 2000, "log"),
    ),
    XspecComparison(
        name="agauss",
        xspec_expression="agauss",
        jaxspec_factory=Agauss,
        param_map={
            "agauss.LineE": "agauss_1.lambdal",
            "agauss.Sigma": "agauss_1.sigma",
            "agauss.norm": "agauss_1.norm",
        },
        param_sets=(
            {"agauss.LineE": 12.0, "agauss.Sigma": 0.1, "agauss.norm": 1.0},
            {"agauss.LineE": 18.0, "agauss.Sigma": 0.5, "agauss.norm": 1.0},
        ),
        grid=(0.3, 3.0, 2000, "lin"),
    ),
    XspecComparison(
        name="zagauss",
        xspec_expression="zagauss",
        jaxspec_factory=Zagauss,
        param_map={
            "zagauss.LineE": "zagauss_1.lambdal",
            "zagauss.Sigma": "zagauss_1.sigma",
            "zagauss.Redshift": "zagauss_1.redshift",
            "zagauss.norm": "zagauss_1.norm",
        },
        param_sets=(
            {
                "zagauss.LineE": 12.0,
                "zagauss.Sigma": 0.1,
                "zagauss.Redshift": 0.0,
                "zagauss.norm": 1.0,
            },
            {
                "zagauss.LineE": 12.0,
                "zagauss.Sigma": 0.3,
                "zagauss.Redshift": 0.2,
                "zagauss.norm": 1.0,
            },
        ),
        grid=(0.3, 3.0, 2000, "lin"),
    ),
    XspecComparison(
        name="cutoffpl",
        xspec_expression="cutoffpl",
        jaxspec_factory=Cutoffpl,
        param_map={
            "cutoffpl.PhoIndex": "cutoffpl_1.alpha",
            "cutoffpl.HighECut": "cutoffpl_1.beta",
            "cutoffpl.norm": "cutoffpl_1.norm",
        },
        param_sets=(
            {"cutoffpl.PhoIndex": 1.5, "cutoffpl.HighECut": 15.0, "cutoffpl.norm": 1.0},
            {"cutoffpl.PhoIndex": 0.5, "cutoffpl.HighECut": 50.0, "cutoffpl.norm": 1.0},
            {"cutoffpl.PhoIndex": 2.2, "cutoffpl.HighECut": 5.0, "cutoffpl.norm": 1.0},
        ),
    ),
    XspecComparison(
        name="logpar",
        xspec_expression="logpar",
        jaxspec_factory=Logparabola,
        # XSPEC logpar exponent is -(alpha + beta*log10 E); jaxspec uses
        # -(a - b*ln E), hence a=alpha and b=-beta/ln(10). pivotE stays at 1 keV.
        param_map={
            "logpar.alpha": "logparabola_1.a",
            "logpar.beta": ("logparabola_1.b", lambda beta: -beta / np.log(10.0)),
            "logpar.norm": "logparabola_1.norm",
        },
        param_sets=(
            {"logpar.alpha": 1.5, "logpar.beta": 0.2, "logpar.norm": 1.0},
            {"logpar.alpha": 2.0, "logpar.beta": -0.3, "logpar.norm": 1.0},
            {"logpar.alpha": 1.0, "logpar.beta": 0.5, "logpar.norm": 1.0},
        ),
    ),
    *DISK_CASES,
    XspecComparison(
        name="grbm",
        xspec_expression="grbm",
        jaxspec_factory=Band,
        # grbm and Band share the negative-index convention: direct mapping
        param_map={
            "grbm.alpha": "band_1.alpha1",
            "grbm.beta": "band_1.alpha2",
            "grbm.tem": "band_1.Ec",
            "grbm.norm": "band_1.norm",
        },
        param_sets=(
            {"grbm.alpha": -1.0, "grbm.beta": -2.3, "grbm.tem": 300.0, "grbm.norm": 1.0},
            {"grbm.alpha": -0.5, "grbm.beta": -2.0, "grbm.tem": 150.0, "grbm.norm": 1.0},
            {"grbm.alpha": -1.5, "grbm.beta": -3.0, "grbm.tem": 500.0, "grbm.norm": 1.0},
        ),
        grid=(1.0, 2000.0, 2000, "log"),
    ),
    XspecComparison(
        name="nsatmos",
        xspec_expression="nsatmos",
        jaxspec_factory=NSatmos,
        # both sides take log10(T_eff [K]); keep R well above 1.125 R_s where the
        # boundary handling differs (XSPEC clamps, jaxspec returns null flux)
        param_map={
            "nsatmos.LogT_eff": "nsatmos_1.Tinf",
            "nsatmos.M_ns": "nsatmos_1.mass",
            "nsatmos.R_ns": "nsatmos_1.radius",
            "nsatmos.dist": "nsatmos_1.distance",
            "nsatmos.norm": "nsatmos_1.norm",
        },
        param_sets=(
            {
                "nsatmos.LogT_eff": 6.0,
                "nsatmos.M_ns": 1.4,
                "nsatmos.R_ns": 10.0,
                "nsatmos.dist": 10.0,
                "nsatmos.norm": 1.0,
            },
            {
                "nsatmos.LogT_eff": 5.7,
                "nsatmos.M_ns": 1.8,
                "nsatmos.R_ns": 12.0,
                "nsatmos.dist": 5.0,
                "nsatmos.norm": 1.0,
            },
            {
                "nsatmos.LogT_eff": 6.3,
                "nsatmos.M_ns": 1.2,
                "nsatmos.R_ns": 11.0,
                "nsatmos.dist": 8.0,
                "nsatmos.norm": 0.6,
            },
        ),
        grid=(0.1, 3.0, 1000, "log"),
        median_tol=1e-2,
        p99_tol=8e-2,
        total_tol=1e-2,
        notes="table-interpolation model: jaxspec (interpax, linear in T/g) and XSPEC "
        "interpolate the same nsatmosdata.fits grid with different schemes; worst "
        "measured median 0.8%, p99 6% in the Wien tail (log-space interpolation was "
        "tried and is not closer).",
    ),
]

# Drop cases whose component is not landed yet (``jaxspec_factory`` is then ``None``).
ADDITIVE_CASES = [case for case in ADDITIVE_CASES if case.jaxspec_factory is not None]


# --- Multiplicative model cases (probed through <mult>*powerlaw, flat PL) --------

MULTIPLICATIVE_CASES = [
    XspecComparison(
        name="tbabs",
        xspec_expression="TBabs*powerlaw",
        jaxspec_factory=lambda: Tbabs() * Powerlaw(),
        param_map={"TBabs.nH": "tbabs_1.nh", **_PL_MAP},
        param_sets=_flat_powerlaw(
            (
                {"TBabs.nH": 0.1},
                {"TBabs.nH": 1.0},
                {"TBabs.nH": 5.0},
            )
        ),
        grid=(0.1, 10.0, 2000, "log"),
        settings=WILM,
        median_tol=1e-3,
        p99_tol=1e-2,
        total_tol=1e-3,
        mask_rel=1e-3,
        notes="interpolated cross-section table (xsect_tbabs_wilm.fits) vs XSPEC's "
        "on-the-fly TBabs computation; the interpolation smooths the sharp photo-"
        "ionization edges, so saturated near-zero bins right at the edges are excluded "
        "(mask_rel=1e-3) — they carry no flux but arbitrarily large relative errors.",
    ),
    XspecComparison(
        name="ztbabs",
        xspec_expression="zTBabs*powerlaw",
        jaxspec_factory=lambda: zTbabs() * Powerlaw(),
        param_map={"zTBabs.nH": "ztbabs_1.nh", "zTBabs.Redshift": "ztbabs_1.z", **_PL_MAP},
        param_sets=_flat_powerlaw(
            (
                {"zTBabs.nH": 1.0, "zTBabs.Redshift": 0.0},
                {"zTBabs.nH": 1.0, "zTBabs.Redshift": 0.5},
                {"zTBabs.nH": 0.3, "zTBabs.Redshift": 2.0},
            )
        ),
        grid=(0.1, 10.0, 2000, "log"),
        settings=WILM,
        median_tol=1e-3,
        p99_tol=1e-1,
        total_tol=1e-3,
        mask_rel=1e-3,
        notes="physics difference, not a bug: jaxspec zTbabs redshifts the full TBabs "
        "cross sections while XSPEC's zTBabs assumes 20% molecular hydrogen and no "
        "grains — XSPEC's own zTBabs(z=0) differs from its TBabs by the same amount "
        "(measured live: median 2.6e-4, p99 6.6e-2, worst 11% at the 0.7 keV edges); "
        "jaxspec zTbabs(z=0) == Tbabs exactly (unit-tested in test_models.py).",
    ),
    XspecComparison(
        name="phabs",
        xspec_expression="phabs*powerlaw",
        jaxspec_factory=lambda: Phabs() * Powerlaw(),
        param_map={"phabs.nH": "phabs_1.nh", **_PL_MAP},
        param_sets=_flat_powerlaw(
            (
                {"phabs.nH": 0.1},
                {"phabs.nH": 1.0},
                {"phabs.nH": 5.0},
            )
        ),
        grid=(0.15, 10.0, 2000, "log"),
        settings=ASPL,
        median_tol=1e-3,
        p99_tol=1e-2,
        total_tol=1e-3,
        notes="interpolated cross-section table (xsect_phabs_aspl.fits); Xset.xsect "
        "pinned to the table provenance (see scripts/xspec_comparison_report.py probe).",
    ),
    XspecComparison(
        name="wabs",
        xspec_expression="wabs*powerlaw",
        jaxspec_factory=lambda: Wabs() * Powerlaw(),
        param_map={"wabs.nH": "wabs_1.nh", **_PL_MAP},
        param_sets=_flat_powerlaw(
            (
                {"wabs.nH": 0.1},
                {"wabs.nH": 1.0},
                {"wabs.nH": 5.0},
            )
        ),
        grid=(0.15, 10.0, 2000, "log"),
        settings=ANGR,
        median_tol=1e-3,
        p99_tol=1e-2,
        total_tol=1e-3,
        notes="interpolated cross-section table (xsect_wabs_angr.fits, Morrison & McCammon 1983).",
    ),
    XspecComparison(
        name="gabs",
        xspec_expression="gabs*powerlaw",
        jaxspec_factory=lambda: Gabs() * Powerlaw(),
        param_map={
            "gabs.LineE": "gabs_1.E0",
            "gabs.Sigma": "gabs_1.sigma",
            "gabs.Strength": "gabs_1.tau",
            **_PL_MAP,
        },
        # the (1.0, 0.5, 1.0) set probes the [0, inf) truncation renormalization
        param_sets=_flat_powerlaw(
            (
                {"gabs.LineE": 2.0, "gabs.Sigma": 0.1, "gabs.Strength": 0.5},
                {"gabs.LineE": 6.4, "gabs.Sigma": 0.5, "gabs.Strength": 2.0},
                {"gabs.LineE": 1.0, "gabs.Sigma": 0.5, "gabs.Strength": 1.0},
            )
        ),
        grid=(0.1, 15.0, 2000, "log"),
    ),
    XspecComparison(
        name="highecut",
        xspec_expression="highecut*powerlaw",
        jaxspec_factory=lambda: Highecut() * Powerlaw(),
        param_map={
            "highecut.cutoffE": "highecut_1.Ec",
            "highecut.foldE": "highecut_1.Ef",
            **_PL_MAP,
        },
        param_sets=_flat_powerlaw(
            (
                {"highecut.cutoffE": 10.0, "highecut.foldE": 15.0},
                {"highecut.cutoffE": 5.0, "highecut.foldE": 50.0},
            )
        ),
        grid=(1.0, 100.0, 2000, "log"),
    ),
    XspecComparison(
        name="zedge",
        xspec_expression="zedge*powerlaw",
        jaxspec_factory=lambda: Zedge() * Powerlaw(),
        param_map={
            "zedge.edgeE": "zedge_1.Ec",
            "zedge.MaxTau": "zedge_1.D",
            "zedge.Redshift": "zedge_1.z",
            **_PL_MAP,
        },
        # the z=0.3 set specifically probes the rest-frame threshold
        param_sets=_flat_powerlaw(
            (
                {"zedge.edgeE": 7.1, "zedge.MaxTau": 1.0, "zedge.Redshift": 0.0},
                {"zedge.edgeE": 0.87, "zedge.MaxTau": 0.5, "zedge.Redshift": 0.0},
                {"zedge.edgeE": 7.1, "zedge.MaxTau": 1.0, "zedge.Redshift": 0.3},
            )
        ),
    ),
    XspecComparison(
        name="tbpcf",
        xspec_expression="TBpcf*powerlaw",
        jaxspec_factory=lambda: Tbpcf() * Powerlaw(),
        # XSPEC TBpcf has a redshift parameter jaxspec lacks; left at 0
        param_map={"TBpcf.nH": "tbpcf_1.nh", "TBpcf.pcf": "tbpcf_1.f", **_PL_MAP},
        param_sets=_flat_powerlaw(
            (
                {"TBpcf.nH": 1.0, "TBpcf.pcf": 0.5},
                {"TBpcf.nH": 5.0, "TBpcf.pcf": 0.9},
                {"TBpcf.nH": 0.5, "TBpcf.pcf": 0.2},
            )
        ),
        grid=(0.1, 10.0, 2000, "log"),
        settings=WILM,
        notes="the (1-f) unabsorbed floor keeps every bin flux-carrying, so unlike "
        "tbabs this runs at the default tolerances.",
    ),
    XspecComparison(
        name="expfac",
        xspec_expression="expfac*powerlaw",
        jaxspec_factory=lambda: Expfac() * Powerlaw(),
        param_map={
            "expfac.Ampl": "expfac_1.A",
            "expfac.Factor": "expfac_1.f",
            "expfac.StartE": "expfac_1.E_c",
            **_PL_MAP,
        },
        param_sets=_flat_powerlaw(
            (
                {"expfac.Ampl": 1.0, "expfac.Factor": 1.0, "expfac.StartE": 0.5},
                {"expfac.Ampl": 2.0, "expfac.Factor": 0.5, "expfac.StartE": 1.0},
            )
        ),
        grid=(0.2, 10.0, 2000, "log"),
    ),
]


# --- Band-flux (calcFlux) cases --------------------------------------------------

FLUX_CASES = [
    XspecComparison(
        name="flux-powerlaw",
        xspec_expression="powerlaw",
        jaxspec_factory=Powerlaw,
        param_map=_PL_MAP,
        param_sets=({"powerlaw.PhoIndex": 2.0, "powerlaw.norm": 1.0},),
    ),
    XspecComparison(
        name="flux-cutoffpl",
        xspec_expression="cutoffpl",
        jaxspec_factory=Cutoffpl,
        param_map={
            "cutoffpl.PhoIndex": "cutoffpl_1.alpha",
            "cutoffpl.HighECut": "cutoffpl_1.beta",
            "cutoffpl.norm": "cutoffpl_1.norm",
        },
        param_sets=({"cutoffpl.PhoIndex": 1.5, "cutoffpl.HighECut": 15.0, "cutoffpl.norm": 1.0},),
    ),
    XspecComparison(
        name="flux-bbody",
        xspec_expression="bbody",
        jaxspec_factory=Blackbody,
        param_map={"bbody.kT": "blackbody_1.kT", "bbody.norm": "blackbody_1.norm"},
        param_sets=({"bbody.kT": 1.0, "bbody.norm": 1.0},),
    ),
    XspecComparison(
        name="flux-tbabs-powerlaw",
        xspec_expression="TBabs*powerlaw",
        jaxspec_factory=lambda: Tbabs() * Powerlaw(),
        param_map={"TBabs.nH": "tbabs_1.nh", **_PL_MAP},
        param_sets=({"TBabs.nH": 0.5, "powerlaw.PhoIndex": 1.7, "powerlaw.norm": 1.0},),
        settings=WILM,
        flux_photon_tol=5e-3,
        flux_energy_tol=5e-3,
        notes="the interpolated cross-section table smooths the sharp edges (see the "
        "tbabs spectrum case), which integrates to a ~0.2% band-flux offset below "
        "1.5 keV.",
    ),
]


# --- APEC family cases ------------------------------------------------------------


def _apec_map(xspec_name: str, symbols: tuple[str, ...] = ()) -> dict[str, str]:
    mapping = {
        f"{xspec_name}.kT": "apec_1.kT",
        f"{xspec_name}.Redshift": "apec_1.redshift",
        f"{xspec_name}.norm": "apec_1.norm",
    }
    if xspec_name in ("apec", "bapec"):
        mapping[f"{xspec_name}.Abundanc"] = "apec_1.abund"  # sic, XSPEC spelling
    if xspec_name.startswith("b"):
        mapping[f"{xspec_name}.Velocity"] = "apec_1.velocity"
    for symbol in symbols:
        mapping[f"{xspec_name}.{symbol}"] = f"apec_1.{symbol}"
    return mapping


def _apec_case(xspec_name, broadening, abundances, symbols, param_sets):
    return XspecComparison(
        name=xspec_name,
        xspec_expression=xspec_name,
        jaxspec_factory=lambda: APEC(broadening=broadening, abundances=abundances),
        param_map=_apec_map(xspec_name, symbols),
        param_sets=param_sets,
        grid=(0.5, 10.0, 1500, "log"),
        settings=APEC_SETTINGS,
        median_tol=5e-3,
        p99_tol=3e-2,
        total_tol=5e-3,
        mask_rel=1e-4,
        notes="thresholds match the validated bvapec comparison (median<0.5%, p99<3% "
        "on flux-carrying bins) — residuals are XSPEC's coarser line treatment.",
    )


APEC_CASES = [
    _apec_case(
        "apec",
        False,
        "fixed",
        (),
        (
            {"apec.kT": 1.0, "apec.Abundanc": 1.0, "apec.Redshift": 0.01, "apec.norm": 1.0},
            {"apec.kT": 6.0, "apec.Abundanc": 0.3, "apec.Redshift": 0.05, "apec.norm": 1.0},
        ),
    ),
    _apec_case(
        "bapec",
        True,
        "fixed",
        (),
        (
            {
                "bapec.kT": 2.0,
                "bapec.Abundanc": 1.0,
                "bapec.Redshift": 0.02,
                "bapec.Velocity": 200.0,
                "bapec.norm": 1.0,
            },
            {
                "bapec.kT": 8.0,
                "bapec.Abundanc": 0.5,
                "bapec.Redshift": 0.0,
                "bapec.Velocity": 500.0,
                "bapec.norm": 1.0,
            },
        ),
    ),
    _apec_case(
        "vapec",
        False,
        "free",
        VAPEC_SYMBOLS,
        (
            {
                "vapec.kT": 4.0,
                "vapec.Fe": 0.7,
                "vapec.O": 1.3,
                "vapec.Si": 0.5,
                "vapec.Redshift": 0.03,
                "vapec.norm": 1.0,
            },
            {
                "vapec.kT": 1.0,
                **{f"vapec.{s}": 0.3 for s in VAPEC_SYMBOLS},
                "vapec.Redshift": 0.0,
                "vapec.norm": 1.0,
            },
        ),
    ),
    _apec_case(
        "bvapec",
        True,
        "free",
        VAPEC_SYMBOLS,
        (
            {
                "bvapec.kT": 6.0,
                "bvapec.Fe": 0.7,
                "bvapec.O": 1.3,
                "bvapec.Si": 0.5,
                "bvapec.Redshift": 0.05,
                "bvapec.Velocity": 200.0,
                "bvapec.norm": 1.0,
            },
            {
                "bvapec.kT": 0.8,
                "bvapec.Fe": 1.5,
                "bvapec.Redshift": 0.01,
                "bvapec.Velocity": 400.0,
                "bvapec.norm": 1.0,
            },
        ),
    ),
    _apec_case(
        "vvapec",
        False,
        "all",
        ELEMENT_SYMBOLS,
        (
            {
                "vvapec.kT": 5.0,
                "vvapec.Fe": 2.0,
                "vvapec.Ni": 0.5,
                "vvapec.O": 0.7,
                "vvapec.Na": 1.5,
                "vvapec.Redshift": 0.02,
                "vvapec.norm": 1.0,
            },
        ),
    ),
    _apec_case(
        "bvvapec",
        True,
        "all",
        ELEMENT_SYMBOLS,
        (
            {
                "bvvapec.kT": 3.0,
                "bvvapec.Fe": 0.5,
                "bvvapec.Ca": 2.0,
                "bvvapec.Redshift": 0.04,
                "bvvapec.Velocity": 300.0,
                "bvvapec.norm": 1.0,
            },
        ),
    ),
]
