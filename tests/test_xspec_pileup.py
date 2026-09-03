"""Live XSPEC comparison: :class:`jaxspec.model.instrument.PileupModel` vs
XSPEC's ``pileup`` convolution model (both implement Davis 2001 CCD pileup;
``pileup`` is XSPEC's name for what Sherpa/ISIS call ``jdpileup``).

This test runs **only** in a HEASOFT/PyXSPEC environment and is gated by
``pytest.importorskip("xspec")`` + the ``xspec`` marker, so it skips cleanly
everywhere else. Run it with either::

    bash scripts/run_xspec_tests.sh tests/test_xspec_pileup.py

or, with HEASOFT living inside the jaxspec conda env::

    HEADAS=$CONDA_PREFIX/heasoft; source "$HEADAS/BUILD_DIR/headas-init.sh"
    pytest tests/test_xspec_pileup.py -m xspec

It folds the same power-law source through the same XMM/PN response on both
sides -- XSPEC via ``pileup*powerlaw`` + ``Model.folded(1)`` (the folded
count-rate precedent used in ``bvapec/*.py``), jaxspec via a ``ForwardModel``
with a :class:`PileupModel` -- then compares the two folded spectra resampled
onto a shared coarse energy grid (robust to channel grouping / energy-trim
differences between the two toolkits).

Parameter mapping (XSPEC ``pileup`` model, per ``spectral/manager/model.dat``):

    XSPEC pileup   PileupModel      value
    ------------   -----------      -----
    fr_time        frame_time       3.2 s  (frame / readout time)
    max_ph         npiled           5      (max piled photons per frame)
    g0             g0               1.0    (grade-0 probability)
    alpha          alpha            0.5    (grade migration)
    psffrac        psf_frac         0.95   (PSF fraction)
    nregions       num_regions      1      (detection cells)
    fracexpo       frac_expo        1.0    (fractional exposure)

Agreement observed against HEASOFT 6.35 (XSPEC 12.35) on the bundled PN
response: in-band median relative difference ~0.6%, p99 ~3% (see the tolerances
below). The small residual is expected (FFT offset / normalisation conventions).
"""

import os

import numpy as np
import pytest

from jaxspec.util.online_storage import table_manager

xspec = pytest.importorskip("xspec")

# --- pileup / PileupModel parameters (see module docstring for the mapping) ---
ALPHA = 0.5
G0 = 1.0
PSF_FRAC = 0.95
NUM_REGIONS = 1
FRAME_TIME = 3.2
FRAC_EXPO = 1.0
NPILED = 5

PHOINDEX = 1.8
NORM = 1e-3

LOW_ENERGY, HIGH_ENERGY = 0.5, 8.0
# In-band relative-agreement tolerances on the resampled spectra. Observed
# median ~0.6% / p99 ~3% against HEASOFT 6.35; these leave headroom for
# XSPEC-version and platform variation.
MEDIAN_TOL = 0.02
P99_TOL = 0.06
COMPARE_BINS = np.geomspace(LOW_ENERGY, HIGH_ENERGY, 25)


def _resample(centers, values, bins):
    """Sum per-channel ``values`` into ``bins`` (robust to differing grids)."""
    centers = np.asarray(centers)
    values = np.asarray(values)
    idx = np.digitize(centers, bins) - 1
    out = np.zeros(len(bins) - 1)
    for b in range(len(out)):
        out[b] = values[idx == b].sum()
    return out


def _fetch_pn_response():
    """Fetch the bundled PN PHA/RMF/ARF and return the PHA path (+ its dir)."""
    pha = table_manager.fetch("example_data/NGC7793_ULX4/PN_spectrum_grp20.fits")
    # RMF/ARF are referenced by (relative) filename in the PHA header; fetching
    # them places them next to the PHA so XSPEC finds them once we chdir there.
    table_manager.fetch("example_data/NGC7793_ULX4/PN.rmf")
    table_manager.fetch("example_data/NGC7793_ULX4/PN.arf")
    return pha, os.path.dirname(pha)


def _xspec_folded_rate(pha_basename):
    """XSPEC pileup*powerlaw folded count rate per channel + channel centres.

    Must be called with the cwd set to the directory holding the PHA/RMF/ARF.
    """
    xspec.Xset.chatter = 0
    xspec.AllData.clear()
    xspec.AllModels.clear()

    spectrum = xspec.Spectrum(pha_basename)
    spectrum.ignore(f"0.0-{LOW_ENERGY:.1f} {HIGH_ENERGY:.1f}-**")

    model = xspec.Model("pileup*powerlaw")
    model.pileup.fr_time = FRAME_TIME
    model.pileup.max_ph = NPILED
    model.pileup.g0 = G0
    model.pileup.alpha = ALPHA
    model.pileup.psffrac = PSF_FRAC
    model.pileup.nregions = NUM_REGIONS
    model.pileup.fracexpo = FRAC_EXPO
    model.powerlaw.PhoIndex = PHOINDEX
    model.powerlaw.norm = NORM

    folded = np.asarray(model.folded(1))  # predicted count rate per channel
    centers = np.asarray(spectrum.energies).mean(axis=1)  # (n_channel, 2) -> centre
    return centers, folded


def _jaxspec_folded_rate(pha):
    """jaxspec PileupModel folded count rate per grouped channel + centres."""
    import jax.numpy as jnp

    from jaxspec.data import ObsConfiguration
    from jaxspec.fit._forward_model import ForwardModel
    from jaxspec.model.additive import Powerlaw
    from jaxspec.model.instrument import PileupModel

    obs = ObsConfiguration.from_pha_file(pha, low_energy=LOW_ENERGY, high_energy=HIGH_ENERGY)
    native = np.asarray(obs.in_energies)
    grid = np.linspace(native.min(), native.max(), 6000)

    fm = ForwardModel(
        Powerlaw(),
        obs,
        instrument_model={
            "data": PileupModel(
                frac_expo=FRAC_EXPO,
                frame_time=FRAME_TIME,
                num_regions=NUM_REGIONS,
                g0=G0,
                npiled=NPILED,
            )
        },
        energy_grid=grid,
    )
    inputs = {
        "spectrum.data.powerlaw_1.alpha": jnp.asarray(PHOINDEX),
        "spectrum.data.powerlaw_1.norm": jnp.asarray(NORM),
        "instrument.data.alpha": jnp.asarray(ALPHA),
        "instrument.data.psf_frac": jnp.asarray(PSF_FRAC),
    }
    counts = np.asarray(fm.evaluate(inputs)["data"]["source"])
    rate = counts / float(obs.exposure.data)  # counts -> count rate to match XSPEC
    centers = np.asarray(obs.out_energies).mean(axis=0)
    return centers, rate


@pytest.mark.xspec
def test_pileup_matches_xspec_pileup(monkeypatch):
    pha, pha_dir = _fetch_pn_response()

    # XSPEC resolves the header's (relative) RMF/ARF against the cwd.
    monkeypatch.chdir(pha_dir)

    xc, xr = _xspec_folded_rate(os.path.basename(pha))
    jc, jr = _jaxspec_folded_rate(pha)

    xspec_binned = _resample(xc, xr, COMPARE_BINS)
    jaxspec_binned = _resample(jc, jr, COMPARE_BINS)

    mask = xspec_binned > xspec_binned.max() * 1e-3
    rel = np.abs(jaxspec_binned[mask] - xspec_binned[mask]) / xspec_binned[mask]
    median_rel = float(np.median(rel))
    p99_rel = float(np.percentile(rel, 99))

    # Printed so the actual agreement is visible even on a pass (run with -s).
    print(f"\npileup vs PileupModel: median rel = {median_rel:.3e}, p99 rel = {p99_rel:.3e}")

    assert median_rel < MEDIAN_TOL, (
        f"PileupModel disagrees with XSPEC pileup in-band: median relative "
        f"difference {median_rel:.3e} exceeds {MEDIAN_TOL}."
    )
    assert p99_rel < P99_TOL, (
        f"PileupModel disagrees with XSPEC pileup in-band: p99 relative "
        f"difference {p99_rel:.3e} exceeds {P99_TOL}."
    )
