"""Fast, network-free unit tests for :class:`jaxspec.model.instrument.PileupModel`
(Davis 2001 CCD pileup).

A tiny synthetic instrument response is built in memory via
``Instrument.from_matrix`` + ``ObsConfiguration.mock_from_instrument`` on a
uniform energy grid (the grid uniformity is the model's design contract). The
single real-response smoke test reuses the shared ``single_obsconf``.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import sparse

from flax import nnx
from helpers import single_obsconf
from jaxspec.data.instrument import Instrument
from jaxspec.data.obsconf import ObsConfiguration
from jaxspec.fit._forward_model import ForwardModel, _build_obs_cache
from jaxspec.model.additive import Powerlaw
from jaxspec.model.instrument import (
    ConstantGain,
    ConstantShift,
    InstrumentModel,
    PileupModel,
)

# Minimal required pileup kwargs (Chandra FRACEXPO / EXPTIME analogues).
PILEUP_KWARGS = {"frac_expo": 1.0, "frame_time": 3.1}


def _synthetic_obsconf(n=40, e_min=0.5, e_max=10.5, area=100.0, exposure=1e4):
    """A tiny in-memory ObsConfiguration on a uniform grid with an identity RMF."""
    edges = np.linspace(e_min, e_max, n + 1)
    e_lo, e_hi = edges[:-1], edges[1:]
    instrument = Instrument.from_matrix(sparse.eye(n), np.full(n, area), e_lo, e_hi, e_lo, e_hi)
    return ObsConfiguration.mock_from_instrument(instrument, exposure=exposure)


def _powerlaw_flux(obs, norm=1e-2, index=1.5):
    """A positive per-bin integrated flux on the obs's native (uniform) grid."""
    in_en = np.asarray(obs.in_energies)
    mid = 0.5 * (in_en[0] + in_en[1])
    width = in_en[1] - in_en[0]
    return jnp.asarray(norm * mid ** (-index) * width)


def test_pileup_construction():
    m = PileupModel(gain=ConstantGain(), shift=ConstantShift(), **PILEUP_KWARGS)
    assert isinstance(m.alpha, nnx.Param)
    assert isinstance(m.psf_frac, nnx.Param)
    assert m.requires_components is True
    # npiled is coerced to a Python int (guards the range() crash); other
    # constants resolve to their documented defaults.
    assert m._constants["npiled"] == 5
    assert isinstance(m._constants["npiled"], int)
    assert m._constants["num_regions"] == 1.0
    assert m._constants["g0"] == 1.0
    # nnx.split/merge must round-trip the module and preserve the static
    # (off-tree) constants dict — this is what the MCMC prior binding relies on.
    merged = nnx.merge(*nnx.split(m))
    assert merged._constants == m._constants


def test_pileup_requires_frame_time():
    """frame_time / frac_expo are required; omitting either raises at construction
    rather than dying with a cryptic ``None * array`` deep inside ``fold``."""
    with pytest.raises(ValueError, match="frame_time"):
        PileupModel(gain=ConstantGain(), frac_expo=1.0)  # frame_time missing
    with pytest.raises(ValueError, match="frac_expo"):
        PileupModel(gain=ConstantGain(), frame_time=3.1)  # frac_expo missing


@pytest.mark.parametrize("npiled", [None, 2, 3.0, 5])
def test_pileup_default_npiled_folds_finite(npiled):
    """Regression: the default ``npiled`` (a float) used to crash
    ``range(2, npiled + 1)`` with a ``TypeError``. Fold must return finite,
    non-negative counts of the right shape for int and float ``npiled`` alike."""
    obs = _synthetic_obsconf()
    kwargs = dict(PILEUP_KWARGS)
    if npiled is not None:
        kwargs["npiled"] = npiled
    m = PileupModel(gain=ConstantGain(), shift=ConstantShift(), **kwargs)
    cache = _build_obs_cache(obs, m, sparse=False)
    out = m.fold(_powerlaw_flux(obs), cache, eval_energies=jnp.asarray(obs.in_energies))

    assert out.shape == (obs.transfer_matrix.data.shape[0],)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(out >= 0))


def test_pileup_requires_eval_energies():
    """``PileupModel.fold`` needs an explicit energy grid; ``None`` raises."""
    obs = _synthetic_obsconf()
    m = PileupModel(gain=ConstantGain(), shift=ConstantShift(), **PILEUP_KWARGS)
    cache = _build_obs_cache(obs, m, sparse=False)
    with pytest.raises(ValueError, match="energy grid"):
        m.fold(_powerlaw_flux(obs), cache)


def test_pileup_no_pileup_limit_matches_base():
    """With ``alpha=0``, ``psf_frac=1``, ``frame_time -> 0`` and unit
    ``num_regions``/``g0``/``frac_expo``, the pileup fold reduces to the base
    ``transfer_matrix @ spectrum`` fold (the higher-order terms vanish)."""
    obs = _synthetic_obsconf()
    flux = _powerlaw_flux(obs)
    eval_energies = jnp.asarray(obs.in_energies)

    pileup = PileupModel(frac_expo=1.0, frame_time=1e-9, num_regions=1.0, g0=1.0, npiled=5)
    pileup.alpha = nnx.Param(jnp.asarray(0.0))
    pileup.psf_frac = nnx.Param(jnp.asarray(1.0))
    pileup_out = np.asarray(
        pileup.fold(flux, _build_obs_cache(obs, pileup, sparse=False), eval_energies=eval_energies)
    )

    base = InstrumentModel()
    base_out = np.asarray(
        base.fold(flux, _build_obs_cache(obs, base, sparse=False), eval_energies=eval_energies)
    )

    mask = base_out > 1e-6  # ignore bins pinned at the base fold's clip floor
    np.testing.assert_allclose(pileup_out[mask], base_out[mask], rtol=1e-6)


def test_pileup_reduces_counts():
    """A bright source piles up: total counts drop vs the no-pileup reference,
    and a longer frame time causes at least as much loss."""
    obs = _synthetic_obsconf()
    bright = _powerlaw_flux(obs, norm=50.0)
    eval_energies = jnp.asarray(obs.in_energies)

    def total(frame_time):
        m = PileupModel(frac_expo=1.0, frame_time=frame_time, npiled=5)
        cache = _build_obs_cache(obs, m, sparse=False)
        return float(jnp.sum(m.fold(bright, cache, eval_energies=eval_energies)))

    no_pileup = total(1e-9)  # negligible frame time -> negligible pileup
    mild = total(1.0)
    strong = total(3.0)

    assert mild < no_pileup  # pileup removes / migrates counts
    assert strong <= mild + 1e-6  # more frame time -> at least as much loss


def test_pileup_split_branches_fold():
    """Posterior-predictive overlays fold a *dict* of per-branch spectra;
    ``fold`` must map the pileup math over the pytree instead of assuming a
    single array (regression for the ``'dict' * BatchTracer`` crash)."""
    obs = _synthetic_obsconf()
    m = PileupModel(gain=ConstantGain(), shift=ConstantShift(), **PILEUP_KWARGS)
    cache = _build_obs_cache(obs, m, sparse=False)
    flux = _powerlaw_flux(obs)
    branches = {"powerlaw_1": flux, "blackbodyrad_1": 0.5 * flux}

    out = m.fold(branches, cache, eval_energies=jnp.asarray(obs.in_energies))

    assert set(out) == set(branches)
    n_channels = obs.transfer_matrix.data.shape[0]
    for arr in out.values():
        assert arr.shape == (n_channels,)
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert bool(jnp.all(arr >= 0))


def test_pileup_real_response_smoke():
    """Fold a default ``PileupModel`` through the real XMM response end-to-end
    via ``ForwardModel.evaluate`` (exercises the real redistribution/grouping/
    area/exposure matrices and the ``ioff``/``tmpar``/``int32`` path)."""
    obs = single_obsconf
    native = np.asarray(obs.in_energies)
    grid = np.linspace(max(native.min(), 1e-2), native.max(), 4000)

    fm = ForwardModel(
        Powerlaw(),
        obs,
        instrument_model={
            "data": PileupModel(gain=ConstantGain(), shift=ConstantShift(), **PILEUP_KWARGS)
        },
        energy_grid=grid,
    )
    inputs = {
        "spectrum.data.powerlaw_1.alpha": jnp.asarray(1.7),
        "spectrum.data.powerlaw_1.norm": jnp.asarray(1e-3),
        "instrument.data.alpha": jnp.asarray(0.5),
        "instrument.data.psf_frac": jnp.asarray(0.95),
        "instrument.data.gain.factor": jnp.asarray(1.0),
        "instrument.data.shift.offset": jnp.asarray(0.0),
    }
    out = fm.evaluate(inputs)["data"]["source"]

    assert out.ndim == 1 and out.shape[0] > 0
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(out >= 0))
