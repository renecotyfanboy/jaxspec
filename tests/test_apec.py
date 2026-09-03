"""Tests for the APEC additive component (the XSPEC apec model family)."""

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from xspec_utils import APEC_CASES, as_pytest_params

# The APEC table is a large artifact downloaded on first use.
from jaxspec.model._additive import apec as _apec

if not _apec.table_is_available():
    pytest.skip("APEC table is neither cached nor reachable", allow_module_level=True)

from jaxspec.model._additive.apec import ELEMENT_SYMBOLS, VAPEC_SYMBOLS
from jaxspec.model.additive import APEC
from jaxspec.model.list import additive_components
from jaxspec.model.multiplicative import Tbabs

ENERGY = jnp.geomspace(0.5, 10.0, 1000)
E_LOW, E_HIGH = ENERGY[:-1], ENERGY[1:]

ALL_VARIANTS = [
    pytest.param(broadening, abundances, id=f"{'b' if broadening else ''}{prefix}apec")
    for broadening in (False, True)
    for abundances, prefix in [("fixed", ""), ("free", "v"), ("all", "vv")]
]


def evaluate(component, e_low=E_LOW, e_high=E_HIGH, **param_values):
    """Set parameter values on the component and return its per-bin photon flux."""
    for key, value in param_values.items():
        getattr(component, key).set_value(jnp.asarray(value))
    return np.asarray(component.integrated_continuum(e_low, e_high))


# --- Construction & registration ----------------------------------------------


def test_registration():
    assert additive_components["APEC"] is APEC


@pytest.mark.parametrize("broadening, abundances", ALL_VARIANTS)
def test_parameter_sets(broadening, abundances):
    component = APEC(broadening=broadening, abundances=abundances)
    params = {name for name, value in vars(component).items() if isinstance(value, nnx.Param)}

    expected = {"kT", "redshift", "norm"}
    if broadening:
        expected |= {"velocity"}
    expected |= {
        "fixed": {"abund"},
        "free": set(VAPEC_SYMBOLS),
        "all": set(ELEMENT_SYMBOLS),
    }[abundances]

    assert params == expected


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="abundances"):
        APEC(abundances="everything")
    with pytest.raises(ValueError, match="abundance table"):
        APEC(abundance_table="not_a_table")
    with pytest.raises(ValueError, match="k_window"):
        APEC(k_window="wide")
    with pytest.raises(ValueError, match="k_window"):
        APEC(k_window=0)


# --- Flux evaluation -----------------------------------------------------------


@pytest.mark.parametrize("broadening, abundances", ALL_VARIANTS)
def test_flux_sanity(broadening, abundances):
    out = evaluate(APEC(broadening=broadening, abundances=abundances), kT=4.0, norm=1e-3)

    assert out.shape == E_LOW.shape
    assert np.all(np.isfinite(out))
    # the FFT-broadened continuum may ring at the ~1e-7 relative level, nothing more
    assert out.min() >= -out.max() * 1e-6
    assert out.max() > 0


def test_variants_agree_at_solar():
    """With every abundance at 1, apec == vapec == vvapec by construction."""
    reference = evaluate(APEC(), kT=3.0)
    assert_allclose(evaluate(APEC(abundances="free"), kT=3.0), reference, rtol=1e-12)
    assert_allclose(evaluate(APEC(abundances="all"), kT=3.0), reference, rtol=1e-12)


def test_fixed_abund_scales_only_vapec_metals():
    """The scalar abundance leaves H, He and trace elements at solar values."""
    metals = {symbol: 0.3 for symbol in VAPEC_SYMBOLS if symbol != "He"}
    reference = evaluate(APEC(abundances="all"), kT=6.0, **metals)
    assert_allclose(evaluate(APEC(), kT=6.0, abund=0.3), reference, rtol=1e-12)


def test_norm_linearity():
    component = APEC()
    assert_allclose(
        evaluate(component, kT=5.0, norm=2e-3), 2 * evaluate(component, norm=1e-3), rtol=1e-7
    )


def test_zero_velocity_matches_unbroadened():
    broadened = APEC(broadening=True)
    assert_allclose(evaluate(broadened, kT=2.0, velocity=0.0), evaluate(APEC(), kT=2.0), rtol=1e-12)


def test_abundance_table_wiring():
    """Abundance parameters expressed in another solar table rescale per element."""
    from jaxspec.util.abundance import abundance_table as abundance_df

    ratio = np.asarray(abundance_df["aspl"], float) / np.asarray(abundance_df["angr"], float)

    aspl = APEC(abundances="all", abundance_table="aspl")
    assert_allclose(np.asarray(aspl._ab_ratio), ratio)

    # solar abundances in 'aspl' == explicitly rescaled abundances in the native 'angr'
    angr_rescaled = {symbol: ratio[i] for i, symbol in enumerate(ELEMENT_SYMBOLS)}
    assert_allclose(
        evaluate(aspl, kT=4.0, norm=1e-3),
        evaluate(APEC(abundances="all"), kT=4.0, norm=1e-3, **angr_rescaled),
        rtol=1e-10,
    )


def test_abundance_table_fixed_mode():
    """In 'fixed' mode the rescaling composes with the single-abund mask: the 12 vapec metals
    get abund * ratio, everything else (H, He, traces) gets ratio alone."""
    from jaxspec.util.abundance import abundance_table as abundance_df

    ratio = np.asarray(abundance_df["aspl"], float) / np.asarray(abundance_df["angr"], float)
    metals = {symbol for symbol in VAPEC_SYMBOLS if symbol != "He"}

    expected_params = {
        symbol: (0.5 if symbol in metals else 1.0) * ratio[i]
        for i, symbol in enumerate(ELEMENT_SYMBOLS)
    }
    assert_allclose(
        evaluate(APEC(abundance_table="aspl"), kT=4.0, abund=0.5),
        evaluate(APEC(abundances="all"), kT=4.0, **expected_params),
        rtol=1e-10,
    )


def test_kT_clips_to_table_grid():
    """Out-of-grid kT clips the emissivity interpolation to the nearest tabulated node."""
    from jaxspec.model._additive.apec import _interp_T, load_apec_table

    table = load_apec_table()
    assert_allclose(
        np.asarray(_interp_T(table["logTg"], table["g"], 500.0)), table["g"][:, -1], rtol=0
    )
    assert_allclose(
        np.asarray(_interp_T(table["logTg"], table["g"], 1e-3)), table["g"][:, 0], rtol=0
    )


def test_suggest_k_window():
    """Window sizing follows the local resolution and has a floor of 32 bins."""
    log_grid = np.geomspace(0.3, 10.0, 2001)
    linear_grid = np.arange(0.3, 12.0, 0.002)  # 2 eV linear bins

    component = APEC(energy_band=(0.3, 10.0))
    assert component.suggest_k_window(log_grid[:-1], log_grid[1:]) == 32
    assert component.suggest_k_window(linear_grid[:-1], linear_grid[1:]) > 32


def test_k_window_auto_sizes_from_the_grid():
    """``k_window="auto"`` resolves the window from the concrete grid it is evaluated on."""
    component = APEC(energy_band=(0.3, 10.0))
    assert component._k_window is None

    log_grid = np.geomspace(0.3, 10.0, 2001)
    evaluate(component, jnp.asarray(log_grid[:-1]), jnp.asarray(log_grid[1:]), kT=4.0)
    assert component._auto_k == component.suggest_k_window(log_grid[:-1], log_grid[1:]) == 32

    linear_grid = np.arange(0.3, 10.0, 0.002)  # a finer grid re-sizes the window
    evaluate(component, jnp.asarray(linear_grid[:-1]), jnp.asarray(linear_grid[1:]), kT=4.0)
    assert component._auto_k == component.suggest_k_window(linear_grid[:-1], linear_grid[1:])
    assert component._auto_k > 32


def test_k_window_auto_matches_explicit_window():
    """The automatic window reproduces an explicit one of the same size exactly."""
    grid = np.geomspace(0.3, 10.0, 2001)
    e_low, e_high = jnp.asarray(grid[:-1]), jnp.asarray(grid[1:])
    values = dict(kT=4.0, velocity=250.0, redshift=0.02, norm=1e-3)
    auto = evaluate(APEC(broadening=True, energy_band=(0.3, 10.0)), e_low, e_high, **values)
    explicit = evaluate(
        APEC(broadening=True, energy_band=(0.3, 10.0), k_window=32), e_low, e_high, **values
    )
    assert_array_equal(auto, explicit)


def test_k_window_auto_traced_grid_falls_back_then_reuses():
    """A traced grid uses the default window until the same grid has been seen concretely."""
    import warnings

    import jax

    component = APEC(energy_band=(0.3, 10.0))
    with pytest.warns(UserWarning, match="traced energy grid"):
        jax.jit(lambda lo, hi: component.integrated_continuum(lo, hi))(E_LOW, E_HIGH)
    assert component._auto_n_bins == -1  # nothing could be sized

    concrete = evaluate(component, kT=4.0)
    assert component._auto_n_bins == E_LOW.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the sized window is reused: no fallback warning
        traced = jax.jit(lambda lo, hi: component.integrated_continuum(lo, hi))(E_LOW, E_HIGH)
    assert_array_equal(np.asarray(traced), concrete)


def test_tables_are_device_arrays_shared_off_the_nnx_tree():
    """Components of one band share the runtime table by identity, outside the NNX state."""
    import jax

    first = APEC(energy_band=(0.5, 10.0))
    second = APEC(broadening=True, abundances="free", energy_band=(0.5, 10.0))
    assert isinstance(first._tables.arrays["cont_F"], jax.Array)
    assert first._tables.arrays["cont_F"] is second._tables.arrays["cont_F"]
    assert nnx.clone(first)._tables is first._tables

    _, _, other = nnx.split(first, nnx.Param, ...)
    assert max(np.size(leaf) for leaf in jax.tree.leaves(other)) <= len(ELEMENT_SYMBOLS)


def test_energy_band_restriction():
    """A band-restricted component reproduces the full table inside its band."""
    energy = jnp.geomspace(2.0, 10.0, 500)
    e_low, e_high = energy[:-1], energy[1:]

    full = evaluate(APEC(broadening=True), e_low, e_high, kT=6.0, velocity=200.0, norm=1e-3)
    banded_component = APEC(broadening=True, energy_band=(2.0, 10.0), k_window=64)
    banded = evaluate(banded_component, e_low, e_high, kT=6.0, velocity=200.0, norm=1e-3)

    from jaxspec.model._additive.apec import load_apec_table

    full_table = load_apec_table()
    assert banded_component._tables.arrays["E0"].size < full_table["E0"].size
    assert banded_component._tables.arrays["E_ref_edges"].size < full_table["E_ref_edges"].size
    assert_allclose(banded, full, rtol=1e-4, atol=full.max() * 1e-7)


@pytest.mark.parametrize("line_chunks", [1, 2, 3, 8, 16])
def test_line_chunks_leave_the_flux_untouched(line_chunks):
    """Line batching changes flux only at the float32 accumulation floor."""
    values = dict(kT=4.0, velocity=300.0, redshift=0.05, norm=1e-3)
    reference = evaluate(APEC(broadening=True, energy_band=(0.5, 10.0), line_chunks=1), **values)
    batched = evaluate(
        APEC(broadening=True, energy_band=(0.5, 10.0), line_chunks=line_chunks), **values
    )

    assert_allclose(batched, reference, rtol=0, atol=reference.max() * 1e-6)


@pytest.mark.parametrize("line_chunks", [2, 3, 8])
def test_line_chunks_are_exact_under_jit(line_chunks):
    """Line batching preserves JIT-compiled flux."""
    import jax

    def flux(chunks):
        component = APEC(broadening=True, energy_band=(0.5, 10.0), line_chunks=chunks)
        for name, value in dict(kT=4.0, velocity=300.0, redshift=0.05, norm=1e-3).items():
            getattr(component, name).set_value(jnp.asarray(value))
        return np.asarray(jax.jit(component.integrated_continuum)(E_LOW, E_HIGH))

    assert_array_equal(flux(line_chunks), flux(1))


def test_line_chunks_leave_the_gradient_untouched():
    """Line batching preserves gradients."""
    import jax

    from jaxspec.model._additive.apec import apec_flux, restricted_table

    static = restricted_table(0.5, 10.0)

    def grad_of(line_chunks):
        def total(kT, abund_vec, sigma_v, z, norm):
            return jnp.sum(
                apec_flux(
                    E_LOW,
                    E_HIGH,
                    kT,
                    abund_vec,
                    sigma_v,
                    z,
                    norm,
                    k_window=128,
                    line_chunks=line_chunks,
                    **static,
                )
            )

        return jax.jit(jax.grad(total, argnums=(0, 1, 2, 3, 4)))

    args = (
        jnp.asarray(4.0),
        jnp.ones(30),
        jnp.asarray(300.0),
        jnp.asarray(0.05),
        jnp.asarray(1e-3),
    )
    reference = grad_of(1)(*args)

    for line_chunks in (2, 3, 8):
        for got, expected in zip(grad_of(line_chunks)(*args), reference):
            got, expected = np.asarray(got), np.asarray(expected)
            assert_allclose(got, expected, rtol=0, atol=np.abs(expected).max() * 1e-6)


def test_suggest_line_chunks_leaves_small_grids_alone():
    """Automatic batching leaves small grids intact."""
    from jaxspec.model._additive.apec import LINE_GRID_MIN_BATCHED, suggest_line_chunks

    assert suggest_line_chunks(500, 32) == 1
    assert suggest_line_chunks(LINE_GRID_MIN_BATCHED, 1) == 1
    assert suggest_line_chunks(2 * LINE_GRID_MIN_BATCHED, 1) > 1
    assert suggest_line_chunks(35_000, 128) > suggest_line_chunks(35_000, 64)
    assert APEC(energy_band=(0.5, 10.0), line_chunks=4)._line_chunks == 4
    assert APEC(energy_band=(0.5, 10.0))._line_chunks is None


def test_non_contiguous_bins_drop_gap_flux():
    """On a gapped grid, every requested bin still gets its exact per-bin integral."""
    energy = np.geomspace(2.0, 10.0, 65)
    component = APEC()

    full = evaluate(component, jnp.asarray(energy[:-1]), jnp.asarray(energy[1:]), kT=6.0)
    gapped = evaluate(component, jnp.asarray(energy[:-1][::2]), jnp.asarray(energy[1:][::2]))

    assert_allclose(gapped, full[::2], rtol=1e-5, atol=full.max() * 1e-7)


def test_golden_regression():
    """Pin the packaged table and flux calculation to a fixed reference spectrum."""
    golden = np.array(
        [
            8.747585658476269e-06,
            8.464542152211487e-06,
            8.377475787980345e-06,
            8.10348018973792e-06,
            8.767128524332193e-06,
            7.3499991458275365e-06,
            7.160885415341173e-06,
            7.080141170465946e-06,
            6.598943797448115e-06,
            6.682849660163545e-06,
            6.0677243658118605e-06,
            5.782767495108239e-06,
            5.686548984042072e-06,
            5.545077216717291e-06,
            5.02217603974503e-06,
            4.7377416867389865e-06,
            4.548867503894613e-06,
            4.289873425429776e-06,
            4.009101619344152e-06,
            3.846551191254229e-06,
            3.5850376793959667e-06,
            3.33335299162623e-06,
            5.3933159081174845e-06,
            8.215893294555353e-06,
            2.6139576233754804e-06,
            2.524913541974423e-06,
            3.0876202697048746e-06,
            2.5577855714061457e-06,
            2.126099279196294e-06,
            1.8216416756242863e-06,
            1.6300877010553572e-06,
            1.4735602329603932e-06,
        ]
    )

    energy = np.geomspace(2.0, 10.0, 33)
    got = evaluate(
        APEC(broadening=True, abundances="free"),
        jnp.asarray(energy[:-1]),
        jnp.asarray(energy[1:]),
        kT=6.0,
        velocity=200.0,
        redshift=0.05,
        norm=1e-3,
        Fe=0.7,
        O=1.3,
        Si=0.5,
    )

    assert_allclose(got, golden, rtol=1e-3, atol=golden.max() * 1e-6)


# --- Integration with the fitting machinery -------------------------------------


def test_fakeit(obsconfs):
    from numpy.random import default_rng

    from jaxspec.data.util import fakeit_for_multiple_parameters

    rng = default_rng(42)
    size = 4
    model = Tbabs() * APEC()
    parameters = {
        "tbabs_1.nh": rng.uniform(0.1, 0.4, size=size),
        "apec_1.kT": rng.uniform(2.0, 8.0, size=size),
        "apec_1.abund": rng.uniform(0.3, 1.5, size=size),
        "apec_1.redshift": np.full(size, 0.01),
        "apec_1.norm": rng.exponential(1e-3, size=size),
    }

    spectra = fakeit_for_multiple_parameters(obsconfs[0], model, parameters, apply_stat=False)

    assert spectra.shape[0] == size
    assert np.all(np.isfinite(spectra))


@pytest.mark.slow
def test_mcmc_smoke(obsconfs):
    import numpyro.distributions as dist

    from helpers import SHORT_MCMC_FIT

    from jaxspec.fit import MCMCFitter

    model = Tbabs() * APEC(energy_band=(0.3, 12.0))
    prior = {
        "spectrum.tbabs_1.nh": dist.Uniform(0.01, 1.0),
        "spectrum.apec_1.kT": dist.Uniform(0.5, 10.0),
        "spectrum.apec_1.abund": dist.Uniform(0.1, 2.0),
        "spectrum.apec_1.redshift": dist.Uniform(0.0, 0.01),
        "spectrum.apec_1.norm": dist.LogUniform(1e-8, 1e-2),
    }

    result = MCMCFitter(model, prior, obsconfs[0]).fit(**SHORT_MCMC_FIT)

    assert np.all(np.isfinite(np.asarray(result.input_parameters["spectrum.apec_1.kT"])))


# --- XSPEC comparison (requires a HEASOFT install with PyXSPEC) -----------------
# Run via `bash scripts/run_xspec_tests.sh`; skips cleanly without XSPEC.


@pytest.mark.xspec
@pytest.mark.parametrize("case, pset", as_pytest_params(APEC_CASES))
def test_apec_family_vs_xspec(case, pset):
    """Every APEC variant against its XSPEC counterpart, evaluated live."""
    pytest.importorskip("xspec")
    from xspec_utils import assert_close_to_xspec

    assert_close_to_xspec(case, pset)


@pytest.mark.xspec
@pytest.mark.slow
def test_against_xspec_bvapec():
    pytest.importorskip("xspec")
    from xspec import AllModels, Model, Xset

    Xset.chatter = 0
    Xset.abund = "angr"
    Xset.addModelString("APECROOT", "3.1.3")  # match the packaged table
    Xset.addModelString("APECTHERMAL", "yes")
    Xset.addModelString("APECBROADPSEUDO", "yes")

    AllModels.clear()
    AllModels.setEnergies("2.0 10.0 1000 lin")

    xspec_model = Model("bvapec")
    xspec_model.bvapec.kT = 6.0
    xspec_model.bvapec.O = 1.3
    xspec_model.bvapec.Si = 0.5
    xspec_model.bvapec.Fe = 0.7
    xspec_model.bvapec.Redshift = 0.05
    xspec_model.bvapec.Velocity = 200.0
    xspec_model.bvapec.norm = 1.0

    edges = np.linspace(2.0, 10.0, 1001)
    expected = np.array(xspec_model.values(0))  # per-bin integrated photon flux

    got = evaluate(
        APEC(broadening=True, abundances="free"),
        jnp.asarray(edges[:-1]),
        jnp.asarray(edges[1:]),
        kT=6.0,
        velocity=200.0,
        redshift=0.05,
        norm=1.0,
        Fe=0.7,
        O=1.3,
        Si=0.5,
    )

    mask = expected > expected.max() * 1e-4
    rel = np.abs(got[mask] - expected[mask]) / expected[mask]

    assert np.median(rel) < 5e-3
    assert np.percentile(rel, 99) < 3e-2
