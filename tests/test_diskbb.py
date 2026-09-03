"""Accuracy, normalization and differentiability tests for the multicolor disk models.

``Diskbb`` and ``Diskpbb`` are the two components whose ``norm`` deliberately
departs from XSPEC's convention: it is the photon flux over ``_flux_band``. The
tests below split that into the two independent claims — the *shape* kernels
reproduce the disk integral, and ``continuum`` divides that shape by its own
band integral so ``norm`` really is the band flux. Formal agreement with XSPEC
lives in ``xspec_utils.py``, which matches the norm through XSPEC's calcFlux.

Reference shape fluxes were computed with ``scipy.integrate.quad`` on the
numerically stable form of the disk integral

    F(E) = 2.78e-3 * (0.75/p) / Tin
           * int_0^Tin (kT/Tin)^(-2/p-1) E^2 exp(-E/kT)/(1 - exp(-E/kT)) dkT

with ``limit=400`` (relative accuracy well below 1e-9 over the tested band),
for a unit disk normalization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xspec_utils import DISK_CASES, as_pytest_params

from jaxspec.model._additive.disk import (
    DEFAULT_FLUX_BAND,
    STANDARD_RADIAL_EXPONENT,
    _disk_band_photon_flux,
    _disk_photon_flux,
    _exp_normalized,
    _usable_log_max,
)
from jaxspec.model.additive import Diskbb, Diskpbb

E_REF = np.array(
    [
        2.0000000000e-01,
        3.5896458428e-01,
        6.4427786382e-01,
        1.1563646777e00,
        2.0754698291e00,
        3.7251008218e00,
        6.6858963394e00,
        1.2000000000e01,
    ]
)

# (Tin, p) -> reference photon flux at E_REF for a unit disk normalization
FLUX_REF = {
    (0.3, 0.75): [
        5.5256152268e-04,
        3.0686619867e-04,
        1.3066249340e-04,
        3.0958864075e-05,
        2.1646410123e-06,
        1.4356610626e-08,
        1.2580878248e-12,
        4.4318380882e-20,
    ],
    (1.0, 0.75): [
        1.5392840758e-02,
        1.0100078726e-02,
        6.3255579924e-03,
        3.5474349158e-03,
        1.5442274397e-03,
        3.8381534637e-04,
        2.9556647432e-05,
        2.3498154488e-07,
    ],
    (3.0, 0.75): [
        2.9304984733e-01,
        1.9735737002e-01,
        1.3178870390e-01,
        8.6152314844e-02,
        5.3474080499e-02,
        2.9367538633e-02,
        1.2194075978e-02,
        2.7366269308e-03,
    ],
    (1.0, 0.55): [
        2.2722924465e-01,
        8.6905926145e-02,
        3.2817078366e-02,
        1.1811772364e-02,
        3.6265007114e-03,
        7.1002134064e-04,
        4.7473195526e-05,
        3.4971885842e-07,
    ],
    (1.0, 1.0): [
        3.0330743813e-03,
        2.7457368713e-03,
        2.2873107634e-03,
        1.6272882089e-03,
        8.5066329155e-04,
        2.4013513056e-04,
        2.0017335942e-05,
        1.6653936779e-07,
    ],
}

BAND = (0.5, 8.0)


def _band_flux(model_cls, band, **params):
    """Photon flux the model actually emits over ``band``, on a fine grid."""
    name = model_cls.__name__.lower()
    grid = jnp.geomspace(band[0], band[1], 5001)
    model = model_cls(flux_band=band)
    return float(
        model.photon_flux(
            grid[:-1],
            grid[1:],
            params={f"{name}_1.{k}": v for k, v in params.items()},
            n_points=5,
        ).sum()
    )


def _disk_model(model_cls, tin=1.3, norm=3.7e-3, p=0.65, band=BAND):
    model = model_cls(flux_band=band)
    model.Tin.set_value(jnp.asarray(tin))
    model.norm.set_value(jnp.asarray(norm))
    if model_cls is Diskpbb:
        model.p.set_value(jnp.asarray(p))
    return model


# --- Shape kernels ---------------------------------------------------------------


@pytest.mark.parametrize("tin,p", FLUX_REF)
def test_disk_shape_accuracy(tin, p):
    """The kernels reproduce the independent references, table and quadrature alike."""
    computed = np.array(_disk_photon_flux(jnp.asarray(E_REF), tin, p))

    assert np.allclose(computed, np.array(FLUX_REF[(tin, p)]), rtol=1e-8)


def test_diskbb_uses_the_tabulated_kernel():
    """``Diskbb`` reaches the interpolated kernel, not the quadrature.

    The quadrature raises each node to ``s - 1``; the table never does. Converting
    ``p`` to an array upstream of `_disk_log_shape` selects the quadrature without
    changing any result, so only the compiled code shows it.
    """
    energy = jnp.geomspace(0.5, 10.0, 100)

    diskbb = jax.jit(Diskbb().continuum).lower(energy).compile().as_text()
    diskpbb = jax.jit(Diskpbb().continuum).lower(energy).compile().as_text()

    assert " power(" not in diskbb
    assert " power(" in diskpbb


def test_diskbb_matches_diskpbb_at_standard_exponent():
    """Cross-validates the two kernels against each other on the physics they share."""
    energy = jnp.geomspace(0.2, 12.0, 100)
    diskbb, diskpbb = Diskbb(), Diskpbb()
    for model in (diskbb, diskpbb):
        model.Tin.set_value(jnp.asarray(1.3))
        model.norm.set_value(jnp.asarray(3.7e-3))
    diskpbb.p.set_value(jnp.asarray(STANDARD_RADIAL_EXPONENT))

    assert np.allclose(
        np.array(diskbb.continuum(energy)),
        np.array(diskpbb.continuum(energy)),
        rtol=1e-8,
    )


# --- Normalization contract ------------------------------------------------------


@pytest.mark.parametrize("band", [DEFAULT_FLUX_BAND, BAND, (2.0, 10.0), (0.1, 100.0)])
@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_is_band_photon_flux(model_cls, band):
    """``norm`` must equal the photon flux the model emits over its band."""
    flux_value = 3.7e-3

    band_flux = _band_flux(model_cls, band, norm=flux_value)

    assert np.isclose(band_flux, flux_value, rtol=1e-4)


@pytest.mark.parametrize("p", [0.5, 0.6, 0.9, 1.5])
def test_diskpbb_norm_is_band_photon_flux_off_canonical_p(p):
    """The contract must also hold away from p=0.75, on the quadrature kernel."""
    flux_value = 3.7e-3

    band_flux = _band_flux(Diskpbb, DEFAULT_FLUX_BAND, norm=flux_value, p=p)

    assert np.isclose(band_flux, flux_value, rtol=1e-4)


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_continuum_is_shape_over_band_integral(model_cls):
    """``continuum`` is the shape kernel divided by its own band integral."""
    energy = jnp.geomspace(0.2, 12.0, 100)
    tin, flux_value = 1.3, 3.7e-3

    model = model_cls(flux_band=BAND)
    model.Tin.set_value(jnp.asarray(tin))
    model.norm.set_value(jnp.asarray(flux_value))

    # Match how each model reaches the kernels: ``Diskbb`` fixes the exponent at trace
    # time and is interpolated, ``Diskpbb`` traces ``p`` and is integrated. Passing the
    # bare float for both would compare the two kernels instead of the identity.
    exponent = STANDARD_RADIAL_EXPONENT
    if model_cls is Diskpbb:
        exponent = jnp.asarray(STANDARD_RADIAL_EXPONENT)
    shape = _disk_photon_flux(energy, tin, exponent)
    band_integral = _disk_band_photon_flux(tin, exponent, *BAND)

    assert np.allclose(
        np.array(model.continuum(energy)),
        np.array(flux_value * shape / band_integral),
        rtol=1e-12,
    )


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_flux_band_rescales_without_changing_shape(model_cls):
    """Changing the band rescales the continuum by a constant and nothing else."""
    energy = jnp.geomspace(0.5, 8.0, 60)

    wide = np.array(model_cls(flux_band=(0.3, 12.0)).continuum(energy))
    narrow = np.array(model_cls(flux_band=(2.0, 8.0)).continuum(energy))
    ratio = wide / narrow

    assert np.allclose(ratio, ratio[0], rtol=1e-10)
    assert not np.isclose(ratio[0], 1.0)


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_flux_band_property_and_default(model_cls):
    assert model_cls()._flux_band == DEFAULT_FLUX_BAND
    assert model_cls(flux_band=BAND)._flux_band == BAND


@pytest.mark.parametrize(
    "band", [(8.0, 0.5), (0.0, 8.0), (0.5, float("inf")), (0.5, float("nan")), (-1.0, 8.0)]
)
@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_flux_band_validation(model_cls, band):
    """Bad bands must raise, not silently produce a NaN continuum."""
    with pytest.raises(ValueError):
        model_cls(flux_band=band)


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_continuum_vanishes_at_non_positive_energy(model_cls):
    """The kernels evaluate their log on a safe branch at E <= 0; the mask is what
    turns that placeholder back into the zero flux a non-positive energy must have."""
    energy = jnp.asarray([-1.0, 0.0, 1.0])

    continuum = np.asarray(model_cls().continuum(energy))

    assert continuum[0] == 0.0
    assert continuum[1] == 0.0
    assert continuum[2] > 0.0


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_is_exactly_linear(model_cls):
    """The continuum is a plain multiple of ``norm`` — exact at zero, exact under
    differentiation, and with an identically zero second derivative."""
    energy = jnp.geomspace(0.5, 8.0, 40)

    def flux(norm):
        model = _disk_model(model_cls, norm=norm)
        return jnp.sum(model.continuum(energy))

    unit = float(flux(jnp.asarray(1.0)))
    for norm in [0.0, 1e-8, 1e-4, 1.0, -1e-4]:
        assert np.isclose(float(flux(jnp.asarray(norm))), norm * unit, rtol=1e-12)
        assert np.isclose(float(jax.grad(flux)(jnp.asarray(norm))), unit, rtol=1e-12)
        assert float(jax.hessian(flux)(jnp.asarray(norm))) == 0.0


def test_band_integrals_broadcast():
    """The band integrals run on posterior-shaped parameters, as fakeit needs."""
    tin = jnp.linspace(0.8, 1.5, 400).reshape(4, 100)
    p = jnp.linspace(0.6, 0.9, 400).reshape(4, 100)

    diskbb_flux = _disk_band_photon_flux(tin, STANDARD_RADIAL_EXPONENT, *BAND)
    assert diskbb_flux.shape == (4, 100)
    assert bool(jnp.all(jnp.isfinite(diskbb_flux)))

    diskpbb_flux = _disk_band_photon_flux(tin, p, *BAND)
    assert diskpbb_flux.shape == (4, 100)
    assert bool(jnp.all(jnp.isfinite(diskpbb_flux)))

    # Mixed shapes: scalar Tin against a vector of p samples, consistent with
    # the scalar call one sample at a time.
    p_vec = jnp.linspace(0.55, 0.95, 7)
    mixed = _disk_band_photon_flux(1.2, p_vec, *BAND)
    assert mixed.shape == (7,)
    assert np.allclose(
        np.array(mixed[3]), np.array(_disk_band_photon_flux(1.2, p_vec[3], *BAND)), rtol=1e-12
    )


# --- Numerical stability ---------------------------------------------------------


@pytest.mark.parametrize("band", [DEFAULT_FLUX_BAND, (2.0, 10.0), (10.0, 79.0)])
@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_cold_disk_stays_finite(model_cls, band):
    """Cold-disk values and first two derivatives remain finite in log space."""
    e_low = jnp.geomspace(band[0], band[1], 20)
    name = model_cls.__name__.lower()

    def flux_sum(tin):
        return (
            model_cls(flux_band=band)
            .photon_flux(e_low, e_low * 1.05, params={f"{name}_1.Tin": tin})
            .sum()
        )

    for tin in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        assert jnp.isfinite(flux_sum(tin)), f"value not finite at Tin={tin}"
        assert jnp.isfinite(jax.grad(flux_sum)(tin)), f"gradient not finite at Tin={tin}"
        assert jnp.isfinite(jax.hessian(flux_sum)(tin)), f"hessian not finite at Tin={tin}"

    # Finiteness alone would also pass on a saturated plateau.
    assert jax.grad(flux_sum)(band[0] / 90.0) != 0.0


@pytest.mark.parametrize("norm", [1e-4, 1e-8, 1e-12])
def test_cap_applies_to_the_scaled_value_not_the_bare_ratio(norm):
    """The cap applies to ``norm * exp(ratio)``: a small ``norm`` brings a large
    ratio back into range, and the result must not be clipped."""
    log_ratio = jnp.asarray(_usable_log_max(jnp.zeros(()).dtype) - 0.5 * np.log(norm))
    expected = float(np.longdouble(norm) * np.exp(np.longdouble(float(log_ratio))))
    assert np.isfinite(expected)  # the scaled value really is representable

    assert np.isclose(float(_exp_normalized(log_ratio, jnp.asarray(norm))), expected, rtol=1e-12)


# --- Differentiability -----------------------------------------------------------


def _photon_flux_sum(model_cls, param, value):
    e_low = jnp.geomspace(0.2, 12.0, 50)
    e_high = e_low * 1.05

    return model_cls().photon_flux(e_low, e_high, params={param: value}).sum()


@pytest.mark.parametrize(
    "model_cls,param,value",
    [
        (Diskbb, "diskbb_1.Tin", 1.3),
        (Diskbb, "diskbb_1.norm", 1e-2),
        (Diskpbb, "diskpbb_1.Tin", 1.3),
        (Diskpbb, "diskpbb_1.p", 0.65),
    ],
)
def test_disk_derivatives_all_modes(model_cls, param, value):
    """All differentiation modes return finite, consistent derivatives."""

    def f(v):
        return _photon_flux_sum(model_cls, param, v)

    grad = jax.grad(f)(value)
    fwd = jax.jacfwd(f)(value)
    hess = jax.hessian(f)(value)

    assert jnp.isfinite(grad)
    assert jnp.isfinite(fwd)
    assert jnp.isfinite(hess)
    assert np.isclose(float(grad), float(fwd), rtol=1e-12)

    eps = 1e-6 * value
    fd = (f(value + eps) - f(value - eps)) / (2 * eps)
    assert np.isclose(float(grad), float(fd), rtol=1e-5)


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_disk_traces_under_jit_and_vmap(model_cls):
    """Disk evaluation remains traceable under JIT, vmap, and differentiation."""
    e_low = jnp.geomspace(0.5, 8.0, 30)
    e_high = e_low * 1.05
    name = model_cls.__name__.lower()

    def flux_sum(tin):
        return model_cls().photon_flux(e_low, e_high, params={f"{name}_1.Tin": tin}).sum()

    jitted = jax.jit(flux_sum)(1.3)
    vmapped = jax.vmap(flux_sum)(jnp.linspace(0.8, 2.0, 5))
    grad_jit = jax.jit(jax.grad(flux_sum))(1.3)

    assert jnp.isfinite(jitted)
    assert vmapped.shape == (5,) and bool(jnp.all(jnp.isfinite(vmapped)))
    assert jnp.isfinite(grad_jit)
    assert np.isclose(float(jitted), float(flux_sum(1.3)), rtol=1e-12)


# --- The published XSPEC normalization -------------------------------------------
#
# `norm` is a band photon flux, but K_X = cos(i)(r_in/d)^2 is what gives the inner
# disk radius, so the components publish it as a derived quantity on every fit.
# Registration and naming are covered in test_derived_quantities.py; these tests pin
# the physics — that the published number really is the XSPEC normalization.


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_xspec_is_norm_over_the_band_integral(model_cls):
    model = _disk_model(model_cls)
    p = STANDARD_RADIAL_EXPONENT if model_cls is Diskbb else 0.65
    band_flux = _disk_band_photon_flux(1.3, p, *BAND)

    assert np.isclose(
        float(model.derived_quantities()["norm_xspec"]), 3.7e-3 / float(band_flux), rtol=1e-12
    )


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_continuum_is_norm_xspec_times_the_unit_disk_shape(model_cls):
    """The defining property, stated without XSPEC: the kernels are the spectrum at
    unit disk normalization, so scaling them by the published K_X must reproduce
    exactly what the model emits."""
    energy = jnp.geomspace(0.2, 12.0, 100)
    model = _disk_model(model_cls)
    p = STANDARD_RADIAL_EXPONENT if model_cls is Diskbb else 0.65
    shape = _disk_photon_flux(energy, 1.3, p)

    assert np.allclose(
        np.array(model.continuum(energy)),
        np.array(model.derived_quantities()["norm_xspec"] * shape),
        rtol=1e-12,
    )


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_xspec_is_exactly_linear_in_norm(model_cls):
    """A plain multiple of ``norm``, exact at zero."""

    def norm_xspec(norm):
        return _disk_model(model_cls, norm=norm).derived_quantities()["norm_xspec"]

    unit = float(norm_xspec(jnp.asarray(1.0)))
    for norm in [0.0, 1e-8, 1e-4, 1.0, -1e-4]:
        assert np.isclose(float(norm_xspec(jnp.asarray(norm))), norm * unit, rtol=1e-12)
        assert np.isclose(float(jax.grad(norm_xspec)(jnp.asarray(norm))), unit, rtol=1e-12)
        assert float(jax.hessian(norm_xspec)(jnp.asarray(norm))) == 0.0


@pytest.mark.parametrize("band", [DEFAULT_FLUX_BAND, (10.0, 79.0)])
@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_xspec_stays_finite_for_cold_disks(model_cls, band):
    """The published ratio stays finite as the band integral underflows."""

    def norm_xspec(tin):
        return _disk_model(model_cls, tin=tin, band=band).derived_quantities()["norm_xspec"]

    for tin in [1e-1, 1e-3, 1e-6, 1e-8]:
        assert jnp.isfinite(norm_xspec(jnp.asarray(tin))), f"value not finite at Tin={tin}"
        assert jnp.isfinite(jax.grad(norm_xspec)(jnp.asarray(tin))), f"grad not finite at {tin}"


@pytest.mark.parametrize("model_cls", [Diskbb, Diskpbb])
def test_norm_xspec_is_zero_for_a_non_positive_temperature(model_cls):
    """Zero rather than NaN, which would propagate to ``az.rhat`` for the whole fit."""
    model = _disk_model(model_cls, tin=0.0)

    assert float(model.derived_quantities()["norm_xspec"]) == 0.0


# --- XSPEC comparison (requires a HEASOFT install with PyXSPEC) -----------------
# Run via `bash scripts/run_xspec_tests.sh`; skips cleanly without XSPEC.
#
# The harness maps XSPEC's norm through its own calcFlux over the band, comparing
# the spectral shape and jaxspec's band-flux contract independently.


@pytest.mark.xspec
@pytest.mark.parametrize("case, pset", as_pytest_params(DISK_CASES))
def test_disk_models_vs_xspec(case, pset):
    pytest.importorskip("xspec")
    from xspec_utils import assert_close_to_xspec

    assert_close_to_xspec(case, pset)


@pytest.mark.xspec
@pytest.mark.parametrize("case, pset", as_pytest_params(DISK_CASES))
def test_published_norm_xspec_recovers_the_xspec_normalization(case, pset):
    """The harness sets ``norm`` to the band photon flux XSPEC reports for ``pset``,
    so the published normalization must return the value XSPEC was given.

    The tolerance covers XSPEC's diskbb approximation, ~0.36% off the exact integral.
    """
    pytest.importorskip("xspec")

    model = case.jaxspec_factory()
    params = case.jaxspec_params(pset)
    component = next(iter(params)).split(".")[0]
    for path, value in params.items():
        getattr(model, path.split(".", 1)[1]).set_value(jnp.asarray(value))

    published = float(model.derived_quantities()["norm_xspec"])
    expected = pset[f"{case.xspec_expression}.norm"]

    assert np.isclose(published, expected, rtol=5e-3 if component == "diskbb_1" else 2e-2)
