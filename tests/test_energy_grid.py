"""Integrated-bin redistribution and working-grid validation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxspec.fit._forward_model import _validate_energy_grid
from jaxspec.model.instrument import InstrumentModel, redistribute


def test_redistribute_respects_gaps_and_partial_coverage():
    result = redistribute(
        [2.0, 6.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 7.0],
        [0.5, 1.5, 2.0, 3.0, 6.0, 8.0],
        [1.0, 3.5, 3.0, 7.0, 8.0, 9.0],
    )
    np.testing.assert_allclose(result, [0.0, 4.0, 0.0, 10.0, 2.0, 0.0])


def test_redistribute_matches_independent_overlap_calculation():
    rng = np.random.default_rng(71)
    edges = np.cumsum(rng.uniform(0.01, 1.0, 101))
    source_low, source_high = edges[::2][:-1], edges[1::2]
    flux = rng.uniform(0.0, 5.0, source_low.size)
    target_low = rng.uniform(0.001, edges[-1] + 2, 73)
    target_high = target_low + rng.uniform(0.0, 3.0, target_low.size)
    overlap = np.maximum(
        np.minimum(target_high[:, None], source_high) - np.maximum(target_low[:, None], source_low),
        0.0,
    )
    weights = overlap / (source_high - source_low)
    expected = weights @ flux
    result = jax.jit(redistribute)(flux, source_low, source_high, target_low, target_high)
    np.testing.assert_allclose(result, expected, atol=2e-13)

    jacobian = jax.jacrev(redistribute)(flux, source_low, source_high, target_low, target_high)
    np.testing.assert_allclose(jacobian, weights, atol=2e-15)


def test_redistribute_conserves_flux_over_complete_target_coverage():
    target_edges = np.linspace(0.5, 8.0, 151)
    result = redistribute(
        [2.0, 6.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 7.0],
        target_edges[:-1],
        target_edges[1:],
    )
    np.testing.assert_allclose(result.sum(), 12.0, atol=1e-13)


def test_redistribute_preserves_faint_bins_on_identity_grid():
    flux = jnp.asarray([1e20, 1.0, 2.0], dtype=jnp.float32)
    edges = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    result = redistribute(flux, edges[:-1], edges[1:], edges[:-1], edges[1:])
    np.testing.assert_array_equal(result, flux)
    assert result.dtype == flux.dtype


@pytest.mark.parametrize("empty_source, empty_target", [(True, False), (False, True), (True, True)])
def test_redistribute_empty_grids(empty_source, empty_target):
    source_low = jnp.asarray([] if empty_source else [1.0], dtype=jnp.float32)
    source_high = source_low + 1.0
    target_low = jnp.asarray([] if empty_target else [1.0, 2.0], dtype=jnp.float32)
    target_high = target_low + 1.0
    flux = jnp.ones((2, 3, source_low.size), dtype=jnp.float32)
    result = jax.jit(redistribute)(flux, source_low, source_high, target_low, target_high)
    assert result.shape == (2, 3, target_low.size)
    assert result.dtype == jnp.float32
    np.testing.assert_array_equal(result, 0.0)


def test_redistribute_preserves_leading_axes():
    flux = np.arange(1.0, 25.0).reshape(2, 3, 4)
    result = jax.jit(redistribute)(flux, [1, 2, 3, 4], [2, 3, 4, 5], [1, 3], [3, 5])
    expected = np.stack((flux[..., :2].sum(-1), flux[..., 2:].sum(-1)), axis=-1)
    np.testing.assert_allclose(result, expected)


def test_redistribute_accepts_multidimensional_target_bounds():
    low = np.asarray([[1.0, 2.0], [1.5, 2.5]])
    result = redistribute([4.0, 6.0], [1.0, 2.0], [2.0, 3.0], low, low + 0.5)
    rows = [redistribute([4.0, 6.0], [1.0, 2.0], [2.0, 3.0], row, row + 0.5) for row in low]
    np.testing.assert_allclose(result, np.stack(rows))


def test_redistribute_zero_width_targets_and_boundary_gradients():
    result = redistribute([4.0, 6.0], [1.0, 3.0], [2.0, 4.0], [1.5, 2.5], [1.5, 2.5])
    np.testing.assert_array_equal(result, [0.0, 0.0])

    def enclosed_flux(low, high):
        return redistribute([4.0, 6.0], [1.0, 3.0], [2.0, 4.0], low, high).sum()

    derivatives = jax.jit(jax.grad(enclosed_flux, argnums=(0, 1)))(
        jnp.asarray([1.25]), jnp.asarray([1.75])
    )
    np.testing.assert_allclose(derivatives, [[-4.0], [4.0]])


@pytest.mark.parametrize(
    "args, match",
    [
        (([1], [[1]], [2], [1], [2]), "one-dimensional"),
        (([1], [1, 2], [2], [1], [2]), "matching shapes"),
        (([1], [1], [2], [1], [2, 3]), "matching shapes"),
        (([1, 2], [1], [2], [1], [2]), "last axis"),
        ((1.0, [1], [2], [1], [2]), "last axis"),
    ],
)
def test_redistribute_rejects_incompatible_shapes(args, match):
    with pytest.raises(ValueError, match=match):
        jax.jit(redistribute)(*args)


def test_working_grid_flux_is_mapped_before_response_folding():
    # A flat density of 2 photons/cm²/s/keV, sampled on a separate working grid.
    working_edges = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    native_edges = jnp.asarray([[1.5, 3.0], [2.5, 4.0]])
    transfer = jnp.asarray([[2.0, 0.5], [0.0, 3.0]])
    cache = {"in_energies": native_edges, "transfer_matrix": transfer}
    instrument = InstrumentModel()
    folded = instrument.fold(
        {"continuum": jnp.full(3, 2.0)},
        cache,
        eval_energies=jnp.stack((working_edges[:-1], working_edges[1:])),
    )
    np.testing.assert_allclose(folded["continuum"], [5.0, 6.0])


@pytest.mark.parametrize("grid", [[1.0, np.inf], [np.nan, 2.0], [-np.inf, 2.0]])
def test_validate_energy_grid_rejects_nonfinite_values(grid):
    with pytest.raises(ValueError, match="finite"):
        _validate_energy_grid(grid)


@pytest.mark.parametrize(
    "grid",
    [[1.0, 1.0 + 1e-9], [1e40, 2e40], [1e-50, 2e-50], [2**32 + 1, 2**32 + 2]],
    ids=["collapsed-bin", "overflow", "underflow", "integer-cast"],
)
def test_validate_energy_grid_checks_effective_float32_precision(grid):
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match=r"active JAX precision \(float32\)"):
            _validate_energy_grid(np.asarray(grid))
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("input_grid", [[0.25, 0.5, 1.0], [1, 2, 3]])
def test_validate_energy_grid_accepts_representable_float32_edges(input_grid):
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        grid = _validate_energy_grid(np.asarray(input_grid))
        assert grid.dtype == jnp.float32
        np.testing.assert_array_equal(grid, input_grid)
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("grid", [[1 + 1j, 2 + 1j], ["1", "2"], [False, True]])
def test_working_grid_rejects_non_real_values(grid):
    with pytest.raises(ValueError, match="real numbers"):
        _validate_energy_grid(grid)
