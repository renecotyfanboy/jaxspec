"""Tests for :mod:`jaxspec.fit._forward_model`: input normalisation / grid
validation helpers and the energy-grid-shift folding regression."""

import jax.numpy as jnp
import numpy as np
import pytest

from helpers import single_obsconf
from jaxspec.fit._forward_model import (
    ForwardModel,
    _normalise_background,
    _normalise_observations,
    _validate_energy_grid,
)
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import BackgroundWithError
from jaxspec.model.instrument import ConstantShift, InstrumentModel


@pytest.mark.parametrize(
    "grid, match",
    [
        (np.zeros((2, 2)), "must be 1-D"),
        (np.array([1.0]), "at least 2 points"),
        (np.array([1.0, 1.0, 2.0]), "strictly increasing"),
        (np.array([-1.0, 2.0]), "strictly positive"),
    ],
)
def test_validate_energy_grid_rejects_bad_grids(grid, match):
    with pytest.raises(ValueError, match=match):
        _validate_energy_grid(grid)


def test_normalise_observations_bad_type_raises():
    with pytest.raises(ValueError, match="Invalid type for observations"):
        _normalise_observations(42)


def test_normalise_background_drops_none_entries():
    """A dict background spec keeps only the non-None entries."""
    result = _normalise_background({"a": BackgroundWithError(), "b": None}, ["a", "b"])
    assert set(result) == {"a"}


def test_energy_grid_shift_consistent_with_native_folding():
    """Regression: the user-grid folding path must apply the instrument energy
    shift in the same frame as the native path. With a fine grid and a smooth
    spectrum both paths must produce (nearly) identical folded counts for a
    shifted instrument — the old code shifted the *source* grid labels in
    ``InstrumentModel.fold``, inverting the shift's sign."""
    obs = single_obsconf
    model = Powerlaw() + Powerlaw()
    offset = 0.2

    native_energies = np.asarray(obs.in_energies)
    grid = jnp.geomspace(
        max(native_energies.min() - offset, 1e-2),
        native_energies.max() + 2 * offset,
        6000,
    )

    def build(energy_grid):
        return ForwardModel(
            model,
            obs,
            instrument_model={"data": InstrumentModel(shift=ConstantShift())},
            energy_grid=energy_grid,
        )

    inputs = {
        "spectrum.data.powerlaw_1.alpha": jnp.asarray(1.7),
        "spectrum.data.powerlaw_1.norm": jnp.asarray(1e-3),
        "spectrum.data.powerlaw_2.alpha": jnp.asarray(2.5),
        "spectrum.data.powerlaw_2.norm": jnp.asarray(5e-4),
        "instrument.data.shift.offset": jnp.asarray(offset),
    }

    native = build(None).evaluate(inputs)["data"]["source"]
    gridded = build(grid).evaluate(inputs)["data"]["source"]

    np.testing.assert_allclose(np.asarray(gridded), np.asarray(native), rtol=1e-3)
