"""Model descriptions preserve configured topology and current numerical state."""

import copy
import json

from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

from jaxspec.model._serialization import _json_config
from jaxspec.model.abc import AdditiveComponent, OperatorComponent, SpectralModel
from jaxspec.model.additive import Diskbb, Diskpbb, Powerlaw
from jaxspec.model.multiplicative import MultiplicativeConstant


class BufferedContinuum(AdditiveComponent):
    """A configured continuum with a mutable, non-parameter numerical buffer."""

    def __init__(self, size=2):
        self.size = size
        self.weights = nnx.data(jnp.ones(size))
        self.norm = nnx.Param(1.0)

    def get_config(self):
        return {"size": self.size}

    def continuum(self, energy):
        return jnp.ones_like(energy) * self.weights.sum() * self.norm


def test_model_description_is_json_and_preserves_public_names():
    model = MultiplicativeConstant() * (Powerlaw() + Powerlaw())
    model.powerlaw_1.alpha.set_value(1.2)
    model.powerlaw_2.norm.set_value(0.02)
    description = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = SpectralModel.from_dict(description)

    assert restored.to_dict() == description
    assert restored.branches == model.branches
    assert "component" not in next(iter(restored._graph.nodes.values()))
    low, high = jnp.array([0.5, 1.5]), jnp.array([1.0, 2.0])
    np.testing.assert_allclose(restored.photon_flux(low, high), model.photon_flux(low, high))


@pytest.mark.parametrize("component", [Diskbb, Diskpbb])
def test_disk_description_preserves_nondefault_normalization_band(component):
    model = SpectralModel.from_component(component(flux_band=(0.8, 15.0)))
    restored = SpectralModel.from_dict(json.loads(json.dumps(model.to_dict())))
    name = f"{component.__name__.lower()}_1"
    assert getattr(restored, name)._flux_band == (0.8, 15.0)
    edges = jnp.geomspace(0.5, 20.0, 20)
    np.testing.assert_allclose(
        restored.photon_flux(edges[:-1], edges[1:]),
        model.photon_flux(edges[:-1], edges[1:]),
    )


def test_description_roundtrip_preserves_updated_parameter_and_buffer():
    model = SpectralModel.from_component(BufferedContinuum(size=3))
    model.bufferedcontinuum_1.weights = nnx.data(jnp.array([2.0, 4.0, 6.0]))
    model.bufferedcontinuum_1.norm.set_value(3.0)
    description = json.loads(json.dumps(model.to_dict()))
    restored = SpectralModel.from_dict(
        description, components={"BufferedContinuum": BufferedContinuum}
    )

    np.testing.assert_array_equal(restored.bufferedcontinuum_1.weights, [2.0, 4.0, 6.0])
    assert restored.bufferedcontinuum_1.size == 3
    np.testing.assert_allclose(restored.photon_flux(jnp.array([1.0]), jnp.array([3.0])), [72.0])
    assert restored.to_dict() == description


def test_unregistered_component_is_not_imported():
    model = SpectralModel.from_component(Powerlaw())
    description = model.to_dict()
    description["nodes"][0]["component"] = "os.system"
    with pytest.raises(ValueError, match="Unknown model component"):
        SpectralModel.from_dict(description)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda data: data.update(version=2), "version"),
        (lambda data: data.update(extra=True), "schema"),
        (lambda data: data["nodes"][0].update(name="_graph"), "component name"),
        (lambda data: data["nodes"][0].update(config={"unexpected": 1}), "configuration"),
        (lambda data: data["edges"].append(data["edges"][0].copy()), "exactly once"),
        (lambda data: data["edges"][0].update(position=1), "position 0"),
        (lambda data: data["state"].append(data["state"][0].copy()), "Duplicate numerical"),
        (lambda data: data["state"][0].update(dtype="object"), "numeric or boolean"),
        (lambda data: data["state"][0].update(value=float("nan")), "finite"),
    ],
)
def test_invalid_description_is_rejected_before_construction(change, message):
    class Counted(BufferedContinuum):
        calls = []

        def __init__(self, size=2):
            self.calls.append(size)
            super().__init__(size=size)

    model = SpectralModel.from_component(Counted())
    description = copy.deepcopy(model.to_dict())
    Counted.calls.clear()
    change(description)
    with pytest.raises(ValueError, match=message):
        SpectralModel.from_dict(description, components={"Counted": Counted})
    assert Counted.calls == []


@pytest.mark.parametrize("mutation", ["missing", "unknown", "shape", "variable"])
def test_state_must_match_reconstructed_template(mutation):
    description = SpectralModel.from_component(Powerlaw()).to_dict()
    if mutation == "missing":
        description["state"].pop()
    elif mutation == "unknown":
        description["state"][0]["path"] = ["powerlaw_1", "missing"]
    elif mutation == "shape":
        description["state"][0].update(shape=[2], value=[1.0, 2.0], storage="numpy")
    else:
        description["state"][0]["variable"] = None
    with pytest.raises(ValueError, match=r"(paths|shape|variable type)"):
        SpectralModel.from_dict(description)


def test_nonfinite_model_state_cannot_be_serialized():
    model = SpectralModel.from_component(Powerlaw())
    model.powerlaw_1.alpha.set_value(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        model.to_dict()


@pytest.mark.parametrize(
    "value",
    [np.float64(2.5), np.float32(2.5), np.int64(2), np.uint64(2), np.bool_(True)],
)
def test_numpy_scalar_parameter_dtype_survives_json_roundtrip(value):
    model = SpectralModel.from_component(Powerlaw())
    model.powerlaw_1.norm.set_value(value)
    description = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    record = next(item for item in description["state"] if item["path"][-1] == "norm")
    assert record["storage"] == "numpy_scalar"
    restored = SpectralModel.from_dict(description)
    actual = restored.powerlaw_1.norm.get_value()
    assert type(actual) is type(value)
    assert actual == value
    np.testing.assert_allclose(restored.photon_flux([1.0], [2.0]), model.photon_flux([1.0], [2.0]))


@pytest.mark.parametrize("value", [np.float64(np.inf), np.float32(np.nan)])
def test_nonfinite_numpy_scalars_are_rejected_in_configuration_and_state(value):
    with pytest.raises(ValueError, match="finite"):
        _json_config({"setting": value})
    model = SpectralModel.from_component(Powerlaw())
    model.powerlaw_1.norm.set_value(value)
    with pytest.raises(ValueError, match="finite"):
        model.to_dict()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(np.float64(2.5), 2.5), (np.int64(2), 2), (np.bool_(True), True)],
)
def test_numpy_configuration_scalars_have_literal_json_representation(value, expected):
    literal = _json_config({"setting": value})
    assert type(literal["setting"]) is type(expected)
    assert literal["setting"] == expected
    assert json.loads(json.dumps(literal, allow_nan=False)) == literal


def test_extended_precision_configuration_does_not_recurse_or_silently_round():
    value = np.longdouble("1.0000000000000000001")
    if isinstance(value.item(), np.generic):
        with pytest.raises(ValueError, match="representable"):
            _json_config(value)
    else:
        assert _json_config(value) == value.item()


def test_table_configuration_accepting_numpy_energy_band_can_roundtrip(tmp_path):
    from test_tabulated import one_param

    from jaxspec.model.tabulated import ATable

    path = one_param(path=tmp_path / "unit.fits", energ_lo=[1.0], energ_hi=[2.0])
    model = SpectralModel.from_component(ATable(path, energy_band=np.array([1.0, 2.0])))
    description = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert description["nodes"][0]["config"]["energy_band"] == [1.0, 2.0]
    restored = SpectralModel.from_dict(description)
    np.testing.assert_allclose(restored.photon_flux([1.0], [2.0]), [1.0])
    assert restored.to_dict() == description


def test_configuration_cannot_silently_drop_opaque_objects():
    class Opaque(BufferedContinuum):
        def get_config(self):
            return {"size": object()}

    with pytest.raises(ValueError, match="JSON literals"):
        SpectralModel.from_component(Opaque()).to_dict()


@pytest.mark.parametrize("constructor_style", ["keywords", "positional"])
def test_variadic_constructor_configuration_cannot_silently_revert_to_defaults(constructor_style):
    class KeywordConfigured(AdditiveComponent):
        def __init__(self, **kwargs):
            self.scale = kwargs.get("scale", 1.0)

        def continuum(self, energy):
            return self.scale * jnp.ones_like(energy)

    class PositionalConfigured(AdditiveComponent):
        def __init__(self, *args):
            self.scale = args[0] if args else 1.0

        def continuum(self, energy):
            return self.scale * jnp.ones_like(energy)

    component = (
        KeywordConfigured(scale=3.0)
        if constructor_style == "keywords"
        else PositionalConfigured(3.0)
    )
    model = SpectralModel.from_component(component)
    np.testing.assert_allclose(model.photon_flux([1.0], [2.0]), [3.0])
    with pytest.raises(TypeError, match="get_config"):
        model.to_dict()


def test_description_preserves_nested_operator_scope_and_order():
    class Reverse(OperatorComponent):
        def apply(self, flux, e_low, e_high):
            return flux[::-1]

    class Weight(OperatorComponent):
        def apply(self, flux, e_low, e_high):
            return flux * (e_low + e_high) / 2

    low, high = jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0])
    model = Reverse()(Weight()(Powerlaw())) + Weight()(Reverse()(Powerlaw()))
    registry = {"Reverse": Reverse, "Weight": Weight}
    description = model.to_dict()
    restored = SpectralModel.from_dict(description, components=registry)
    first = Reverse()(Weight()(Powerlaw())).photon_flux(low, high)
    second = Weight()(Reverse()(Powerlaw())).photon_flux(low, high)
    assert not np.allclose(first, second)
    np.testing.assert_allclose(restored.photon_flux(low, high), first + second)
    assert restored.to_dict() == description


def test_empty_numerical_buffers_roundtrip():
    model = SpectralModel.from_component(BufferedContinuum(size=0))
    restored = SpectralModel.from_dict(
        json.loads(json.dumps(model.to_dict())), components={"BufferedContinuum": BufferedContinuum}
    )
    assert restored.bufferedcontinuum_1.weights.shape == (0,)


def test_custom_registry_alias_can_restore_canonical_class_name():
    model = SpectralModel.from_component(BufferedContinuum(size=4))
    restored = SpectralModel.from_dict(model.to_dict(), components={"buffer": BufferedContinuum})
    assert restored.bufferedcontinuum_1.size == 4


def test_abstract_registered_component_is_rejected():
    class AbstractContinuum(AdditiveComponent):
        @abstractmethod
        def continuum(self, energy): ...

    description = SpectralModel.from_component(Powerlaw()).to_dict()
    description["nodes"][0]["component"] = "AbstractContinuum"
    with pytest.raises(ValueError, match="abstract"):
        SpectralModel.from_dict(description, components={"AbstractContinuum": AbstractContinuum})


def test_duplicate_canonical_class_names_are_rejected():
    class Powerlaw(AdditiveComponent):
        pass

    description = SpectralModel.from_component(BufferedContinuum()).to_dict()
    with pytest.raises(ValueError, match="Ambiguous"):
        SpectralModel.from_dict(description, components={"custom_powerlaw": Powerlaw})
