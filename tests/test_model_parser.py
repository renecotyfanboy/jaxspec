"""Model text must follow the same typed composition rules as object expressions."""

import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

from jaxspec.model._parser import component_registry, parse_model
from jaxspec.model.abc import AdditiveComponent, ModelComponent, OperatorComponent, SpectralModel
from jaxspec.model.additive import Additiveconstant, Diskbb, Powerlaw
from jaxspec.model.multiplicative import MultiplicativeConstant
from jaxspec.model.tabulated import ATable, ETable, MTable


@pytest.mark.parametrize(
    ("expression", "model"),
    [
        ("powerlaw", SpectralModel.from_component(Powerlaw())),
        ("Powerlaw()", SpectralModel.from_component(Powerlaw())),
        (
            "multiplicativeconstant * (Powerlaw + additiveconstant())",
            MultiplicativeConstant() * (Powerlaw() + Additiveconstant()),
        ),
        (
            " powerlaw + MultiplicativeConstant() * Powerlaw() ",
            Powerlaw() + MultiplicativeConstant() * Powerlaw(),
        ),
    ],
)
def test_model_strings_match_object_composition(expression, model):
    parsed = parse_model(expression)
    edges = jnp.array([0.5, 1.0, 2.0, 3.0])
    params = {"powerlaw_1.alpha": 2.2, "powerlaw_1.norm": 0.3}
    np.testing.assert_allclose(
        parsed.photon_flux(edges[:-1], edges[1:], params=params),
        model.photon_flux(edges[:-1], edges[1:], params=params),
    )
    expected = nnx.to_flat_state(nnx.state(model, nnx.Param))
    actual = nnx.to_flat_state(nnx.state(parsed, nnx.Param))
    assert [path for path, _ in actual] == [path for path, _ in expected]


def test_model_strings_retain_constructor_configuration():
    parsed = parse_model("diskbb(flux_band=(0.2, 30.0))")
    expected = SpectralModel.from_component(Diskbb(flux_band=(0.2, 30.0)))
    edges = jnp.array([0.5, 1.0, 2.0])
    assert parsed.diskbb_1._flux_band == (0.2, 30.0)
    np.testing.assert_allclose(
        parsed.photon_flux(edges[:-1], edges[1:]),
        expected.photon_flux(edges[:-1], edges[1:]),
    )


@pytest.mark.parametrize("component", [ATable, MTable, ETable])
def test_model_strings_read_explicit_local_table_configuration(tmp_path, component):
    from test_tabulated import one_param

    path = one_param(
        path=tmp_path / "a table's spectrum.fits",
        energ_lo=[1.0],
        energ_hi=[2.0],
        additive=component is ATable,
    )
    parsed = parse_model(f"{component.__name__}(path={path!r})")
    expected = SpectralModel.from_component(component(path))
    if component is not ATable:
        parsed = parsed * Additiveconstant()
        expected = expected * Additiveconstant()
    np.testing.assert_allclose(
        parsed.photon_flux(jnp.array([1.0]), jnp.array([2.0])),
        expected.photon_flux(jnp.array([1.0]), jnp.array([2.0])),
    )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "powerlaw - powerlaw",
        "powerlaw / powerlaw",
        "powerlaw ** 2",
        "2 * powerlaw",
        "powerlaw + multiplicativeconstant",
        "powerlaw * powerlaw",
        "unknown",
        "powerlaw(alpha=2)",
        "diskbb((0.1, 10.0))",
        "powerlaw.__class__",
        "powerlaw[0]",
        "__import__('os').system('touch sentinel')",
        "(lambda: powerlaw)()",
        "[powerlaw for _ in range(1)]",
        "powerlaw() if True else powerlaw()",
        "powerlaw(**{})",
        "powerlaw(*[])",
        "diskbb(flux_band=(0.1, 5 * 2))",
        "diskbb(flux_band=(0.1, float('inf')))",
        "diskbb(flux_band=(0.1, 1e999))",
        "diskbb(flux_band=(0.1, -1e999))",
        "diskbb(flux_band=(0.1, 10), flux_band=(0.1, 20))",
        "powerlaw(_private=True)",
        "powerlaw; powerlaw",
        "powerlaw = powerlaw",
        "powerlaw()()",
    ],
)
def test_invalid_model_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        parse_model(expression)


@pytest.mark.parametrize(
    "suffix",
    [
        " + unknown",
        " + powerlaw(alpha=2)",
        " + multiplicativeconstant",
        " * powerlaw",
        " + diskbb(flux_band=(0.1, 5 * 2))",
    ],
)
def test_invalid_complete_expression_never_runs_constructors(suffix):
    calls = []

    class RecordingEmission(AdditiveComponent):
        def __init__(self):
            calls.append("constructed")

    with pytest.raises(ValueError):
        parse_model("recording" + suffix, components={"recording": RecordingEmission})
    assert calls == []


def test_custom_components_require_explicit_registration():
    class LocalEmission(AdditiveComponent):
        def __init__(self, offset=0.0, options=None):
            self.norm = nnx.Param(offset)
            self.options = options

        def continuum(self, energy):
            return self.norm * jnp.ones_like(energy)

    with pytest.raises(ValueError, match="Unknown"):
        parse_model("localemission")
    model = parse_model(
        "local(offset=-2.5, options={'switch': True, 'limits': [1, None]})",
        components={"local": LocalEmission},
    )
    assert model.localemission_1.options == {"switch": True, "limits": [1, None]}
    np.testing.assert_allclose(model.photon_flux(jnp.array([1.0]), jnp.array([3.0])), [-5.0])


@pytest.mark.parametrize(
    "registration",
    [
        {"os.system": Powerlaw},
        {"_hidden": Powerlaw},
        {"class": Powerlaw},
        {1: Powerlaw},
        {"powerlaw": Additiveconstant},
        {"custom": object},
        {"custom": Powerlaw()},
        {"alias": Powerlaw, "ALIAS": Additiveconstant},
    ],
)
def test_registry_rejects_invalid_or_ambiguous_registrations(registration):
    with pytest.raises((ValueError, TypeError)):
        component_registry(registration)


@pytest.mark.parametrize(
    "expression",
    [
        " " * 16_385,
        "+".join(["powerlaw"] * 65),
        "diskbb(flux_band=[" + ",".join(["1"] * 1_025) + "])",
    ],
)
def test_parser_bounds_expression_size_and_nesting(expression):
    with pytest.raises(ValueError, match="limit"):
        parse_model(expression)


def test_parser_requires_text():
    with pytest.raises(TypeError, match="string"):
        parse_model(None)


def test_scoped_operator_strings_preserve_nonlinear_scope_and_configuration():
    class TestPower(OperatorComponent):
        def __init__(self, power=2):
            self.power = power

        def apply(self, flux, e_low, e_high):
            return flux**self.power

    registry = {"testpower": TestPower}
    grouped = parse_model(
        "testpower(additiveconstant + additiveconstant, power=3)", components=registry
    )
    separate = parse_model(
        "testpower(additiveconstant, power=3) + testpower(additiveconstant, power=3)",
        components=registry,
    )
    expected = TestPower(power=3)(Additiveconstant() + Additiveconstant())
    lower, upper = jnp.array([1.0, 2.0]), jnp.array([2.0, 4.0])
    source = (Additiveconstant() + Additiveconstant()).photon_flux(lower, upper)
    actual = grouped.photon_flux(lower, upper)
    np.testing.assert_allclose(actual, source**3)
    np.testing.assert_allclose(actual, expected.photon_flux(lower, upper))
    assert not np.allclose(actual, separate.photon_flux(lower, upper))
    assert grouped.testpower_1.power == 3
    assert len(grouped.photon_flux(lower, upper, split_branches=True)) == 1


@pytest.mark.parametrize(
    "expression",
    [
        "testoperator * powerlaw",
        "testoperator() * powerlaw",
        "testoperator(powerlaw, powerlaw)",
        "testoperator(multiplicativeconstant)",
        "testoperator(powerlaw, missing=True)",
    ],
)
def test_operators_require_valid_explicit_scope_before_construction(expression):
    calls = []

    class TestOperator(OperatorComponent):
        def __init__(self):
            calls.append("constructed")

        def apply(self, flux, e_low, e_high):
            return flux

    with pytest.raises(ValueError):
        parse_model(expression, components={"testoperator": TestOperator})
    assert calls == []


def test_factor_operator_strings_respect_declared_measure():
    class TestFactorOperator(OperatorComponent):
        input_type = "multiplicative"
        output_type = "multiplicative"

        def apply(self, flux, e_low, e_high):
            return 1 + flux

    registry = {"testfactor": TestFactorOperator}
    parsed = parse_model("testfactor(multiplicativeconstant) * powerlaw", components=registry)
    expected = TestFactorOperator()(MultiplicativeConstant()) * Powerlaw()
    lower, upper = jnp.array([1.0, 2.0]), jnp.array([2.0, 4.0])
    np.testing.assert_allclose(parsed.photon_flux(lower, upper), expected.photon_flux(lower, upper))
    with pytest.raises(ValueError, match="multiplicative subexpression"):
        parse_model("testfactor(powerlaw)", components=registry)
    with pytest.raises(ValueError, match="constructor arguments"):
        parse_model("testfactor(multiplicativeconstant, missing=True)", components=registry)


def test_operator_type_requires_operator_base_class():
    class PretendOperator(ModelComponent):
        type = "operator"
        input_type = "additive"
        output_type = "additive"

    with pytest.raises(ValueError, match="OperatorComponent"):
        parse_model("pretend(powerlaw)", components={"pretend": PretendOperator})
