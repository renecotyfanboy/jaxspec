"""Expression structure, public names and explicit NNX numerical-state contracts."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import numpyro.distributions as dist
import pytest

from flax import nnx

from jaxspec.data import Instrument, ObsConfiguration, Observation
from jaxspec.fit import BayesianModel, MCMCFitter
from jaxspec.fit._forward_model import ForwardModel
from jaxspec.fit._prior_resolution import _enumerate_leaves
from jaxspec.model.abc import AdditiveComponent, ModelComponent, OperatorComponent, SpectralModel
from jaxspec.model.additive import Additiveconstant, Powerlaw
from jaxspec.model.background import SpectralModelBackground
from jaxspec.model.multiplicative import MultiplicativeConstant, Phabs, Tbabs


class ArrayEmission(AdditiveComponent):
    """A tiny array-backed spectrum with explicitly tracked, non-fitted state."""

    def __init__(self, values):
        self.norm = nnx.Param(jnp.asarray(1.0))
        self.values = nnx.Variable(jnp.asarray(values))

    def integrated_continuum(self, e_low, e_high):
        return self.norm[...] * self.values[...] * (e_high - e_low)


class Square(OperatorComponent):
    """Toy nonlinear operator for expression-scope tests, not a physical model."""

    def apply(self, flux, e_low, e_high):
        return flux**2


class Offset(OperatorComponent):
    """Toy operator that does not commute with squaring."""

    def apply(self, flux, e_low, e_high):
        return flux + 1.0


class DoubleFactor(OperatorComponent):
    input_type = "multiplicative"
    output_type = "multiplicative"

    def apply(self, flux, e_low, e_high):
        return 2.0 * flux


@pytest.fixture
def tiny_observation():
    low, high = np.arange(1.0, 4.0), np.arange(2.0, 5.0)
    instrument = Instrument.from_matrix(np.eye(3), np.ones(3), low, high, low, high)
    observation = Observation.from_matrix(
        [12, 16, 14], np.eye(3), np.arange(3), np.zeros(3), 10.0, background=[2, 2, 2]
    )
    return ObsConfiguration.from_instrument(instrument, observation)


@pytest.mark.parametrize("reuse_model", [False, True])
def test_repeated_expression_occurrences_keep_distinct_parameters(reuse_model):
    component = Additiveconstant()
    operand = component + component if reuse_model else component
    model = operand + operand
    names = ["additiveconstant_1", "additiveconstant_2"]
    if reuse_model:
        names += ["additiveconstant_3", "additiveconstant_4"]
    assert set(model.parameters) == {f"{name}.norm" for name in names}
    assert len({id(getattr(model, name).norm) for name in names}) == len(names)
    assert len(model.root_nodes) == len(names)
    assert nx.is_tree(model.graph)

    model.additiveconstant_1.norm.set_value(3.0)
    np.testing.assert_allclose(model.photon_flux([1.0], [2.0]), [len(names) + 2.0])
    assert float(component.norm.get_value()) == 1.0
    for name in names[1:]:
        assert float(getattr(model, name).norm.get_value()) == 1.0
    if reuse_model:
        assert set(operand.parameters) == {"additiveconstant_1.norm", "additiveconstant_2.norm"}


def test_composition_uses_replaced_current_component():
    original = SpectralModel.from_component(Additiveconstant())
    replacement = Additiveconstant()
    replacement.norm.set_value(7.0)
    original.additiveconstant_1 = replacement
    composed = original + Additiveconstant()
    np.testing.assert_allclose(composed.photon_flux([1.0], [2.0]), [8.0])
    assert float(composed.parameters["additiveconstant_1.norm"]) == 7.0
    replacement.norm.set_value(11.0)
    np.testing.assert_allclose(composed.photon_flux([1.0], [2.0]), [8.0])


def test_graph_inspection_is_detached_and_contains_no_model_state():
    model = SpectralModel.from_component(ArrayEmission([2.0, 3.0])) + Additiveconstant()
    graph = model.graph
    for _, metadata in graph.nodes(data=True):
        assert "component" not in metadata
        assert not any(
            isinstance(value, ModelComponent | nnx.Variable | np.ndarray | jax.Array)
            for value in metadata.values()
        )
    component_id = next(node for node, data in graph.nodes(data=True) if "name" in data)
    graph.nodes[component_id]["name"] = "tampered_1"
    graph.remove_node("out")
    assert "out" in model.graph
    assert "tampered_1" not in {data.get("name") for _, data in model.graph.nodes(data=True)}
    np.testing.assert_allclose(model.photon_flux([1.0, 2.0], [2.0, 3.0]), [3.0, 4.0])


def test_parameters_expose_current_public_paths_without_buffers():
    model = SpectralModel.from_component(ArrayEmission([2.0, 3.0])) + Powerlaw()
    model.arrayemission_1.norm[...] = 4.0
    parameters = model.parameters
    assert set(parameters) == {"arrayemission_1.norm", "powerlaw_1.alpha", "powerlaw_1.norm"}
    assert float(parameters["arrayemission_1.norm"]) == 4.0
    parameters["arrayemission_1.norm"] = 100.0
    assert float(model.arrayemission_1.norm[...]) == 4.0


def test_jit_with_explicit_split_state_observes_array_updates():
    model = SpectralModel.from_component(ArrayEmission([2.0, 3.0]))
    graphdef, params, buffers = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def evaluate(parameter_state, array_state):
        restored = nnx.merge(graphdef, parameter_state, array_state)
        return restored.photon_flux(jnp.asarray([1.0, 2.0]), jnp.asarray([2.0, 4.0]))

    np.testing.assert_allclose(evaluate(params, buffers), [2.0, 6.0])
    model.arrayemission_1.values[...] = jnp.asarray([5.0, 7.0])
    new_graphdef, new_params, new_buffers = nnx.split(model, nnx.Param, ...)
    assert new_graphdef == graphdef
    np.testing.assert_allclose(evaluate(new_params, new_buffers), [5.0, 14.0])


def test_nnx_jit_observes_current_model_owned_arrays():
    model = SpectralModel.from_component(ArrayEmission([2.0, 3.0]))

    @nnx.jit
    def evaluate(spectrum):
        return spectrum.photon_flux(jnp.asarray([1.0, 2.0]), jnp.asarray([2.0, 4.0]))

    np.testing.assert_allclose(evaluate(model), [2.0, 6.0])
    model.arrayemission_1.values[...] = jnp.asarray([5.0, 7.0])
    np.testing.assert_allclose(evaluate(model), [5.0, 14.0])


def test_ordinary_branch_names_remain_stable():
    model = Tbabs() * (Powerlaw() + Phabs() * Powerlaw())
    assert model.branches == [
        "tbabs_1*powerlaw_1",
        "phabs_1*tbabs_1*powerlaw_2",
    ]
    split = model.photon_flux([1.0, 2.0], [2.0, 3.0], split_branches=True)
    assert list(split) == model.branches
    np.testing.assert_allclose(sum(split.values()), model.photon_flux([1.0, 2.0], [2.0, 3.0]))


def test_nonlinear_operator_receives_whole_sum_and_one_branch():
    first, second = Additiveconstant(), Additiveconstant()
    second.norm.set_value(2.0)
    model = Square()(first + second)
    expected = np.asarray([9.0, 36.0])
    split = model.photon_flux([1.0, 2.0], [2.0, 4.0], split_branches=True)
    assert len(split) == 1
    assert list(split) == model.branches
    np.testing.assert_allclose(next(iter(split.values())), expected)
    np.testing.assert_allclose(model.photon_flux([1.0, 2.0], [2.0, 4.0]), expected)
    separate = Square()(first) + Square()(second)
    np.testing.assert_allclose(separate.photon_flux([1.0, 2.0], [2.0, 4.0]), [5.0, 20.0])


def test_noncommuting_operator_order_is_preserved():
    square_after_offset = Square()(Offset()(Additiveconstant()))
    offset_after_square = Offset()(Square()(Additiveconstant()))
    low, high = jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 6.0])
    np.testing.assert_allclose(square_after_offset.photon_flux(low, high), [9.0, 25.0])
    np.testing.assert_allclose(offset_after_square.photon_flux(low, high), [5.0, 17.0])
    assert square_after_offset.branches != offset_after_square.branches


def test_factor_sums_and_factor_operators_retain_dimensionless_semantics():
    left, right = MultiplicativeConstant(), MultiplicativeConstant()
    left.norm.set_value(2.0)
    right.norm.set_value(3.0)
    emission = Additiveconstant()
    model = DoubleFactor()(left + right) * emission
    low, high = jnp.asarray([1.0, 2.0]), jnp.asarray([2.0, 4.0])
    expected = 2 * (left._factor(low, high) + right._factor(low, high)) * (high - low)
    np.testing.assert_allclose(model.photon_flux(low, high), expected)
    assert len(model.branches) == 1


def test_identity_factor_operator_preserves_energy_integral():
    class IdentityFactor(OperatorComponent):
        input_type = "multiplicative"
        output_type = "multiplicative"

        def apply(self, flux, e_low, e_high):
            return flux

    class LinearContinuum(AdditiveComponent):
        def continuum(self, energy):
            return energy

    low, high = jnp.array([1.0, 3.0]), jnp.array([3.0, 10.0])
    plain = MultiplicativeConstant() * LinearContinuum()
    wrapped = IdentityFactor()(MultiplicativeConstant()) * LinearContinuum()
    np.testing.assert_allclose(
        wrapped.photon_flux(low, high, n_points=1000),
        plain.photon_flux(low, high, n_points=1000),
    )
    np.testing.assert_allclose(
        wrapped.energy_flux(low, high, n_points=1000),
        plain.energy_flux(low, high, n_points=1000),
        rtol=1e-12,
    )
    # The continuum is E photons/(keV cm² s), so its energy moment is ∫E² dE.
    np.testing.assert_allclose(
        wrapped.energy_flux(low, high, n_points=1000), (high**3 - low**3) / 3, rtol=1e-5
    )


def test_energy_midpoint_approximation_is_limited_to_transported_branch():
    class LinearContinuum(AdditiveComponent):
        def continuum(self, energy):
            return energy

    low, high = jnp.array([1.0, 3.0]), jnp.array([3.0, 10.0])
    model = Square()(LinearContinuum()) + LinearContinuum()
    photon_integrals = (high**2 - low**2) / 2
    transported_energy = photon_integrals**2 * (high + low) / 2
    untouched_energy = LinearContinuum().energy_flux(low, high, n_points=1000)

    split = model.flux_func(low, high, energy_flux=True, n_points=1000, return_branches=True)
    operator_branch = next(value for key, value in split.items() if key.startswith("square_1("))
    np.testing.assert_allclose(operator_branch, transported_energy)
    np.testing.assert_allclose(split["linearcontinuum_2"], untouched_energy, rtol=1e-12)
    np.testing.assert_allclose(
        model.energy_flux(low, high, n_points=1000), transported_energy + untouched_energy
    )
    np.testing.assert_allclose(
        model.photon_flux(low, high, n_points=1000), photon_integrals**2 + photon_integrals
    )


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: Additiveconstant() + MultiplicativeConstant(), "Cannot add"),
        (lambda: MultiplicativeConstant() + Additiveconstant(), "Cannot add"),
        (lambda: Powerlaw() * Additiveconstant(), "Cannot multiply"),
        (lambda: Square() * Additiveconstant(), "explicit expression"),
        (lambda: Additiveconstant() * Square(), "explicit expression"),
        (lambda: SpectralModel.from_component(Square()), "explicit expression"),
        (lambda: Square()(MultiplicativeConstant()), "requires one additive"),
        (lambda: DoubleFactor()(Additiveconstant()), "requires one multiplicative"),
    ],
)
def test_invalid_expression_kinds_are_rejected(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_operator_kinds_must_be_declared_by_the_class():
    class ConfiguredKind(OperatorComponent):
        def __init__(self):
            self.input_type = "multiplicative"
            self.output_type = "multiplicative"

        def apply(self, flux, e_low, e_high):
            return flux

    # Strings and saved models validate kinds before constructing components;
    # object construction must therefore reject instance-only changes too.
    with pytest.raises(ValueError, match="class-declared"):
        ConfiguredKind()(MultiplicativeConstant())


@pytest.mark.parametrize("operand", [1.0, "powerlaw", None])
def test_arithmetic_rejects_non_model_operands(operand):
    with pytest.raises(TypeError, match="spectral"):
        Additiveconstant() + operand
    with pytest.raises(TypeError, match="spectral"):
        Square()(operand)


def test_factors_alone_do_not_produce_photon_flux():
    model = MultiplicativeConstant() + MultiplicativeConstant()
    with pytest.raises(ValueError, match="no additive component"):
        model.photon_flux([1.0], [2.0])


@pytest.mark.parametrize("n_points", [0, 1, 2.5, True])
def test_flux_requires_meaningful_quadrature_resolution(n_points):
    model = SpectralModel.from_component(Additiveconstant())
    with pytest.raises(ValueError, match="n_points"):
        model.photon_flux([1.0], [2.0], n_points=n_points)


def test_bare_background_keeps_public_parameter_names(tiny_observation):
    forward = ForwardModel(
        Additiveconstant(),
        tiny_observation,
        background_model=SpectralModelBackground(Powerlaw()),
    )
    assert set(_enumerate_leaves(forward)) == {
        "spectrum.additiveconstant_1.norm",
        "background.alpha",
        "background.norm",
    }
    inputs = {
        "spectrum.data.additiveconstant_1.norm": 2.0,
        "background.data.spectral_model.alpha": 0.0,
        "background.data.spectral_model.norm": 0.5,
    }
    prediction = forward.evaluate(inputs)["data"]
    np.testing.assert_allclose(prediction["source"], [20.0, 20.0, 20.0])
    np.testing.assert_allclose(prediction["background"], [5.0, 5.0, 5.0])
    bayesian = BayesianModel(
        Additiveconstant(),
        {"spectrum.additiveconstant_1.norm": 2.0, "background.alpha": 0.0, "background.norm": 0.5},
        tiny_observation,
        background_model=SpectralModelBackground(Powerlaw()),
    )
    bayesian.prior_samples(num_samples=1)


def test_scoped_operator_fits_with_working_grid_and_predicts(tiny_observation):
    # This short fit only checks trace/inference integration, not convergence.
    model = Square()(Additiveconstant() + Additiveconstant())
    prior = {
        "spectrum.additiveconstant_1.norm": dist.Uniform(0.1, 1.0),
        "spectrum.additiveconstant_2.norm": 0.3,
    }
    fitter = MCMCFitter(model, prior, tiny_observation, energy_grid=np.arange(1.0, 5.0))
    result = fitter.fit(
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        mcmc_kwargs={"progress_bar": False},
    )
    samples = result.input_parameters["spectrum.additiveconstant_1.norm"]
    assert samples.shape[:2] == (1, 5)
    assert np.isfinite(samples).all()
    forward = fitter.forward_model
    prediction = forward.evaluate(
        {
            "spectrum.data.additiveconstant_1.norm": 0.7,
            "spectrum.data.additiveconstant_2.norm": 0.3,
        },
        split_branches=True,
    )["data"]["source"]
    assert len(prediction) == 1
    np.testing.assert_allclose(next(iter(prediction.values())), [10.0, 10.0, 10.0])
