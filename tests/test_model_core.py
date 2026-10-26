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
from jaxspec.model.abc import AdditiveComponent, ModelComponent, SpectralModel
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


@pytest.fixture
def tiny_observation():
    low, high = np.arange(1.0, 4.0), np.arange(2.0, 5.0)
    instrument = Instrument.from_matrix(np.eye(3), np.ones(3), low, high, low, high)
    observation = Observation.from_matrix(
        [12, 16, 14], np.eye(3), np.arange(3), np.zeros(3), 10.0, background=[2, 2, 2]
    )
    return ObsConfiguration.from_instrument(instrument, observation)


def test_distinct_components_keep_distinct_parameters():
    model = Additiveconstant() + Additiveconstant()
    assert set(model.parameters) == {"additiveconstant_1.norm", "additiveconstant_2.norm"}
    assert nx.is_tree(model.graph)
    model.additiveconstant_1.norm.set_value(3.0)
    np.testing.assert_allclose(model.photon_flux([1.0], [2.0]), [4.0])


def _same_component_twice():
    component = Additiveconstant()
    return component + component


def _same_model_twice():
    model = Tbabs() * Powerlaw()
    return model + model


def _same_factor_on_two_branches():
    absorption = Tbabs()
    return absorption * Powerlaw() + absorption * Additiveconstant()


@pytest.mark.parametrize(
    "build", [_same_component_twice, _same_model_twice, _same_factor_on_two_branches]
)
def test_reusing_a_component_is_refused(build):
    with pytest.raises(ValueError, match="used twice"):
        build()


def test_composition_uses_replaced_current_component():
    original = SpectralModel.from_component(Additiveconstant())
    replacement = Additiveconstant()
    replacement.norm.set_value(7.0)
    original.additiveconstant_1 = replacement
    composed = original + Additiveconstant()
    np.testing.assert_allclose(composed.photon_flux([1.0], [2.0]), [8.0])
    assert composed.additiveconstant_1 is replacement


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


def test_factor_sums_retain_dimensionless_semantics():
    left, right = MultiplicativeConstant(), MultiplicativeConstant()
    left.norm.set_value(2.0)
    right.norm.set_value(3.0)
    emission = Additiveconstant()
    model = (left + right) * emission
    low, high = jnp.asarray([1.0, 2.0]), jnp.asarray([2.0, 4.0])
    expected = (left._factor(low, high) + right._factor(low, high)) * (high - low)
    np.testing.assert_allclose(model.photon_flux(low, high), expected)
    assert len(model.branches) == 1


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: Additiveconstant() + MultiplicativeConstant(), "Cannot add"),
        (lambda: MultiplicativeConstant() + Additiveconstant(), "Cannot add"),
        (lambda: Powerlaw() * Additiveconstant(), "Cannot multiply"),
    ],
)
def test_invalid_expression_kinds_are_rejected(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize("operand", [1.0, "powerlaw", None])
def test_arithmetic_rejects_non_model_operands(operand):
    with pytest.raises(TypeError, match="spectral"):
        Additiveconstant() + operand


def test_factors_alone_do_not_produce_photon_flux():
    model = MultiplicativeConstant() + MultiplicativeConstant()
    with pytest.raises(ValueError, match="no additive component"):
        model.photon_flux([1.0], [2.0])


@pytest.mark.parametrize("n_points", [0, 1, 2.5, True])
def test_flux_requires_meaningful_quadrature_resolution(n_points):
    model = SpectralModel.from_component(Additiveconstant())
    with pytest.raises(ValueError, match="n_points"):
        model.photon_flux([1.0], [2.0], n_points=n_points)


@pytest.mark.parametrize("n_points", [np.int64(3), jnp.int32(3)])
def test_flux_accepts_numpy_and_jax_integer_quadrature(n_points):
    model = SpectralModel.from_component(Additiveconstant())
    np.testing.assert_allclose(model.photon_flux([1.0], [2.0], n_points=n_points), [1.0])


def test_numpy_energy_grids_stay_concrete_under_jit():
    traced = []

    class GridProbe(AdditiveComponent):
        def __init__(self):
            self.norm = nnx.Param(1.0)

        def integrated_continuum(self, e_low, e_high):
            traced.append(isinstance(e_low, jax.core.Tracer))
            return self.norm * (e_high - e_low)

    model = SpectralModel.from_component(GridProbe())
    low, high = np.array([1.0, 2.0]), np.array([2.0, 4.0])

    @jax.jit
    def flux(norm):
        return model.photon_flux(low, high, params={"gridprobe_1.norm": norm})

    np.testing.assert_allclose(flux(3.0), [3.0, 6.0])
    # Components such as APEC size static work from concrete grids.
    assert traced == [False]


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


def test_working_grid_fits_and_predicts_branches(tiny_observation):
    # This short fit only checks trace/inference integration, not convergence.
    model = Additiveconstant() + Additiveconstant()
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
    assert list(prediction) == ["additiveconstant_1", "additiveconstant_2"]
    np.testing.assert_allclose(prediction["additiveconstant_1"], [7.0, 7.0, 7.0])
    np.testing.assert_allclose(prediction["additiveconstant_2"], [3.0, 3.0, 3.0])
