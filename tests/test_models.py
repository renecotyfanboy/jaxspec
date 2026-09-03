import inspect
import re

import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

from jaxspec.model.additive import Additiveconstant, Blackbodyrad, Powerlaw
from jaxspec.model.list import additive_components, multiplicative_components
from jaxspec.model.multiplicative import MultiplicativeConstant, Tbabs


def _no_args_constructible(cls) -> bool:
    """Whether the sweep below can instantiate ``cls()``.

    Components driven by an external file (e.g. the OGIP table models in
    ``jaxspec.model.tabulated``) require constructor arguments; they land in the
    registry whenever their module was imported before ``jaxspec.model.list`` and
    would break the no-args sweeps, so they are covered by their own test module
    instead.
    """
    return all(
        parameter.default is not parameter.empty
        or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
    )


additive_components = {k: v for k, v in additive_components.items() if _no_args_constructible(v)}
multiplicative_components = {
    k: v for k, v in multiplicative_components.items() if _no_args_constructible(v)
}


@pytest.mark.parametrize("test_input", list(additive_components.keys()))
def test_additive_components(test_input):
    energy = jnp.geomspace(0.5, 10, 1000)
    e_low = energy[:-1]
    e_high = energy[1:]

    spectral_model = (
        MultiplicativeConstant()
        * MultiplicativeConstant()
        * (Additiveconstant() + additive_components[test_input]())
    )
    out = spectral_model.flux_func(e_low, e_high)
    assert out.shape == e_low.shape


@pytest.mark.parametrize("name", sorted({**additive_components, **multiplicative_components}))
def test_every_component_exposes_fittable_parameters(name):
    """Every component must expose at least one ``nnx.Param`` leaf.

    ``jnp.asarray(nnx.Param(x))`` silently unwraps to a bare array via
    ``__jax_array__``, leaving a component with zero fittable leaves: it cannot be
    fitted and ``params=`` overrides are ignored without error. ``Lorentz`` shipped that
    way. Shape assertions cannot see it, so assert on the parameter state directly.
    """
    component = {**additive_components, **multiplicative_components}[name]()
    _, params, *_ = nnx.split(component, nnx.Param, ...)

    assert len(params.flat_state()) > 0, (
        f"{name} exposes no nnx.Param leaves — check for jnp.asarray(nnx.Param(...)), "
        f"which unwraps the Param instead of promoting the value inside it."
    )


@pytest.mark.parametrize("test_input", list(multiplicative_components.keys()))
def test_multiplicative_components(test_input):
    energy = jnp.geomspace(0.5, 10, 1000)
    e_low = energy[:-1]
    e_high = energy[1:]

    spectral_model = (
        MultiplicativeConstant()
        * multiplicative_components[test_input]()
        * (Additiveconstant() + Additiveconstant())
    )
    out = spectral_model.flux_func(e_low, e_high)
    assert out.shape == e_low.shape


@pytest.mark.slow
def test_mermaid_representation():
    spectral_model = Tbabs() * (Powerlaw() + Blackbodyrad())
    mermaid = spectral_model.to_mermaid()
    node_pattern = re.compile(r'^\s+([0-9a-f-]+|out)(?:\("([^"]+)"\)|\{(.+)\})$')
    edge_pattern = re.compile(r"^\s+([0-9a-f-]+|out) --> ([0-9a-f-]+|out)$")

    nodes = {}
    edges = set()

    for line in mermaid.splitlines():
        if line == "graph LR":
            continue

        if node_match := node_pattern.match(line):
            node_id, label, operator = node_match.groups()
            nodes[node_id] = label or operator
            continue

        if edge_match := edge_pattern.match(line):
            edges.add(edge_match.groups())
            continue

        pytest.fail(f"Unexpected Mermaid line: {line}")

    assert set(nodes.values()) == {
        "Tbabs (1)",
        "Powerlaw (1)",
        "Blackbodyrad (1)",
        "**+**",
        "**x**",
        "Output",
    }

    tbabs_id = next(node_id for node_id, label in nodes.items() if label == "Tbabs (1)")
    powerlaw_id = next(node_id for node_id, label in nodes.items() if label == "Powerlaw (1)")
    blackbody_id = next(node_id for node_id, label in nodes.items() if label == "Blackbodyrad (1)")
    add_id = next(node_id for node_id, label in nodes.items() if label == "**+**")
    mul_id = next(node_id for node_id, label in nodes.items() if label == "**x**")
    out_id = next(node_id for node_id, label in nodes.items() if label == "Output")

    assert edges == {
        (tbabs_id, mul_id),
        (powerlaw_id, add_id),
        (blackbody_id, add_id),
        (add_id, mul_id),
        (mul_id, out_id),
    }


def test_tbpcf_at_full_covering_fraction_matches_tbabs():
    """At `f=1` a partial-covering absorber is a plain absorber.

    The two read the same Wilms cross-section table, but `Tbpcf` used to interpolate it
    against *linear* energy while `Tbabs` uses log-log, so identical physics gave slightly
    different transmission.
    """
    from jaxspec.model.multiplicative import Tbpcf

    energy = jnp.geomspace(0.3, 12.0, 500)
    tbabs, tbpcf = Tbabs(), Tbpcf()
    tbpcf.f.value = jnp.asarray(1.0)

    np.testing.assert_allclose(
        np.asarray(tbpcf.factor(energy)), np.asarray(tbabs.factor(energy)), rtol=1e-12
    )


def test_cross_section_tables_are_shared_between_components():
    """The cached loader must hand out the same array object, not a fresh copy.

    `nnx.clone` then shares the table across per-observation replicas instead of
    deep-copying it, which is the whole point of the plain-ndarray table convention.
    """
    from jaxspec.model.multiplicative import Tbpcf

    assert Tbabs()._sigma is Tbpcf()._sigma
    assert Tbabs()._energy is Tbabs()._energy
