import re

import jax.numpy as jnp
import pytest

from jaxspec.model.additive import Additiveconstant, Blackbodyrad, Powerlaw
from jaxspec.model.list import additive_components, multiplicative_components
from jaxspec.model.multiplicative import MultiplicativeConstant, Tbabs


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
    out = spectral_model.turbo_flux(e_low, e_high)
    assert out.shape == e_low.shape


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
    out = spectral_model.turbo_flux(e_low, e_high)
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
