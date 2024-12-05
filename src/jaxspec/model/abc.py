from __future__ import annotations

import operator

from abc import ABC
from functools import partial
from uuid import uuid4

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import networkx as nx

from simpleeval import simple_eval

from jaxspec.util.typing import PriorDictType

from ._graph_util import compose, export_to_mermaid


def set_parameters(params: PriorDictType, state: nnx.State) -> nnx.State:
    """
    Set the parameters of a spectral model using `nnx'` routines.

    Parameters:
        params: Dictionary of parameters to set.
        model: Spectral model.

    Returns:
        A spectral model with the newly set parameters.
    """

    state_dict = state.to_pure_dict()  # haiku-like 2 level dictionary

    for key, value in params.items():
        # Split the key to extract the module name and parameter name
        module_name, param_name = key.rsplit("_", 1)
        state_dict["modules"][module_name][param_name] = value

    state.replace_by_pure_dict(state_dict)

    return state


class Composable(ABC):
    """
    Defines the set of operations between model components and spectral models
    """

    def sanitize_inputs(self, other):
        if isinstance(self, ModelComponent):
            model_1 = SpectralModel.from_component(self)
        else:
            model_1 = self

        if isinstance(other, ModelComponent):
            model_2 = SpectralModel.from_component(other)
        else:
            model_2 = other

        return model_1, model_2

    def __add__(self, other):
        model_1, model_2 = self.sanitize_inputs(other)
        return model_1.compose(model_2, operation="add", operation_func=operator.add)

    def __mul__(self, other):
        model_1, model_2 = self.sanitize_inputs(other)
        return model_1.compose(model_2, operation="mul", operation_func=operator.mul)


class SpectralModel(nnx.Module, Composable):
    # graph: nx.DiGraph = eqx.field(static=True)
    # modules: dict

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.modules = {}

        for node, data in self.graph.nodes(data=True):
            if "component" in data["type"]:
                self.modules[data["name"]] = data["component"]  # (**data['kwargs'])

    @classmethod
    def from_string(cls, string: str) -> SpectralModel:
        """
        This constructor enable to build a model from a string. The string should be a valid python expression, with
        the following constraints :

        * The model components should be defined in the jaxspec.model.list module
        * The model components should be separated by a * or a + (no convolution yet)
        * The model components should be written with their parameters in parentheses

        Parameters:
            string : The string to parse

        Examples:
            An absorbed model with a powerlaw and a blackbody:

            >>> model = SpectralModel.from_string("Tbabs()*(Powerlaw() + Blackbody())")
        """

        from .list import model_components

        return simple_eval(string, functions=model_components)

    def to_string(self) -> str:
        """
        This method return the string representation of the model.

        Examples:
            Build a model from a string and convert it back to a string:

            >>> model = SpectralModel.from_string("Tbabs()*(Powerlaw() + Blackbody())")
            >>> model.to_string()
            "Tbabs()*(Powerlaw() + Blackbody())"
        """
        return str(self)

    """
    def __str__(self) -> str:
        def build_expression(node_id):
            node = self.graph.nodes[node_id]
            if node["type"] == "component":
                string = node["component"].__name__

                if node["kwargs"]:
                    kwargs = ", ".join([f"{k}={v}" for k, v in node["kwargs"].items()])
                    string += f"({kwargs})"
                else:
                    string += "()"
                return string

            elif node["type"] == "operation":
                predecessors = list(self.graph.predecessors(node_id))
                operands = [build_expression(pred) for pred in predecessors]
                operation = node["operation_label"]
                return f"({f' {operation} '.join(operands)})"
            elif node["type"] == "out":
                predecessors = list(self.graph.predecessors(node_id))
                return build_expression(predecessors[0])

        return "This must be changed"  # build_expression("out")[1:-1]
    """

    def compose(self, other, operation=None, operation_func=None):
        """
        This function operate a composition between the operation graph of two models
        1) It fuses the two graphs using which joins at the 'out' nodes and change components name to unique identifiers
        2) It relabels the 'out' node with a unique identifier and labels it with the operation
        3) It links the operation to a new 'out' node
        """

        composed_graph = compose(
            self.graph, other.graph, operation=operation, operation_func=operation_func
        )

        return SpectralModel(composed_graph)

    @classmethod
    def from_component(cls, component):
        node_id = str(uuid4())
        graph = nx.DiGraph()

        node_properties = {
            "type": f"{component.type}_component",
            "name": f"{component.__class__.__name__}_1".lower(),
            "component": component,
            "depth": 0,
        }

        graph.add_node(node_id, **node_properties)
        graph.add_node("out", type="out", depth=1)
        graph.add_edge(node_id, "out")

        return cls(graph)

    def _find_multiplicative_components(self, node_id):
        """
        Recursively finds all the multiplicative components connected to the node with the given ID.
        """
        node = self.graph.nodes[node_id]
        multiplicative_nodes = []

        if node.get("type") == "mul_operation":
            # Recursively find all the multiplicative components using the predecessors
            predecessors = self.graph.pred[node_id]
            for node_id in predecessors:
                if "multiplicative_component" == self.graph.nodes[node_id].get("type"):
                    multiplicative_nodes.append(node_id)
                elif "mul_operation" == self.graph.nodes[node_id].get("type"):
                    multiplicative_nodes.extend(self._find_multiplicative_components(node_id))

        return multiplicative_nodes

    @property
    def root_nodes(self) -> list[str]:
        return [
            node_id
            for node_id, in_degree in self.graph.in_degree(self.graph.nodes)
            if in_degree == 0 and ("additive" in self.graph.nodes[node_id].get("type"))
        ]

    @property
    def branches(self) -> list[str]:
        branches = []

        for root_node_id in self.root_nodes:
            root_node_name = self.graph.nodes[root_node_id].get("name")
            path = nx.shortest_path(self.graph, source=root_node_id, target="out")
            multiplicative_components = []

            # Search all multiplicative components connected to this node
            # and apply them at mean energy
            for node_id in path[::-1]:
                multiplicative_components.extend(
                    [node_id for node_id in self._find_multiplicative_components(node_id)]
                )

            branch = ""

            for multiplicative_node_id in multiplicative_components:
                multiplicative_node_name = self.graph.nodes[multiplicative_node_id].get("name")
                branch += f"{multiplicative_node_name}*"

            branch += f"{root_node_name}"
            branches.append(branch)

        return branches

    def turbo_flux(self, e_low, e_high, energy_flux=False, n_points=2, return_branches=False):
        continuum = {}

        ## Evaluate the expected contribution for each component
        for node_id in nx.dag.topological_sort(self.graph):
            node = self.graph.nodes[node_id]

            if node["type"] == "additive_component":
                node_name = node["name"]
                runtime_modules = self.modules[node_name]

                if not energy_flux:
                    continuum[node_name] = runtime_modules._photon_flux(
                        e_low, e_high, n_points=n_points
                    )

                else:
                    continuum[node_name] = runtime_modules._energy_flux(
                        e_low, e_high, n_points=n_points
                    )

            elif node["type"] == "multiplicative_component":
                node_name = node["name"]
                runtime_modules = self.modules[node_name]
                continuum[node_name] = runtime_modules.factor((e_low + e_high) / 2)

            else:
                pass

        ## Propagate the absorption for each branch
        root_nodes = self.root_nodes

        branches = {}

        for root_node_id in root_nodes:
            root_node_name = self.graph.nodes[root_node_id].get("name")
            root_continuum = continuum[root_node_name]

            path = nx.shortest_path(self.graph, source=root_node_id, target="out")
            multiplicative_components = []

            # Search all multiplicative components connected to this node
            # and apply them at mean energy
            for node_id in path[::-1]:
                multiplicative_components.extend(
                    [node_id for node_id in self._find_multiplicative_components(node_id)]
                )

            branch = ""

            for multiplicative_node_id in multiplicative_components:
                multiplicative_node_name = self.graph.nodes[multiplicative_node_id].get("name")
                root_continuum *= continuum[multiplicative_node_name]
                branch += f"{multiplicative_node_name}*"

            branch += f"{root_node_name}"
            branches[branch] = root_continuum

        if return_branches:
            return branches

        return sum(branches.values())

    def to_mermaid(self, file: str | None = None):
        """
        This method returns the mermaid representation of the model.

        Parameters:
            file : The file to write the mermaid representation to.

        Returns:
            A string containing the mermaid representation of the model.
        """
        return export_to_mermaid(self.graph, file)

    @partial(jax.jit, static_argnums=0, static_argnames=("n_points", "split_branches"))
    def photon_flux(self, params, e_low, e_high, n_points=2, split_branches=False):
        r"""
        Compute the expected counts between $E_\min$ and $E_\max$ by integrating the model.

        $$ \Phi_{\text{photon}}\left(E_\min, ~E_\max\right) =
        \int _{E_\min}^{E_\max}\text{d}E ~ \mathcal{M}\left( E \right)
        \quad \left[\frac{\text{photons}}{\text{cm}^2\text{s}}\right]$$

        Parameters:
            params : The parameters of the model.
            e_low : The lower bound of the energy bins.
            e_high : The upper bound of the energy bins.
            n_points : The number of points used to integrate the model in each bin.

        !!! info
            This method is internally used in the inference process and should not be used directly. See
            [`photon_flux`][jaxspec.analysis.results.FitResult.photon_flux] to compute
            the photon flux associated with a set of fitted parameters in a
            [`FitResult`][jaxspec.analysis.results.FitResult]
            instead.
        """

        graphdef, state = nnx.split(self)
        state = set_parameters(params, state)

        return nnx.call((graphdef, state)).turbo_flux(
            e_low, e_high, n_points=n_points, return_branches=split_branches
        )[0]

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def energy_flux(self, params, e_low, e_high, n_points=2):
        r"""
        Compute the expected energy flux between $E_\min$ and $E_\max$ by integrating the model.

        $$ \Phi_{\text{energy}}\left(E_\min, ~E_\max\right) =
        \int _{E_\min}^{E_\max}\text{d}E ~ E ~ \mathcal{M}\left( E \right)
        \quad \left[\frac{\text{keV}}{\text{cm}^2\text{s}}\right]$$

        Parameters:
            params : The parameters of the model.
            e_low : The lower bound of the energy bins.
            e_high : The upper bound of the energy bins.
            n_points : The number of points used to integrate the model in each bin.

        !!! info
            This method is internally used in the inference process and should not be used directly. See
            [`energy_flux`](/references/results/#jaxspec.analysis.results.FitResult.energy_flux) to compute
            the energy flux associated with a set of fitted parameters in a
            [`FitResult`](/references/results/#jaxspec.analysis.results.FitResult)
            instead.
        """

        graphdef, state = nnx.split(self)
        state = set_parameters(params, state)

        return nnx.call((graphdef, state)).turbo_flux(
            e_low, e_high, n_points=n_points, energy_flux=True
        )[0]


class ModelComponent(nnx.Module, Composable, ABC):
    """
    Abstract class for model components
    """

    ...


class AdditiveComponent(ModelComponent):
    type = "additive"

    def continuum(self, energy):
        r"""
        Compute the continuum of the component.

        Parameters:
            energy : The energy at which to compute the continuum.
        """
        return jnp.zeros_like(energy)

    def integrated_continuum(self, e_low, e_high):
        r"""
        Compute the integrated continuum between $E_\min$ and $E_\max$.

        Parameters:
            e_low: Lower bound of the energy bin.
            e_high: Upper bound of the energy bin.
        """
        return jnp.zeros_like((e_low + e_high) / 2)

    def _photon_flux(self, e_low, e_high, n_points=2):
        energy = jnp.linspace(e_low, e_high, n_points, axis=-1)
        continuum = self.continuum(energy)
        integrated_continuum = self.integrated_continuum(e_low, e_high)

        return jsp.integrate.trapezoid(continuum, energy, axis=-1) + integrated_continuum

    def _energy_flux(self, e_low, e_high, n_points=2):
        energy = jnp.linspace(e_low, e_high, n_points, axis=-1)
        continuum = self.continuum(energy)
        integrated_continuum = self.integrated_continuum(e_low, e_high)

        return jsp.integrate.trapezoid(
            continuum * energy**2, jnp.log(energy), axis=-1
        ) + integrated_continuum * (e_high - e_low)

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def photon_flux(self, params, e_low, e_high, n_points=2):
        return SpectralModel.from_component(self).photon_flux(
            params, e_low, e_high, n_points=n_points
        )

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def energy_flux(self, params, e_low, e_high, n_points=2):
        return SpectralModel.from_component(self).energy_flux(
            params, e_low, e_high, n_points=n_points
        )


class MultiplicativeComponent(ModelComponent):
    type = "multiplicative"

    def factor(self, energy):
        """
        Absorption factor applied for a given energy

        Parameters:
            energy : The energy at which to compute the factor.
        """
        return jnp.ones_like(energy)
