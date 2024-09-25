from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import uuid4

import haiku as hk
import jax
import jax.numpy as jnp
import networkx as nx
import rich

from haiku._src import base
from jax.scipy.integrate import trapezoid
from rich.table import Table
from simpleeval import simple_eval


class SpectralModel:
    """
    This class is supposed to handle the composition of models through basic
    operations, and allows tracking of the operation graph and individual parameters.
    """

    raw_graph: nx.DiGraph
    graph: nx.DiGraph
    labels: dict[str, str]
    n_parameters: int

    def __init__(self, internal_graph, labels):
        self.raw_graph = internal_graph
        self.labels = labels
        self.graph = self.build_namespace()

        self.n_parameters = hk.data_structures.tree_size(self.params)

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

    def __str__(self) -> SpectralModel:
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

        return build_expression("out")[1:-1]

    @property
    def transformed_func_photon(self):
        def func_to_transform(e_low, e_high, n_points=2):
            return self.flux(e_low, e_high, n_points=n_points, energy_flux=False)

        return hk.without_apply_rng(hk.transform(func_to_transform))

    @property
    def transformed_func_energy(self):
        def func_to_transform(e_low, e_high, n_points=2):
            return self.flux(e_low, e_high, n_points=n_points, energy_flux=True)

        return hk.without_apply_rng(hk.transform(func_to_transform))

    @property
    def params(self):
        return self.transformed_func_photon.init(None, jnp.ones(10), jnp.ones(10))

    def __rich_repr__(self):
        table = Table(title=str(self))

        table.add_column("Component", justify="right", style="bold", no_wrap=True)
        table.add_column("Parameter")

        params = self.params

        for component in params.keys():
            once = True

            for parameters in params[component].keys():
                table.add_row(component if once else "", parameters)
                once = False

        return table

    def __repr_html_(self):
        return self.__rich_repr__()

    def __repr__(self):
        rich.print(self.__rich_repr__())
        return ""

    def photon_flux(self, params, e_low, e_high, n_points=2):
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

        params = jax.tree_map(lambda x: jnp.asarray(x), params)
        e_low = jnp.asarray(e_low)
        e_high = jnp.asarray(e_high)

        return self.transformed_func_photon.apply(params, e_low, e_high, n_points=n_points)

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

        params = jax.tree_map(lambda x: jnp.asarray(x), params)
        e_low = jnp.asarray(e_low)
        e_high = jnp.asarray(e_high)

        return self.transformed_func_energy.apply(params, e_low, e_high, n_points=n_points)

    def build_namespace(self):
        """
        This method build a namespace for the model components, to avoid name collision
        """

        name_space = []
        new_graph = self.raw_graph.copy()

        for node_id in nx.dag.topological_sort(new_graph):
            node = new_graph.nodes[node_id]

            if node and node["type"] == "component":
                name_space.append(node["name"])
                n = name_space.count(node["name"])
                nx.set_node_attributes(new_graph, {node_id: name_space[-1] + f"_{n}"}, "name")

        return new_graph

    def flux(self, e_low, e_high, energy_flux=False, n_points=2):
        """
        This method return the expected counts between e_low and e_high by integrating the model.
        It contains most of the "usine à gaz" which makes jaxspec works.
        It evaluates the graph of operations and returns the result.
        It should be transformed using haiku.
        """

        # TODO : enable interpolation and integration with more than 2 points for the continuum

        if n_points == 2:
            energies = jnp.hstack((e_low, e_high[-1]))
            energies_to_integrate = jnp.stack((e_low, e_high))

        else:
            energies_to_integrate = jnp.linspace(e_low, e_high, n_points)
            energies = energies_to_integrate

        fine_structures_flux = jnp.zeros_like(e_low)
        runtime_modules = {}
        continuum = {}

        # Iterate through the graph in topological order and
        # compute the continuum contribution for each component

        for node_id in nx.dag.topological_sort(self.graph):
            node = self.graph.nodes[node_id]

            # Instantiate the haiku modules
            if node and node["type"] == "component":
                runtime_modules[node_id] = node["component"](name=node["name"], **node["kwargs"])
                continuum[node_id] = runtime_modules[node_id].continuum(energies)

            elif node and node["type"] == "operation":
                component_1 = list(self.graph.in_edges(node_id))[0][0]  # noqa: RUF015
                component_2 = list(self.graph.in_edges(node_id))[1][0]
                continuum[node_id] = node["function"](
                    continuum[component_1], continuum[component_2]
                )

        if n_points == 2:
            flux_1D = continuum[list(self.graph.in_edges("out"))[0][0]]  # noqa: RUF015
            flux = jnp.stack((flux_1D[:-1], flux_1D[1:]))

        else:
            flux = continuum[list(self.graph.in_edges("out"))[0][0]]  # noqa: RUF015

        if energy_flux:
            continuum_flux = trapezoid(
                flux * energies_to_integrate**2,
                x=jnp.log(energies_to_integrate),
                axis=0,
            )

        else:
            continuum_flux = trapezoid(
                flux * energies_to_integrate, x=jnp.log(energies_to_integrate), axis=0
            )

        # Iterate from the root nodes to the output node and
        # compute the fine structure contribution for each component

        root_nodes = [
            node_id
            for node_id, in_degree in self.graph.in_degree(self.graph.nodes)
            if in_degree == 0 and self.graph.nodes[node_id].get("component_type") == "additive"
        ]

        for root_node_id in root_nodes:
            path = nx.shortest_path(self.graph, source=root_node_id, target="out")
            nodes_id_in_path = [node_id for node_id in path]

            flux_from_component, mean_energy = runtime_modules[root_node_id].emission_lines(
                e_low, e_high
            )

            multiplicative_nodes = []

            # Search all multiplicative components connected to this node
            # and apply them at mean energy
            for node_id in nodes_id_in_path[::-1]:
                multiplicative_nodes.extend(
                    [node_id for node_id in self.find_multiplicative_components(node_id)]
                )

            for mul_node in multiplicative_nodes:
                flux_from_component *= runtime_modules[mul_node].continuum(mean_energy)

            if energy_flux:
                fine_structures_flux += trapezoid(
                    flux_from_component * energies_to_integrate,
                    x=jnp.log(energies_to_integrate),
                    axis=0,
                )

            else:
                fine_structures_flux += flux_from_component

        return continuum_flux + fine_structures_flux

    def find_multiplicative_components(self, node_id):
        """
        Recursively finds all the multiplicative components connected to the node with the given ID.
        """
        node = self.graph.nodes[node_id]
        multiplicative_nodes = []

        if node.get("operation_type") == "mul":
            # Recursively find all the multiplicative components using the predecessors
            predecessors = self.graph.pred[node_id]
            for node_id in predecessors:
                if self.graph.nodes[node_id].get("component_type") == "multiplicative":
                    multiplicative_nodes.append(node_id)
                elif self.graph.nodes[node_id].get("operation_type") == "mul":
                    multiplicative_nodes.extend(self.find_multiplicative_components(node_id))

        return multiplicative_nodes

    def __call__(self, pars, e_low, e_high, **kwargs):
        return self.photon_flux(pars, e_low, e_high, **kwargs)

    @classmethod
    def from_component(cls, component, **kwargs) -> SpectralModel:
        """
        Build a model from a single component
        """

        graph = nx.DiGraph()

        # Add the component node
        # Random static node id to keep it trackable in the graph
        node_id = str(uuid4())

        if component.type == "additive":

            def lam_func(e):
                return (
                    component(**kwargs).continuum(e)
                    + component(**kwargs).emission_lines(e, e + 1)[0]
                )

        elif component.type == "multiplicative":

            def lam_func(e):
                return component().continuum(e)

        else:

            def lam_func(e):
                return print("Some components are not working at this stage")

        node_properties = {
            "type": "component",
            "component_type": component.type,
            "name": component.__name__.lower(),
            "component": component,
            # "params": hk.transform(lam_func).init(None, jnp.ones(1)),
            "fine_structure": False,
            "kwargs": kwargs,
            "depth": 0,
        }

        graph.add_node(node_id, **node_properties)

        # Add the output node
        labels = {node_id: component.__name__.lower(), "out": "out"}

        graph.add_node("out", type="out", depth=1)
        graph.add_edge(node_id, "out")

        return cls(graph, labels)

    def compose(
        self, other: SpectralModel, operation=None, function=None, name=None
    ) -> SpectralModel:
        """
        This function operate a composition between the operation graph of two models
        1) It fuses the two graphs using which joins at the 'out' nodes
        2) It relabels the 'out' node with a unique identifier and labels it with the operation
        3) It links the operation to a new 'out' node
        """

        # Compose the two graphs with their output as common node
        # and add the operation node by overwriting the 'out' node
        node_id = str(uuid4())
        graph = nx.relabel_nodes(nx.compose(self.raw_graph, other.raw_graph), {"out": node_id})
        nx.set_node_attributes(graph, {node_id: "operation"}, "type")
        nx.set_node_attributes(graph, {node_id: operation}, "operation_type")
        nx.set_node_attributes(graph, {node_id: function}, "function")
        nx.set_node_attributes(graph, {node_id: name}, "operation_label")

        # Merge label dictionaries
        labels = self.labels | other.labels
        labels[node_id] = operation

        # Now add the output node and link it to the operation node
        graph.add_node("out", type="out")
        graph.add_edge(node_id, "out")

        # Compute the new depth of each node
        longest_path = nx.dag_longest_path_length(graph)

        for node in graph.nodes:
            nx.set_node_attributes(
                graph,
                {node: longest_path - nx.shortest_path_length(graph, node, "out")},
                "depth",
            )

        return SpectralModel(graph, labels)

    def __add__(self, other: SpectralModel) -> SpectralModel:
        return self.compose(other, operation="add", function=lambda x, y: x + y, name="+")

    def __mul__(self, other: SpectralModel) -> SpectralModel:
        return self.compose(other, operation="mul", function=lambda x, y: x * y, name=r"*")

    def export_to_mermaid(self, file=None):
        mermaid_code = "graph LR\n"  # LR = left to right

        # Add nodes
        for node, attributes in self.graph.nodes(data=True):
            if attributes["type"] == "component":
                name, number = attributes["name"].split("_")
                mermaid_code += f'    {node}("{name.capitalize()} ({number})")\n'

            if attributes["type"] == "operation":
                if attributes["operation_type"] == "add":
                    mermaid_code += f"    {node}{{+}}\n"

                if attributes["operation_type"] == "mul":
                    mermaid_code += f"    {node}{{x}}\n"

            if attributes["type"] == "out":
                mermaid_code += f'    {node}("Output")\n'

        # Draw connexion between nodes
        for source, target in self.graph.edges():
            mermaid_code += f"    {source} --> {target}\n"

        if file is None:
            return mermaid_code
        else:
            with open(file, "w") as f:
                f.write(mermaid_code)

    def plot(self, figsize=(8, 8)):
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        pos = nx.multipartite_layout(self.graph, subset_key="depth", scale=1)

        nodes_out = [x for x, y in self.graph.nodes(data=True) if y["type"] == "out"]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes_out, node_color="tab:green")
        nx.draw_networkx_edges(self.graph, pos, width=1.0)

        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=nx.get_node_attributes(self.graph, "name"),
            font_size=12,
            font_color="black",
            bbox={"fc": "tab:red", "boxstyle": "round", "pad": 0.3},
        )
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=nx.get_node_attributes(self.graph, "operation_label"),
            font_size=12,
            font_color="black",
            bbox={"fc": "tab:blue", "boxstyle": "circle", "pad": 0.3},
        )

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class ComponentMetaClass(type(hk.Module)):
    """
    This metaclass enable the construction of model from components with a simple
    syntax while style enabling the components to be used as haiku modules.
    """

    def __call__(self, **kwargs) -> SpectralModel:
        """
        This method enable to use model components as haiku modules when folded in a haiku transform
        function and also to instantiate them as SpectralModel when out of a haiku transform
        """

        if not base.frame_stack:
            return SpectralModel.from_component(self, **kwargs)

        else:
            return super().__call__(**kwargs)


class ModelComponent(hk.Module, ABC, metaclass=ComponentMetaClass):
    """
    Abstract class for model components
    """

    type: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AdditiveComponent(ModelComponent, ABC):
    type = "additive"

    def continuum(self, energy):
        """
        Method for computing the continuum associated to the model.
        By default, this is set to 0, which means that the model has no continuum.
        This should be overloaded by the user if the model has a continuum.
        """

        return jnp.zeros_like(energy)

    def emission_lines(self, e_min, e_max) -> (jax.Array, jax.Array):
        """
        Method for computing the fine structure of an additive model between two energies.
        By default, this is set to 0, which means that the model has no emission lines.
        This should be overloaded by the user if the model has a fine structure.
        """

        return jnp.zeros_like(e_min), (e_min + e_max) / 2

    '''
    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on
        double exponential quadrature for finite intervals to compute an approximation
        of the integral of a model.

        references
        ----------
        * $Takahasi and Mori (1974) <https://ems.press/journals/prims/articles/2686>$_
        * $Mori and Sugihara (2001) <https://doi.org/10.1016/S0377-0427(00)00501-X>$_
        * $Tanh-sinh quadrature <https://en.wikipedia.org/wiki/Tanh-sinh_quadrature>$_ from Wikipedia

        """

        t = jnp.linspace(-4, 4, 71) # The number of points used is hardcoded and this is not ideal
        # Quadrature nodes as defined in reference
        phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
        # Change of variable to turn the integral from E_min to E_max into an integral from -1 to 1
        x = (e_max - e_min) / 2 * phi + (e_max + e_min) / 2
        dx = (e_max - e_min) / 2 * dphi

        return jnp.trapz(self(x) * dx, x=t)
    '''


class MultiplicativeComponent(ModelComponent, ABC):
    type = "multiplicative"

    @abstractmethod
    def continuum(self, energy): ...
