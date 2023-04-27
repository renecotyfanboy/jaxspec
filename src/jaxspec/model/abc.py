from __future__ import annotations
import haiku as hk
import jax
import jax.numpy as jnp
import networkx as nx
from haiku._src import base
from uuid import uuid4
from abc import ABC, abstractmethod


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

        model_callable = hk.without_apply_rng(hk.transform(lambda e: self.continuum(e)))

        self.eval = model_callable.apply
        self.params = model_callable.init(None, jnp.ones(1))
        self.n_parameters = hk.data_structures.tree_size(self.params)

    def build_namespace(self):
        """
        This method build a namespace for the model components, to avoid name collision
        """

        name_space = []
        new_graph = self.raw_graph.copy()

        for node_id in nx.dag.topological_sort(new_graph):
            node = new_graph.nodes[node_id]

            if node and node['type'] == 'component':
                name_space.append(node['name'])
                n = name_space.count(node['name'])
                nx.set_node_attributes(new_graph, {node_id: name_space[-1] + f'_{n}'}, 'name')

        return new_graph

    def continuum(self, energy):
        """
        This method evaluate the graph of operations and returns the result. It should be transformed using haiku
        """

        continuum = {}

        for node_id in nx.dag.topological_sort(self.graph):

            node = self.graph.nodes[node_id]

            if node and node['type'] == 'component':

                continuum[node_id] = node['component'](name=node['name'], **node['kwargs'])(energy)

            elif node and node['type'] == 'operation':

                component_1 = list(self.graph.in_edges(node_id))[0][0]
                component_2 = list(self.graph.in_edges(node_id))[1][0]
                continuum[node_id] = node['function'](continuum[component_1], continuum[component_2])

        return continuum[list(self.graph.in_edges('out'))[0][0]]

    def find_multiplicative_components(self, node_id):
        """
        Recursively finds all the multiplicative components connected to the node with the given ID.
        """
        node = self.graph.nodes[node_id]
        multiplicative_nodes = []

        if node.get('operation_type') == 'mul':
            # Recursively find all the multiplicative components using the predecessors
            predecessors = self.graph.pred[node_id]
            for node_id in predecessors:
                if self.graph.nodes[node_id].get('component_type') == 'multiplicative':
                    multiplicative_nodes.append(node_id)
                elif self.graph.nodes[node_id].get('operation_type') == 'mul':
                    multiplicative_nodes.extend(self.find_multiplicative_components(node_id))

        return multiplicative_nodes

    def fine_structure(self, e_low, e_high):
        """
        This method evaluate the graph of operations and returns the fine structures.
        """

        fine = {}
        energy = {}
        flux_to_return = jnp.zeros_like(e_low)
        root_nodes = [node_id for node_id, in_degree in self.graph.in_degree(self.graph.nodes) if in_degree == 0]

        for node_id in root_nodes:

            node = self.graph.nodes[node_id]

            if node.get('fine_structure'):

                name = node.get('name')
                path = nx.shortest_path(self.graph, source=node_id, target='out')
                nodes_id_in_path = [node_id for node_id in path]

                flux_from_component, mean_energy = node['component']().fine_structure(e_low, e_high)

                multiplicative_nodes = []

                # Search all multiplicative components connected to this node
                for node_id in nodes_id_in_path[::-1]:
                    multiplicative_nodes.extend([node_id for node_id in self.find_multiplicative_components(node_id)])

                for mul_node in multiplicative_nodes:
                    flux_from_component *= self.graph.nodes[mul_node]['component']()(mean_energy)

                flux_to_return += flux_from_component

        return flux_to_return
                #print(f"{name} : {[self.graph.nodes[node_id]['name'] for node_id in multiplicative_nodes]}")

    def binned(self, e_low, e_high):

        energies = jnp.stack((e_low, e_high))

        return self.fine_structure(e_low, e_high) + jnp.trapz(self.continuum(energies), x=energies, axis=0)

    @classmethod
    def from_component(cls, component: ModelComponent, **kwargs) -> SpectralModel:
        """
        Build a model from a single component
        """

        graph = nx.DiGraph()

        # Add the component node
        # Random static node id to keep it trackable in the graph
        node_id = str(uuid4())

        node_properties = {
            'type': 'component',
            'component_type': component.type,
            'name': component.__name__.lower(),
            'component': component,
            'params': hk.transform(lambda e: component()(e)).init(None, jnp.ones(1)),
            'fine_structure': False
            'kwargs' : kwargs,
            'depth' : 0
        }

        if component.type == 'additive' and component.has_fine_structure:
            node_properties['fine_structure'] = True

        graph.add_node(node_id, **node_properties)

        # Add the output node
        labels = {node_id: component.__name__.lower(), 'out': 'out'}

        graph.add_node('out', type='out', depth=1)
        graph.add_edge(node_id, 'out')

        return cls(graph, labels)

    def compose(self, other: SpectralModel, operation=None, function=None, name=None) ->SpectralModel:
        """
        This function operate a composition between the operation graph of two models
        1) It fuses the two graphs using which joins at the 'out' nodes
        2) It relabels the 'out' node with a unique identifier and labels it with the operation
        3) It links the operation to a new 'out' node
        """

        # Compose the two graphs with their output as common node
        # and add the operation node by overwriting the 'out' node
        node_id = str(uuid4())
        graph = nx.relabel_nodes(nx.compose(self.raw_graph, other.raw_graph), {'out': node_id})
        nx.set_node_attributes(graph, {node_id: 'operation'}, 'type')
        nx.set_node_attributes(graph, {node_id: operation}, 'operation_type')
        nx.set_node_attributes(graph, {node_id: function}, 'function')
        nx.set_node_attributes(graph, {node_id: name}, 'operation_label')

        if node_id in self.labels.keys() or node_id in other.labels.keys() \
                and not set(other.labels.keys()) & set(self.labels.keys()):
            # Check that no node ID is duplicated
            # This should never happen, but if it does, it's a bad luck
            # However it's a good occasion to play lotto tonight
            class BadLuckError(Exception): pass
            raise BadLuckError('Congratulation, you may have experienced an uuid4 collision, this has ~50% '
                               'chance to happen if you generate an uuid4 every second for a century. If '
                               'rerunning this code fixes this issue, consider playing lotto tonight. ')

        # Merge label dictionaries
        labels = self.labels | other.labels
        labels[node_id] = operation

        # Now add the output node and link it to the operation node
        graph.add_node('out', type='out')
        graph.add_edge(node_id, 'out')

        # Compute the new depth of each node
        longest_path = nx.dag_longest_path_length(graph)

        for node in graph.nodes:
            nx.set_node_attributes(graph, {node: longest_path-nx.shortest_path_length(graph, node, 'out')}, 'depth')

        return SpectralModel(graph, labels)

    def __add__(self, other: SpectralModel|ModelComponent) -> SpectralModel:

        if type(other) is not SpectralModel:
            other = SpectralModel.from_component(other)

        return self.compose(other, operation='add', function=lambda x, y: x + y, name='+')

    def __mul__(self, other: SpectralModel|ModelComponent) -> SpectralModel:

        if type(other) is not SpectralModel:
            other = SpectralModel.from_component(other)

        return self.compose(other, operation='mul', function=lambda x, y: x * y, name=r'$\times$')

    def __call__(self, params, energy):
        return self.eval(params, energy)

    def plot(self, figsize=(8, 8)):

        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        pos = nx.multipartite_layout(self.graph, subset_key="depth", scale=1)

        nodes_out = [x for x, y in self.graph.nodes(data=True) if y['type'] == 'out']
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes_out, node_color="tab:green")
        nx.draw_networkx_edges(self.graph, pos, width=1.0)

        nx.draw_networkx_labels(self.graph, pos, labels=nx.get_node_attributes(self.graph, 'name'), font_size=12,
                                font_color="black", bbox={"fc": "tab:red", 'boxstyle': 'round', 'pad': 0.3})
        nx.draw_networkx_labels(self.graph, pos, labels=nx.get_node_attributes(self.graph, 'operation_label'),
                                font_size=12, font_color="black",
                                bbox={"fc": "tab:blue", 'boxstyle': 'circle', 'pad': 0.3})

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class ComponentMetaClass(type(hk.Module)):
    """
    This metaclass enable the construction of model from components with a simple syntax while style enabling
    the components to be used as haiku modules
    """

    def __call__(self, *args, **kwargs):
        """
        This method enable to use model components as haiku modules when folded in a haiku transform
        function and also to instantiate them as SpectralModel when out of a haiku transform
        """

        if not base.frame_stack:

            return SpectralModel.from_component(self, **kwargs)

        else:
            return super().__call__(*args, **kwargs)

    def __add__(self, other) -> SpectralModel:

        if isinstance(other, SpectralModel):
            return SpectralModel.from_component(self) + other
        else:
            return SpectralModel.from_component(self) + SpectralModel.from_component(other)

    def __mul__(self, other):

        if isinstance(other, SpectralModel):
            return SpectralModel.from_component(self) * other
        else:
            return SpectralModel.from_component(self) * SpectralModel.from_component(other)


class ModelComponent(hk.Module, ABC, metaclass=ComponentMetaClass):
    """
    Abstract class for model components
    """
    type: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, energy):
        """
        Return the model evaluated at a given energy
        """
        pass


class AdditiveComponent(ModelComponent, ABC):
    type = 'additive'
    has_fine_structure = False

    def fine_structure(self, e_min, e_max) -> (jax.Array, jax.Array):
        """
        Method for computing the fine structure of an additive model between two energies.
        By default, this is set to 0, which means that the model has no emission lines.
        This should be surcharged by the user if the model has a fine structure.
        """
        if self.has_fine_structure:
            raise NotImplementedError('This model has a fine structure, please surcharge the fine_structure method')

        return jnp.zeros_like(e_min), (e_min+e_max)/2

    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on double exponential quadrature for
        finite intervals to compute an approximation of the integral of a model.

        References
        ----------
        * `Takahasi and Mori (1974) <https://ems.press/journals/prims/articles/2686>`_
        * `Mori and Sugihara (2001) <https://doi.org/10.1016/S0377-0427(00)00501-X>`_
        * `Tanh-sinh quadrature <https://en.wikipedia.org/wiki/Tanh-sinh_quadrature>`_ from Wikipedia

        """

        t = jnp.linspace(-4, 4, 71) # The number of points used is hardcoded and this is not ideal
        # Quadrature nodes as defined in reference
        phi = jnp.tanh(jnp.pi / 2 * jnp.sinh(t))
        dphi = jnp.pi / 2 * jnp.cosh(t) * (1 / jnp.cosh(jnp.pi / 2 * jnp.sinh(t)) ** 2)
        # Change of variable to turn the integral from E_min to E_max into an integral from -1 to 1
        x = (e_max - e_min) / 2 * phi + (e_max + e_min) / 2
        dx = (e_max - e_min) / 2 * dphi

        return jnp.trapz(self(x) * dx, x=t)


class AnalyticalAdditive(AdditiveComponent, ABC):

    @abstractmethod
    def primitive(self, energy):
        r"""
        Analytical primitive of the model

        """
        pass

    def integral(self, e_min, e_max):
        r"""
        Method for integrating an additive model between two energies. It relies on the primitive of the model.
        """

        return self.primitive(e_max) - self.primitive(e_min)


class MultiplicativeComponent(ModelComponent, ABC):
    type = 'multiplicative'

    pass
