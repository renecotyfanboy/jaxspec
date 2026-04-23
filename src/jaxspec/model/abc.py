from __future__ import annotations

import operator

from abc import ABC
from functools import cached_property, partial
from uuid import uuid4

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import networkx as nx

from ._graph_util import compose, export_to_mermaid
from ._parametrizable import ParametrizableMixin


def _apply_pure_updates(state_dict: dict, updates: dict) -> dict:
    """Apply dotted-path ``updates`` into a pure-dict view of an ``nnx.State``.

    A leading routing prefix (e.g. ``"spectrum."``, ``"background."``) is
    dropped when its first segment is not a key in ``state_dict``, so callers
    can pass unified prior keys without stripping the prefix. Updates that
    route to a different model (i.e. whose post-strip path is still unknown)
    are silently skipped, so mixed-prefix dicts from ``FitResult.input_parameters``
    can be passed directly without filtering.
    """
    for path, value in updates.items():
        parts = path.split(".")
        if parts and parts[0] not in state_dict:
            parts = parts[1:]
            if not parts or parts[0] not in state_dict:
                continue
        if not parts:
            continue
        cursor = state_dict
        for name in parts[:-1]:
            cursor = cursor[name]
        cursor[parts[-1]] = value
    return state_dict


class HideUnderscoreMixin:
    """Hide underscore-prefixed attributes from ``nnx.display`` and ``repr``.

    Flax NNX shows any attribute classified as pytree data even when its name
    starts with ``_`` (only *static* underscore attributes are hidden by
    default). For jaxspec models we want the ``_name`` convention to mean
    "implementation detail, don't show" regardless of whether the value is an
    array / :class:`nnx.Variable`.

    Place this mixin **before** :class:`nnx.Module` in the MRO.
    """

    def __treescope_repr__(self, path, subtree_renderer):
        import treescope

        from flax.nnx import visualization

        children = {n: v for n, v in vars(self).items() if not n.startswith("_")}
        return visualization.render_object_constructor(
            object_type=type(self),
            attributes=children,
            path=path,
            subtree_renderer=subtree_renderer,
            color=treescope.formatting_util.color_from_string(type(self).__qualname__),
        )

    def __nnx_repr__(self):
        from flax.nnx import reprlib

        yield reprlib.Object(type=type(self))
        for name, value in vars(self).items():
            if not name.startswith("_"):
                yield reprlib.Attr(name, value)


class ComposableMixin:
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


class SpectralModel(ParametrizableMixin, HideUnderscoreMixin, ComposableMixin, nnx.Module):
    _graph: nx.DiGraph
    prior_prefix: str = "spectrum."

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

        for node, data in self._graph.nodes(data=True):
            if "component" in data["type"]:
                setattr(self, data["name"], data["component"])

    def compose(self, other, operation=None, operation_func=None):
        """
        This function operates a composition between the operation graph of two models
        1) It fuses the two graphs using which joins at the 'out' nodes and change components name to unique identifiers
        2) It relabels the 'out' node with a unique identifier and labels it with the operation
        3) It links the operation to a new 'out' node
        """

        composed_graph = compose(
            self._graph, other._graph, operation=operation, operation_func=operation_func
        )

        return SpectralModel(composed_graph)

    @classmethod
    def from_component(cls, component):
        node_id = str(uuid4())
        _graph = nx.DiGraph()

        node_properties = {
            "type": f"{component.type}_component",
            "name": f"{component.__class__.__name__}_1".lower(),
            "component": component,
            "depth": 0,
        }

        _graph.add_node(node_id, **node_properties)
        _graph.add_node("out", type="out", depth=1)
        _graph.add_edge(node_id, "out")

        return cls(_graph)

    def _find_multiplicative_components(self, node_id):
        """
        Recursively finds all the multiplicative components connected to the node with the given ID.
        """
        node = self._graph.nodes[node_id]
        multiplicative_nodes = []

        if node.get("type") == "mul_operation":
            # Recursively find all the multiplicative components using the predecessors
            predecessors = self._graph.pred[node_id]
            for node_id in predecessors:
                if "multiplicative_component" == self._graph.nodes[node_id].get("type"):
                    multiplicative_nodes.append(node_id)
                elif "mul_operation" == self._graph.nodes[node_id].get("type"):
                    multiplicative_nodes.extend(self._find_multiplicative_components(node_id))

        return multiplicative_nodes

    @property
    def root_nodes(self) -> list[str]:
        return [
            node_id
            for node_id, in_degree in self._graph.in_degree(self._graph.nodes)
            if in_degree == 0 and ("additive" in self._graph.nodes[node_id].get("type"))
        ]

    def _iter_branches(self):
        """Yield ``(branch_name, mult_node_ids, root_node_name)`` for every additive root.

        ``mult_node_ids`` is the deduplicated list of multiplicative-component
        node ids along the path from the additive root to ``out``, in the same
        ``list(set(...))`` order used historically to build branch names and
        multiply absorption factors in :meth:`turbo_flux`.
        """
        for root_node_id in self.root_nodes:
            root_node_name = self._graph.nodes[root_node_id].get("name")
            path = nx.shortest_path(self._graph, source=root_node_id, target="out")

            mult_ids: list[str] = []
            for node_id in path[::-1]:
                mult_ids.extend(self._find_multiplicative_components(node_id))
            mult_ids = list(set(mult_ids))

            branch = (
                "".join(f"{self._graph.nodes[mid].get('name')}*" for mid in mult_ids)
                + root_node_name
            )
            yield branch, mult_ids, root_node_name

    @cached_property
    def branches(self) -> list[str]:
        return [branch for branch, _, _ in self._iter_branches()]

    def turbo_flux(self, e_low, e_high, energy_flux=False, n_points=2, return_branches=False):
        continuum = {}

        ## Evaluate the expected contribution for each component
        for node_id in nx.dag.topological_sort(self._graph):
            node = self._graph.nodes[node_id]

            if node["type"] == "additive_component":
                node_name = node["name"]
                runtime_modules = getattr(self, node_name)

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
                runtime_modules = getattr(self, node_name)
                continuum[node_name] = runtime_modules._factor(e_low, e_high, n_points=n_points)

            else:
                pass

        ## Propagate the absorption for each branch
        branches = {}
        for branch_name, mult_ids, root_node_name in self._iter_branches():
            flux = continuum[root_node_name]
            for mid in mult_ids:
                flux = flux * continuum[self._graph.nodes[mid].get("name")]
            branches[branch_name] = flux

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
        return export_to_mermaid(self._graph, file)

    def _with_params(self, params: dict | None) -> SpectralModel:
        """Return a copy of ``self`` with ``params`` applied as dotted-path
        overrides. Returns ``self`` unchanged when ``params`` is ``None``."""
        if params is None:
            return self
        graphdef, param_state, other = nnx.split(self, nnx.Param, ...)
        pure = nnx.to_pure_dict(param_state)
        pure = _apply_pure_updates(pure, params)
        nnx.replace_by_pure_dict(param_state, pure)
        return nnx.merge(graphdef, param_state, other)

    @partial(jax.jit, static_argnums=0, static_argnames=("n_points", "split_branches"))
    def photon_flux(
        self,
        e_low,
        e_high,
        *,
        params: dict | None = None,
        n_points: int = 2,
        split_branches: bool = False,
    ):
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
        """
        return self._with_params(params).turbo_flux(
            e_low, e_high, n_points=n_points, return_branches=split_branches
        )

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def energy_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
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
        """
        return self._with_params(params).turbo_flux(
            e_low, e_high, n_points=n_points, energy_flux=True
        )

    def integrated_flux(
        self,
        e_min: float,
        e_max: float,
        *,
        params: dict | None = None,
        energy: bool = False,
        n_points: int = 5,
        n_grid: int = 1_000,
    ):
        r"""
        Integrate the photon (default) or energy flux of the model over
        $[E_\min, E_\max]$ and return the scalar result.

        Supports batched parameters: every value in ``params`` may carry
        arbitrary leading axes (e.g. ``(n_chains, n_draws, n_obs)``). The
        result has the same leading shape.

        Parameters:
            e_min : Lower bound of the energy band.
            e_max : Upper bound of the energy band.
            params : Dotted-path parameter dict. If ``None``, uses whatever
                state is currently on the module.
            energy : If ``True``, integrate the energy flux (keV/cm²/s);
                otherwise the photon flux (photons/cm²/s).
            n_points : Quadrature points per energy bin.
            n_grid : Number of grid points across $[E_\min, E_\max]$.
        """
        energy_grid = jnp.linspace(e_min, e_max, n_grid)
        e_low = energy_grid[:-1]
        e_high = energy_grid[1:]
        flux_fn = self.energy_flux if energy else self.photon_flux

        if params is None:
            return flux_fn(e_low, e_high, n_points=n_points).sum(axis=-1)

        flat_tree, pytree_def = jax.tree.flatten(params)

        @jax.jit
        @jnp.vectorize
        def vectorized_flux(*pars):
            parameters_pytree = jax.tree.unflatten(pytree_def, pars)
            return flux_fn(e_low, e_high, params=parameters_pytree, n_points=n_points)

        return vectorized_flux(*flat_tree).sum(axis=-1)


class ModelComponent(HideUnderscoreMixin, nnx.Module, ComposableMixin, ABC):
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

        return (
            jsp.integrate.trapezoid(continuum * energy**2, jnp.log(energy), axis=-1)
            + integrated_continuum * (e_high + e_low) / 2.0
        )

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def photon_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
        return SpectralModel.from_component(self).photon_flux(
            e_low, e_high, params=params, n_points=n_points
        )

    @partial(jax.jit, static_argnums=0, static_argnames="n_points")
    def energy_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
        return SpectralModel.from_component(self).energy_flux(
            e_low, e_high, params=params, n_points=n_points
        )


class MultiplicativeComponent(ModelComponent):
    type = "multiplicative"

    def _factor(self, e_low, e_high, n_points=2):
        energy = jnp.linspace(e_low, e_high, n_points, axis=-1)
        factor = self.factor(energy)

        return jsp.integrate.trapezoid(factor * energy, jnp.log(energy), axis=-1) / (e_high - e_low)
        # return jnp.mean(factor, axis = -1)

    def factor(self, energy):
        """
        Absorption factor applied for a given energy

        Parameters:
            energy : The energy at which to compute the factor.
        """
        return jnp.ones_like(energy)
