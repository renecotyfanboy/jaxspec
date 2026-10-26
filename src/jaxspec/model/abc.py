from __future__ import annotations

from abc import ABC
from uuid import uuid4

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import networkx as nx

from flax import nnx

from ._graph_util import compose, export_to_mermaid, validate_graph
from ._mixin import Composable, HasDerivedQuantities, HideUnderscore


class SpectralModel(HideUnderscore, Composable, nnx.Module):
    _graph: nx.DiGraph

    def __init__(self, graph: nx.DiGraph):
        graph = graph.copy()
        self._kind = validate_graph(graph)
        for node, data in graph.nodes(data=True):
            if "component" in data["type"]:
                component = data.pop("component", None)
                if (
                    not isinstance(component, ModelComponent)
                    or data["type"] != f"{component.type}_component"
                ):
                    raise ValueError(f"Component {data['name']!r} does not match its graph kind.")
                setattr(self, data["name"], component)
        # NNX tracks numerical state through the named modules, never through a
        # second reference hidden in a static NetworkX graph.
        self._graph = nx.freeze(graph)

    def compose(self, other, operation=None):
        """Combine two spectral models with an addition or multiplication node."""

        if not isinstance(other, SpectralModel):
            raise TypeError("Model arithmetic requires spectral components or models.")
        graphs = []
        for model in (self, other):
            graph = model._graph.copy()
            for _, node in graph.nodes(data=True):
                if "component" in node["type"]:
                    node["component"] = nnx.clone(getattr(model, node["name"]))
            graphs.append(graph)
        return SpectralModel(compose(*graphs, operation=operation))

    @classmethod
    def from_component(cls, component):
        """Wrap an emission or factor component, retaining its public parameter names."""
        if not isinstance(component, ModelComponent):
            raise TypeError("Expected a spectral ModelComponent.")
        node_id = str(uuid4())
        _graph = nx.DiGraph()

        node_properties = {
            "type": f"{component.type}_component",
            "name": f"{component.__class__.__name__}_1".lower(),
            "component": component,
        }

        _graph.add_node(node_id, **node_properties)
        _graph.add_node("out", type="out")
        _graph.add_edge(node_id, "out")

        return cls(_graph)

    @property
    def graph(self) -> nx.DiGraph:
        """Return a detached structural graph with ordered edges and no model arrays."""
        return self._graph.copy()

    @property
    def parameters(self) -> dict:
        """Current fitted values keyed by public paths, e.g. ``powerlaw_1.norm``."""
        return {
            ".".join(map(str, path)): variable.get_value()
            for path, variable in nnx.to_flat_state(nnx.state(self, nnx.Param))
        }

    @property
    def root_nodes(self) -> list[str]:
        return [
            node_id
            for node_id, in_degree in self._graph.in_degree(self._graph.nodes)
            if in_degree == 0 and ("additive" in self._graph.nodes[node_id].get("type"))
        ]

    def _evaluate_branches(self, e_low=None, e_high=None, *, energy_flux=False, n_points=2):
        """Evaluate the ordered tree into one value per additive branch.

        With no grid, only branch names are inspected; component code is not run.
        """
        inspect_only = e_low is None
        results = {}
        for node_id in nx.topological_sort(self._graph):
            node = self._graph.nodes[node_id]
            node_type = node["type"]
            children = sorted(
                self._graph.predecessors(node_id),
                key=lambda child: self._graph.edges[child, node_id].get("position", 0),
            )
            if node_type == "out":
                _, _, branches = results[children[0]]
                return {"*".join(key): value for key, value in branches.items()}
            if node_type.endswith("_component"):
                name = node["name"]
                component = getattr(self, name)
                kind = component.type
                if inspect_only:
                    value = None
                elif kind == "multiplicative":
                    value = component._factor(e_low, e_high, n_points=n_points)
                elif energy_flux:
                    value = component._energy_flux(e_low, e_high, n_points=n_points)
                else:
                    value = component._photon_flux(e_low, e_high, n_points=n_points)
                results[node_id] = kind, name, {(name,): value}
                continue
            left_kind, left_expr, left = results[children[0]]
            right_kind, right_expr, right = results[children[1]]
            kind = "additive" if "additive" in (left_kind, right_kind) else "multiplicative"
            if node_type == "add_operation":
                expression = f"({left_expr}+{right_expr})"
                if kind == "additive":
                    branches = {**left, **right}
                else:
                    value = None if inspect_only else sum(left.values()) + sum(right.values())
                    branches = {(expression,): value}
            else:
                expression = f"({left_expr}*{right_expr})"
                branches = {}
                for left_key, left_value in left.items():
                    for right_key, right_value in right.items():
                        if left_kind == "additive":
                            key = (*sorted((*left_key[:-1], *right_key)), left_key[-1])
                        elif right_kind == "additive":
                            key = (*sorted((*left_key, *right_key[:-1])), right_key[-1])
                        else:
                            key = tuple(sorted((*left_key, *right_key)))
                        branches[key] = None if inspect_only else left_value * right_value
            results[node_id] = kind, expression, branches

    @property
    def branches(self) -> list[str]:
        """Deterministic branch labels, one per additive component."""
        return list(self._evaluate_branches())

    def flux_func(self, e_low, e_high, energy_flux=False, n_points=2, return_branches=False):
        """Evaluate integrated photon flux on explicit lower/upper bin edges in keV."""
        if self._kind != "additive":
            raise ValueError(
                "This model has no additive component at its output, so it produces no flux. "
                "Multiply dimensionless factors by an emitting component."
            )
        if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 2:
            raise ValueError("n_points must be an integer of at least two.")
        e_low, e_high = jnp.asarray(e_low), jnp.asarray(e_high)
        if e_low.shape != e_high.shape:
            raise ValueError("Lower and upper energy bounds must have matching shapes.")
        branches = self._evaluate_branches(
            e_low, e_high, energy_flux=energy_flux, n_points=n_points
        )
        return branches if return_branches else sum(branches.values())

    def to_mermaid(self, file: str | None = None):
        """
        This method returns the mermaid representation of the model.

        Parameters:
            file: The file to write the mermaid representation to.

        Returns:
            A string containing the mermaid representation of the model.
        """
        return export_to_mermaid(self._graph, file)

    def _with_params(self, params: dict | None) -> SpectralModel:
        """Return a copy of ``self`` with ``params`` applied as dotted-path
        overrides. Returns ``self`` unchanged when ``params`` is ``None``.

        A leading routing prefix (e.g. ``"spectrum."``) is dropped when its
        first segment isn't a key in the param tree, and updates that still
        don't route to a known leaf after stripping are silently skipped.
        That lets callers pass unified prior dicts (e.g.
        ``FitResult.input_parameters``) without filtering by prefix.
        """
        if params is None:
            return self
        graphdef, param_state, other = nnx.split(self, nnx.Param, ...)
        pure = nnx.to_pure_dict(param_state)
        for path, value in params.items():
            parts = path.split(".")
            if parts and parts[0] not in pure:
                parts = parts[1:]
                if not parts or parts[0] not in pure:
                    continue
            cursor = pure
            for name in parts[:-1]:
                cursor = cursor[name]
            cursor[parts[-1]] = value
        nnx.replace_by_pure_dict(param_state, pure)
        return nnx.merge(graphdef, param_state, other)

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
            params: The parameters of the model.
            e_low: The lower bound of the energy bins.
            e_high: The upper bound of the energy bins.
            n_points: The number of points used to integrate the model in each bin.
        """
        return self._with_params(params).flux_func(
            e_low, e_high, n_points=n_points, return_branches=split_branches
        )

    def energy_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
        r"""
        Compute the expected energy flux between $E_\min$ and $E_\max$ by integrating the model.

        $$ \Phi_{\text{energy}}\left(E_\min, ~E_\max\right) =
        \int _{E_\min}^{E_\max}\text{d}E ~ E ~ \mathcal{M}\left( E \right)
        \quad \left[\frac{\text{keV}}{\text{cm}^2\text{s}}\right]$$

        Parameters:
            params: The parameters of the model.
            e_low: The lower bound of the energy bins.
            e_high: The upper bound of the energy bins.
            n_points: The number of points used to integrate the model in each bin.
        """
        return self._with_params(params).flux_func(
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
            e_min: Lower bound of the energy band.
            e_max: Upper bound of the energy band.
            params: Dotted-path parameter dict. If ``None``, uses whatever
                state is currently on the module.
            energy: If ``True``, integrate the energy flux (keV/cm²/s);
                otherwise the photon flux (photons/cm²/s).
            n_points: Quadrature points per energy bin.
            n_grid: Number of grid points across $[E_\min, E_\max]$.
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


class ModelComponent(HideUnderscore, HasDerivedQuantities, nnx.Module, Composable, ABC):
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
            energy: The energy at which to compute the continuum.
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

    def photon_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
        return SpectralModel.from_component(self).photon_flux(
            e_low, e_high, params=params, n_points=n_points
        )

    def energy_flux(self, e_low, e_high, *, params: dict | None = None, n_points: int = 2):
        return SpectralModel.from_component(self).energy_flux(
            e_low, e_high, params=params, n_points=n_points
        )


class MultiplicativeComponent(ModelComponent):
    type = "multiplicative"

    def _factor(self, e_low, e_high, n_points=2):
        energy = jnp.linspace(e_low, e_high, n_points, axis=-1)
        factor = self.factor(energy)
        bin_width = e_high - e_low
        # Shift clipping can collapse a bin; avoid division by zero.
        safe_width = jnp.where(bin_width > 0, bin_width, 1.0)
        avg = jsp.integrate.trapezoid(factor * energy, jnp.log(energy), axis=-1) / safe_width
        return jnp.where(bin_width > 0, avg, 0.0)

    def factor(self, energy):
        """
        Absorption factor applied for a given energy

        Parameters:
            energy: The energy at which to compute the factor.
        """
        return jnp.ones_like(energy)
