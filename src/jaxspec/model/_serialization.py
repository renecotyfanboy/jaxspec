"""Versioned, JSON-compatible model descriptions with explicit numerical state."""

from __future__ import annotations

import inspect
import math

import jax
import networkx as nx
import numpy as np

from flax import nnx

from ._graph_util import validate_graph


def _json_config(value):
    """Copy literal constructor configuration, rejecting opaque or non-finite values."""
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufU":
            raise ValueError("Model configuration must contain finite JSON literals only.")
        # Most NumPy scalars have an exact Python counterpart. Extended-precision
        # scalars can return another NumPy scalar from item(); reject those rather
        # than recurse forever or silently discard precision.
        value = value.item()
        if isinstance(value, np.generic):
            raise ValueError("Model configuration requires scalars representable as JSON literals.")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list | tuple):
        return [_json_config(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _json_config(item) for key, item in value.items()}
    raise ValueError("Model configuration must contain finite JSON literals only.")


def _encode_state_leaf(path, leaf):
    """Describe one tracked numerical leaf without executable Python objects."""
    variable = isinstance(leaf, nnx.Variable)
    value = leaf.get_value() if variable else leaf
    if type(value) in (float, int, bool):
        storage = "scalar"
    elif isinstance(value, np.generic):
        storage = "numpy_scalar"
    elif isinstance(value, np.ndarray):
        storage = "numpy"
    elif isinstance(value, jax.Array):
        storage = "jax"
    else:
        raise ValueError(f"Unsupported numerical state at {path!r}: {type(value).__name__}.")
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"State at {path!r} must contain finite real numbers or booleans.")
    if not path or any(type(part) not in (str, int) for part in path):
        raise ValueError(f"Unsupported numerical state path: {path!r}.")
    return {
        "path": list(path),
        "variable": f"{type(leaf).__module__}.{type(leaf).__qualname__}" if variable else None,
        "storage": storage,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "value": _json_config(array.tolist()),
    }


def _decode_state_leaf(record):
    """Validate a numerical-state record before any model constructor runs."""
    keys = {"path", "variable", "storage", "dtype", "shape", "value"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("Invalid numerical state record.")
    path, shape = record["path"], record["shape"]
    if not isinstance(path, list) or not path or any(type(p) not in (str, int) for p in path):
        raise ValueError("Numerical state paths must be non-empty lists of strings or integers.")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Numerical state shapes must be lists of non-negative integers.")
    if record["variable"] is not None and not isinstance(record["variable"], str):
        raise ValueError("Invalid numerical state variable type.")
    if record["storage"] not in ("scalar", "numpy_scalar", "numpy", "jax"):
        raise ValueError("Unknown numerical state storage kind.")
    if not isinstance(record["dtype"], str):
        raise ValueError("Invalid numerical state dtype.")
    try:
        dtype = np.dtype(record["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Numerical state requires a real numeric or boolean dtype.")
        values = np.asarray(record["value"])
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("Numerical state must contain finite real values.")
        array = values.astype(dtype)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid numerical state values or dtype.") from exc
    if 0 in shape and array.size == 0:
        array = array.reshape(shape)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError("Numerical state values do not match their finite shape and dtype.")
    if dtype.kind in "biu" and not np.array_equal(array, values):
        raise ValueError("Numerical state values cannot be represented by their dtype.")
    if record["storage"] in ("scalar", "numpy_scalar") and shape:
        raise ValueError("Scalar numerical state cannot have array dimensions.")
    return tuple(path), array


def serialize_model(model):
    """Return a version-1 JSON-compatible model description.

    The graph records ordered operations, public component names and constructor
    configuration. ``state`` separately records the current NNX parameters and
    tracked buffers. Constructor-managed external tables are reloaded by their
    configuration; their file contents are not embedded. Unsupported configuration
    or non-finite/opaque numerical state raises ``ValueError``.
    """
    validate_graph(model._graph)
    ordered = list(nx.topological_sort(model._graph))
    identifiers = {node: "out" if node == "out" else f"n{i}" for i, node in enumerate(ordered)}
    nodes = []
    for node in ordered:
        metadata = model._graph.nodes[node]
        item = {"id": identifiers[node], "type": metadata["type"]}
        if metadata["type"].endswith("_component"):
            component = getattr(model, metadata["name"])
            config = _json_config(component.get_config())
            if not isinstance(config, dict):
                raise ValueError("Component get_config() must return a dictionary.")
            item.update(name=metadata["name"], component=type(component).__name__, config=config)
        nodes.append(item)
    edges = [
        {"source": identifiers[source], "target": identifiers[target], "position": position}
        for target in ordered
        for position, source in sorted(
            (model._graph.edges[source, target].get("position", 0), source)
            for source in model._graph.predecessors(target)
        )
    ]
    state = [_encode_state_leaf(path, leaf) for path, leaf in nnx.to_flat_state(nnx.state(model))]
    return {"version": 1, "nodes": nodes, "edges": edges, "state": state}


def _restore_graph(data, registry):
    """Validate serialized graph structure and configurations without constructing models."""
    from .abc import OperatorComponent

    if not isinstance(data, dict) or set(data) != {"version", "nodes", "edges", "state"}:
        raise ValueError("Invalid serialized model schema.")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported serialized model version.")
    if any(not isinstance(data[key], list) for key in ("nodes", "edges", "state")):
        raise ValueError("Serialized model nodes, edges and state must be lists.")
    graph, constructors = nx.DiGraph(), {}
    for item in data["nodes"]:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ValueError("Invalid serialized model node.")
        component_node = item["type"].endswith("_component")
        keys = {"id", "type", "name", "component", "config"} if component_node else {"id", "type"}
        if set(item) != keys or not isinstance(item["id"], str) or item["id"] in graph:
            raise ValueError("Invalid or duplicate serialized model node identifier.")
        metadata = {"type": item["type"]}
        if component_node:
            name = item["component"]
            if not isinstance(name, str) or name.casefold() not in registry:
                raise ValueError(f"Unknown model component: {name!r}.")
            component = registry[name.casefold()]
            if inspect.isabstract(component):
                raise ValueError(f"Model component {name!r} is abstract.")
            kind = getattr(component, "type", None)
            if item["type"] != f"{kind}_component":
                raise ValueError("Serialized component kind does not match its registered class.")
            config = _json_config(item["config"])
            if not isinstance(config, dict) or any(
                not key.isidentifier() or key.startswith("_") for key in config
            ):
                raise ValueError("Serialized component configuration must be a dictionary.")
            try:
                if component.__init__ is object.__init__ and config:
                    raise TypeError("this component does not accept constructor arguments")
                inspect.signature(component.__init__).bind(None, **config)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid constructor configuration for {name!r}.") from exc
            metadata["name"] = item["name"]
            if kind == "operator":
                if not issubclass(component, OperatorComponent):
                    raise ValueError(f"Operator {name!r} must inherit from OperatorComponent.")
                metadata.update(input_type=component.input_type, output_type=component.output_type)
            constructors[item["id"]] = (component, config)
        graph.add_node(item["id"], **metadata)
    for edge in data["edges"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "target", "position"}:
            raise ValueError("Invalid serialized graph edge.")
        source, target, position = edge["source"], edge["target"], edge["position"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Serialized edge endpoints must be node identifiers.")
        if source not in graph or target not in graph or graph.has_edge(source, target):
            raise ValueError("Serialized edges must connect known nodes exactly once.")
        if type(position) is not int or position not in (0, 1):
            raise ValueError("Serialized edge positions must be 0 or 1.")
        graph.add_edge(source, target, position=position)
    validate_graph(graph)
    for node in graph:
        if graph.in_degree(node) == 1:
            source = next(graph.predecessors(node))
            if graph.edges[source, node]["position"] != 0:
                raise ValueError("Unary graph edges require position 0.")
    return graph, constructors
