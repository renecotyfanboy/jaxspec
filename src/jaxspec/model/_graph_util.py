"""Helper functions to deal with the graph logic within model building"""

import re

from uuid import uuid4

import networkx as nx


def get_component_names(graph: nx.DiGraph):
    """
    Get the set of component names from the nodes of a graph.

    Parameters:
        graph: The graph to get the component names from.
    """
    return set(
        data["name"] for _, data in graph.nodes(data=True) if "component" in data.get("type")
    )


def increment_name(name: str, used_names: set):
    """
    Increment the suffix number in a name if it is formated as 'name_1'.

    Parameters:
        name: The name to increment.
        used_names: The set of names that are already used.
    """
    match = re.match(r"(.*?)(?:_(\d+))?$", name)
    base_name = match.group(1)
    suffix = match.group(2)
    if suffix:
        number = int(suffix)
    else:
        number = 1

    new_name = name
    while new_name in used_names:
        number += 1
        new_name = f"{base_name}_{number}"

    return new_name


def compose_with_rename(graph_1: nx.DiGraph, graph_2: nx.DiGraph):
    """
    Compose two graphs by updating the 'name' attributes of nodes in graph_2,
    and return the graph joined on the 'out' node.

    Parameters:
        graph_1: The first graph to compose.
        graph_2: The second graph to compose.
    """

    # Every occurrence is a distinct expression node, including ``model + model``.
    # Relabel on a copy so neither operand's topology or names are mutated.
    graph_2 = nx.relabel_nodes(
        graph_2, {node: str(uuid4()) for node in graph_2 if node != "out"}, copy=True
    )

    used_names = get_component_names(graph_1)

    for node, data in graph_2.nodes(data=True):
        if "component" in data.get("type"):
            original_name = data["name"]
            new_name = original_name

            if new_name in used_names:
                new_name = increment_name(original_name, used_names)
                data["name"] = new_name
                used_names.add(new_name)

            else:
                used_names.add(new_name)

    composed_graph = nx.compose(graph_1, graph_2)

    return composed_graph


def compose(graph_1: nx.DiGraph, graph_2: nx.DiGraph, operation: str = ""):
    """
    Compose two graphs by joining the 'out' node of graph_1 and graph_2, and turning
    it to an 'operation' node with the relevant operator and add a new 'out' node.

    The operation is stored in the node's ``type`` attribute as ``"add_operation"`` or
    ``"mul_operation"`` and dispatched by
    :meth:`~jaxspec.model.abc.SpectralModel.flux_func`.

    Parameters:
        graph_1: The first graph to compose.
        graph_2: The second graph to compose.
        operation: The string describing the operation to perform.
    """

    if operation not in {"add", "mul"}:
        raise ValueError(f"Unsupported model operation: {operation!r}.")
    graph_1, graph_2 = graph_1.copy(), graph_2.copy()
    for position, operand in enumerate((graph_1, graph_2)):
        for source in operand.predecessors("out"):
            operand.edges[source, "out"]["position"] = position
    combined_graph = compose_with_rename(graph_1, graph_2)
    node_id = str(uuid4())
    graph = nx.relabel_nodes(combined_graph, {"out": node_id})
    nx.set_node_attributes(graph, {node_id: f"{operation}_operation"}, "type")

    # Now add the output node and link it to the operation node
    graph.add_node("out", type="out")
    graph.add_edge(node_id, "out")

    return graph


def validate_graph(graph: nx.DiGraph) -> str:
    """Validate an ordered expression tree and return its output kind.

    Graph nodes contain structural metadata only. Addition combines equal kinds;
    multiplication permits at most one photon spectrum. Operators have exactly
    one explicitly scoped operand and declare its input and output kinds.
    """
    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        raise ValueError("A spectral graph must be a directed expression tree.")
    if "out" not in graph or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("A spectral graph must be acyclic and have an 'out' node.")
    kinds, names = {}, set()
    for node_id in nx.topological_sort(graph):
        node = graph.nodes[node_id]
        node_type = node.get("type")
        children = list(graph.predecessors(node_id))
        expected_out = 0 if node_id == "out" else 1
        if graph.out_degree(node_id) != expected_out:
            raise ValueError("Each expression node must have exactly one parent.")
        if node_type == "out":
            if node_id != "out" or len(children) != 1:
                raise ValueError("A spectral graph must have exactly one output expression.")
            kinds[node_id] = kinds[children[0]]
        elif node_type in {"add_operation", "mul_operation"}:
            if len(children) != 2 or {
                graph.edges[child, node_id].get("position") for child in children
            } != {0, 1}:
                raise ValueError("Binary model operations require two ordered operands.")
            left, right = (kinds[child] for child in children)
            if node_type == "add_operation" and left != right:
                raise ValueError("Cannot add emission and dimensionless factors.")
            if node_type == "mul_operation" and left == right == "additive":
                raise ValueError("Cannot multiply two additive models; use '+' to add emission.")
            kinds[node_id] = "additive" if "additive" in (left, right) else "multiplicative"
        elif node_type in {"additive_component", "multiplicative_component", "operator_component"}:
            name = node.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*_[1-9][0-9]*", name):
                raise ValueError(f"Invalid public component name: {name!r}.")
            if name in names:
                raise ValueError(f"Duplicate public component name: {name!r}.")
            names.add(name)
            if node_type == "operator_component":
                input_type, output_type = node.get("input_type"), node.get("output_type")
                if input_type not in {"additive", "multiplicative"} or output_type not in {
                    "additive",
                    "multiplicative",
                }:
                    raise ValueError("Operators must declare valid input and output kinds.")
                if len(children) != 1 or kinds[children[0]] != input_type:
                    raise ValueError(f"Operator {name!r} requires one {input_type} expression.")
                kinds[node_id] = output_type
            else:
                if children:
                    raise ValueError("Emission and factor components cannot have operands.")
                kinds[node_id] = node_type.removesuffix("_component")
        else:
            raise ValueError(f"Unknown spectral graph node type: {node_type!r}.")
    if graph.nodes["out"].get("type") != "out":
        raise ValueError("The 'out' node must have type 'out'.")
    return kinds["out"]


_MERMAID_OP_SYMBOLS = {"add": "**+**", "mul": "**x**"}


def _render_mermaid_node(node, attributes) -> str:
    if attributes["type"] == "out":
        return f'    {node}("Output")\n'

    operation_type, node_type = attributes["type"].split("_")

    if node_type == "component":
        name, number = attributes["name"].rsplit("_", 1)
        return f'    {node}("{name.capitalize()} ({number})")\n'

    if node_type == "operation":
        symbol = _MERMAID_OP_SYMBOLS.get(operation_type)
        if symbol is not None:
            return f"    {node}{{{symbol}}}\n"

    return ""


def export_to_mermaid(graph, file=None):
    mermaid_code = "graph LR\n"  # LR = left to right

    for node, attributes in graph.nodes(data=True):
        mermaid_code += _render_mermaid_node(node, attributes)

    for source, target in graph.edges():
        mermaid_code += f"    {source} --> {target}\n"

    if file is None:
        return mermaid_code
    else:
        with open(file, "w") as f:
            f.write(mermaid_code)
