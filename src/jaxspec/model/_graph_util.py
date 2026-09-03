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

    # Renaming must not mutate the right-hand model's graph.
    graph_2 = graph_2.copy()

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

    combined_graph = compose_with_rename(graph_1, graph_2)
    node_id = str(uuid4())
    graph = nx.relabel_nodes(combined_graph, {"out": node_id})
    nx.set_node_attributes(graph, {node_id: f"{operation}_operation"}, "type")

    # Now add the output node and link it to the operation node
    graph.add_node("out", type="out")
    graph.add_edge(node_id, "out")

    return graph


_MERMAID_OP_SYMBOLS = {"add": "**+**", "mul": "**x**"}


def _render_mermaid_node(node, attributes) -> str:
    if attributes["type"] == "out":
        return f'    {node}("Output")\n'

    operation_type, node_type = attributes["type"].split("_")

    if node_type == "component":
        name, number = attributes["name"].split("_")
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
