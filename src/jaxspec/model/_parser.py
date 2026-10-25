"""Restricted model-expression parsing; input text is never executed as Python."""

from __future__ import annotations

import ast
import inspect
import math

from collections.abc import Mapping
from dataclasses import dataclass
from keyword import iskeyword

from .abc import ModelComponent


@dataclass(frozen=True)
class _Expression:
    """An inert, validated instruction; no component exists until the full tree passes."""

    kind: str
    operation: str
    operands: tuple[_Expression, ...] = ()
    component: type[ModelComponent] | None = None
    config: dict | None = None


def component_registry(
    components: Mapping[str, type[ModelComponent]] | None = None,
) -> dict[str, type[ModelComponent]]:
    """Return the explicit, case-insensitive allowlist used to read model descriptions.

    ``components`` registers trusted custom component classes. Names may be aliases,
    but cannot replace another class under an existing name. Registration never
    imports a module named by a model expression.
    """
    from . import additive, multiplicative
    from .tabulated import ATable, ETable, MTable

    classes = [getattr(additive, name) for name in additive.__all__]
    classes.extend(
        getattr(multiplicative, name)
        for name in (
            "MultiplicativeConstant",
            "Expfac",
            "Tbabs",
            "zTbabs",
            "Phabs",
            "Wabs",
            "Gabs",
            "Highecut",
            "Zedge",
            "Tbpcf",
            "FDcut",
        )
    )
    classes.extend((ATable, MTable, ETable))
    registry = {cls.__name__.casefold(): cls for cls in classes}
    if components is not None and not isinstance(components, Mapping):
        raise TypeError("components must be a mapping from names to ModelComponent classes.")
    for name, cls in (components or {}).items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            or iskeyword(name)
        ):
            raise ValueError("Component names must be public Python identifiers.")
        if not inspect.isclass(cls) or not issubclass(cls, ModelComponent):
            raise TypeError(f"Registered component {name!r} must be a ModelComponent class.")
        key = name.casefold()
        if key in registry and registry[key] is not cls:
            raise ValueError(f"Component name {name!r} is already registered.")
        registry[key] = cls
    return registry


def _literal_config(node: ast.AST):
    """Read finite primitive constructor values without evaluating any expression."""
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        if isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float):
            value = _literal_config(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
    elif isinstance(node, ast.List | ast.Tuple):
        values = [_literal_config(child) for child in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    elif isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(
                    "Constructor configuration dictionaries require literal string keys."
                )
            if key.value in result:
                raise ValueError(f"Duplicate constructor configuration key {key.value!r}.")
            result[key.value] = _literal_config(value)
        return result
    raise ValueError("Constructor configuration must contain finite literal values only.")
