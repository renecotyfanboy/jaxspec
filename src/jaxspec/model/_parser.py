"""Restricted model-expression parsing; input text is never executed as Python."""

from __future__ import annotations

import ast
import inspect
import math

from collections.abc import Mapping
from dataclasses import dataclass
from keyword import iskeyword

from .abc import ModelComponent, OperatorComponent


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


def _validate_expression(node: ast.AST, registry: dict[str, type[ModelComponent]]) -> _Expression:
    """Resolve names, constructor signatures and spectral types before instantiation."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mult):
        left = _validate_expression(node.left, registry)
        right = _validate_expression(node.right, registry)
        if isinstance(node.op, ast.Add):
            if left.kind != right.kind:
                raise ValueError("Cannot add an emission spectrum and a dimensionless factor.")
            return _Expression(left.kind, "add", (left, right))
        if left.kind == right.kind == "additive":
            raise ValueError("Cannot multiply two additive models; use '+' to add emission.")
        kind = "additive" if "additive" in (left.kind, right.kind) else "multiplicative"
        return _Expression(kind, "mul", (left, right))

    if isinstance(node, ast.Name):
        name, arguments, keywords = node.id, [], []
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name, arguments, keywords = node.func.id, node.args, node.keywords
    else:
        raise ValueError(
            "Model expressions support component names, calls, parentheses, '+' and '*' only."
        )

    cls = registry.get(name.casefold())
    if cls is None:
        raise ValueError(f"Unknown model component {name!r}.")
    if inspect.isabstract(cls):
        raise ValueError(f"Model component {name!r} is abstract.")
    kind = getattr(cls, "type", None)
    if kind not in {"additive", "multiplicative", "operator"}:
        raise ValueError(f"Model component {name!r} has an unsupported spectral type.")

    config = {}
    for keyword in keywords:
        if keyword.arg is None or keyword.arg.startswith("_"):
            raise ValueError("Constructor arguments must be explicit public keywords.")
        if keyword.arg in config:
            raise ValueError(f"Duplicate constructor keyword {keyword.arg!r}.")
        config[keyword.arg] = _literal_config(keyword.value)
    try:
        if cls.__init__ is object.__init__ and config:
            raise TypeError("this component does not accept constructor arguments")
        inspect.signature(cls.__init__).bind(None, **config)
    except TypeError as exc:
        raise ValueError(f"Invalid constructor arguments for {name!r}: {exc}") from exc

    if kind == "operator":
        if not issubclass(cls, OperatorComponent):
            raise ValueError(f"Operator {name!r} must inherit from OperatorComponent.")
        if len(arguments) != 1:
            raise ValueError(
                f"Operator {name!r} requires one explicit scope, e.g. {name}(powerlaw)."
            )
        scope = _validate_expression(arguments[0], registry)
        input_type, output_type = cls.input_type, cls.output_type
        if input_type not in {"additive", "multiplicative"} or output_type not in {
            "additive",
            "multiplicative",
        }:
            raise ValueError(f"Operator {name!r} declares an invalid input or output type.")
        if scope.kind != input_type:
            raise ValueError(f"Operator {name!r} requires a {input_type} subexpression.")
        return _Expression(output_type, "apply", (scope,), cls, config)
    if arguments:
        raise ValueError(f"Component {name!r} accepts keyword constructor configuration only.")
    return _Expression(kind, "component", component=cls, config=config)
