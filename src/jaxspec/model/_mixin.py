from __future__ import annotations

from jax.typing import ArrayLike


class HideUnderscore:
    """Hide underscore-prefixed attributes from ``nnx.display`` and ``repr``.

    Flax NNX shows any attribute classified as pytree data even when its name
    starts with ``_`` (only *static* underscore attributes are hidden by
    default). For jaxspec models we want the ``_name`` convention to mean
    "implementation detail, don't show" regardless of whether the value is an
    array / ``Variable``.

    Place this mixin **before** ``Module`` in the MRO.
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


class HasDerivedQuantities:
    """Extension contract for quantities derived from a model object's parameters."""

    def derived_quantities(self) -> dict[str, ArrayLike]:
        """Quantities to publish in the posterior, as ``{name: value}``.

        Names are non-empty and dot-free; values are traceable. Return values only —
        the fitting layer names them and registers the NumPyro sites.
        """
        return {}


class Composable:
    """Define the operations between model components and spectral models."""

    def sanitize_inputs(self, other):
        # Import lazily so ``abc`` can re-export this mixin without a module cycle.
        from .abc import ModelComponent, SpectralModel

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
        return model_1.compose(model_2, operation="add")

    def __mul__(self, other):
        model_1, model_2 = self.sanitize_inputs(other)
        return model_1.compose(model_2, operation="mul")
