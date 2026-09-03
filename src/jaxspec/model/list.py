import inspect

from .abc import AdditiveComponent, ModelComponent, MultiplicativeComponent

# Imported for their side effect: defining the classes is what registers them as
# subclasses for the sweep below. ``additive`` declares ``__all__``, so the base
# classes above no longer come through the star import.
from .additive import *  # noqa: F403
from .multiplicative import *  # noqa: F403


def all_models(cls: ModelComponent) -> list[ModelComponent]:
    """
    Return a list of all the subclasses of a given ModelComponent class
    """
    subclasses = list(
        set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_models(c)])
    )

    return [s for s in subclasses if not inspect.isabstract(s)]


model_components = {cls.__name__: cls for cls in all_models(ModelComponent)}
additive_components = {cls.__name__: cls for cls in all_models(AdditiveComponent)}
multiplicative_components = {cls.__name__: cls for cls in all_models(MultiplicativeComponent)}
