import inspect
from .abc import ModelComponent
from .additive import *  # noqa: F403
from .multiplicative import *  # noqa: F403


def all_models(cls: ModelComponent) -> list[ModelComponent]:
    """
    Return a list of all the subclasses of a given ModelComponent class
    """
    subclasses = list(set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_models(c)]))

    return [s for s in subclasses if not inspect.isabstract(s)]


# I want to put all the existent models in the namespace to build them in the good ol' XSPEC way
model_components = {cls.__name__: cls for cls in all_models(ModelComponent)}
additive_components = {
    cls.__name__: cls
    for cls in all_models(AdditiveComponent)  # noqa: F405
}
multiplicative_components = {
    cls.__name__: cls
    for cls in all_models(MultiplicativeComponent)  # noqa: F405
}
