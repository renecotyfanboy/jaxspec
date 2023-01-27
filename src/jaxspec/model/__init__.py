import inspect
import sys
from .abc import ModelComponent, AdditiveComponent, MultiplicativeComponent
from .additive import *
from .multiplicative import *


def all_subclasses(cls):
    return list(set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]))

# I want to put all the existent models in the namespace to build them in the good ol' XSPEC way
_modules = {name.lower(): cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass) if issubclass(cls, ModelComponent)}
_additive = all_subclasses(AdditiveComponent)
_multiplicative = all_subclasses(MultiplicativeComponent)

#Pretty ugly, we should find a prettier way to do this
del _modules['additivecomponent']
del _modules['multiplicativecomponent']
del _modules['modelcomponent']
del _modules['analyticaladditive']