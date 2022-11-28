import inspect
import sys
from .abc import ModelComponent
from .additive import *
from .multiplicative import *


# I want to put all the existent models in the namespace to build them in the good ol' XSPEC way
_modules = {name.lower(): cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass) if issubclass(cls, ModelComponent)}

#Pretty ugly, we should find a prettier way to do this
del _modules['additivecomponent']
del _modules['multiplicativecomponent']
del _modules['modelcomponent']