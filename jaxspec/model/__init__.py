import inspect
import sys
from .abc import ModelComponent
from .additive import *
from .multiplicative import *


# I want to put all the existent models in the namespace to build them in the good ol' XSPEC way
_modules = {name.lower(): cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
            if issubclass(cls, ModelComponent)}

# Find this pretty ugly, but didn't fix it yet
del _modules['modelcomponent'], _modules['additivecomponent'], _modules['multiplicativecomponent']