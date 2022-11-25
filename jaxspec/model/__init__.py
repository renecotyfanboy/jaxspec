import inspect
import sys
from .abc import ModelComponent
from .additive import *
from .multiplicative import *


# I want to put all the existent models in the namespace to build them in the good ol' XSPEC way
# No cherry picking for ModelComponents is required since they are lazily called
# and not built until explicitly called
# Maybe add some safeguard at some point
_modules = {name.lower(): cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)}
