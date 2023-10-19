from . import model_components
import ast
import haiku as hk
from simpleeval import simple_eval
from collections.abc import Mapping


class _LazyModules(Mapping):
    """
    Class borrowed from https://stackoverflow.com/questions/16669367/setup-dictionary-lazily
    Allows to setup the mapping between a module and a string lazily, i.e dict['model']
    is evaluated when called and not before. This is necessary to build a hk.Module from the
    various model components we require
    """

    def __init__(self, raw_modules, energy):
        self._raw_dict = raw_modules
        self.energy = energy

    def __getitem__(self, key):
        module = self._raw_dict.__getitem__(key)
        return module()(self.energy)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def build_model(model_string="expfac*lorentz"):
    @hk.without_apply_rng
    @hk.transform
    def model_func(energy):
        operators = {ast.Add: lambda x, y: x + y, ast.Mult: lambda x, y: x * y}
        lazy_modules = _LazyModules(model_components, energy)
        return simple_eval(model_string, operators=operators, names=lazy_modules)

    return model_func
