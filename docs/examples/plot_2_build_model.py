"""
# Model building made easy

With JAXspec, you can easily build a model in the same fashion as you would do using
your favorite spectral fitting library. The following example shows how to build simple
models using additive and multiplicative components.
"""
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model_simple = Tbabs()*Powerlaw()
model_simple.export_to_mermaid()

# %% New cell
# Same thing with a more complex model
from jaxspec.model.additive import Blackbody, Powerlaw
from jaxspec.model.multiplicative import Tbabs, Phabs
model_complex = Tbabs()*(Powerlaw() + Phabs()*Blackbody()) + Blackbody()
print(model_complex)
# %% New cell
# This is a ne test