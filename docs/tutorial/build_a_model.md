With JAXspec, you can easily build a model in the same fashion as you would do using
your favorite spectral fitting library. The following example shows how to build simple
models using additive and multiplicative components.

```python

from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model_simple = Tbabs()*Powerlaw()
```

<!---
```python
try:
    model_simple.export_to_mermaid(file='../docs/runtime/various_model_graphs/model_simple.txt')
except:
    #this is for github actions
    model_simple.export_to_mermaid(file='docs/runtime/various_model_graphs/model_simple.txt')
```
-->

Here is a graphical representation for this

``` mermaid
--8<-- "docs/runtime/various_model_graphs/model_simple.txt"
```

```python
from jaxspec.model.additive import Blackbody, Powerlaw
from jaxspec.model.multiplicative import Tbabs, Phabs
model_complex = Tbabs()*(Powerlaw() + Phabs()*Blackbody()) + Blackbody()
```

<!---
```python
try:
    model_complex.export_to_mermaid(file='../docs/runtime/various_model_graphs/model_complex.txt')
except:
    #this is for github actions
    model_complex.export_to_mermaid(file='docs/runtime/various_model_graphs/model_complex.txt')
```
-->

``` mermaid
--8<-- "docs/runtime/various_model_graphs/model_complex.txt"
```