# Model building made easy

With `jaxspec`, you can easily build a model in the same fashion as you would do using
your favorite spectral fitting library. The following example shows how to build simple
models using additive and multiplicative components.

```python
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model_simple = Tbabs() * Powerlaw()
```

These lines will build a simple absorbed powerlaw model. It can be
represented with the following graph.

```mermaid
graph LR
    fd38dc8d-0084-4fab-a076-cefc682de13a("Tbabs (1)")
    b23e9dcc-d80f-41ea-ba67-69cb15a8bd3f{"$$\times$$"}
    4a654b57-a412-4b5a-a095-099bfbba245e("Powerlaw (1)")
    out("Output")
    fd38dc8d-0084-4fab-a076-cefc682de13a --> b23e9dcc-d80f-41ea-ba67-69cb15a8bd3f
    b23e9dcc-d80f-41ea-ba67-69cb15a8bd3f --> out
    4a654b57-a412-4b5a-a095-099bfbba245e --> b23e9dcc-d80f-41ea-ba67-69cb15a8bd3f
```

Using the Additive and Multiplicative components defined in `jaxspec`, you can
build arbitrary complex models, in the same fashion as you would do in other
spectral fitting libraries.

```python
from jaxspec.model.additive import Blackbody, Powerlaw
from jaxspec.model.multiplicative import Tbabs, Phabs

model_complex = Tbabs() * (Powerlaw() + Phabs() * Blackbody()) + Blackbody()
```

This build the following model

``` mermaid
graph LR
    42d43fd8-597d-4e3e-9dc7-291e4fcafb5c("Tbabs (1)")
    a7e5876d-e812-4abb-af8b-c5f73c3958fb{"$$\times$$"}
    bdbbcb80-01c5-4ebc-9b1b-0f42984b42b5("Powerlaw (1)")
    bf37c5ee-cccd-4553-aca2-0135d49a8956{"$$+$$"}
    27c4e284-6238-4dd2-8f2a-6c8fc96a23f6("Phabs (1)")
    c4b385da-040d-475a-a7b2-1137db3a2807{"$$\times$$"}
    36aa2fbf-8713-4e9b-a488-be7910c864c7("Blackbody (1)")
    d357476a-e6c5-4730-b37d-ba872cd1d4cf{"$$+$$"}
    0383b40b-0e6d-4bad-878a-0687e7ce2b94("Blackbody (2)")
    out("Output")
    42d43fd8-597d-4e3e-9dc7-291e4fcafb5c --> a7e5876d-e812-4abb-af8b-c5f73c3958fb
    a7e5876d-e812-4abb-af8b-c5f73c3958fb --> d357476a-e6c5-4730-b37d-ba872cd1d4cf
    bdbbcb80-01c5-4ebc-9b1b-0f42984b42b5 --> bf37c5ee-cccd-4553-aca2-0135d49a8956
    bf37c5ee-cccd-4553-aca2-0135d49a8956 --> a7e5876d-e812-4abb-af8b-c5f73c3958fb
    27c4e284-6238-4dd2-8f2a-6c8fc96a23f6 --> c4b385da-040d-475a-a7b2-1137db3a2807
    c4b385da-040d-475a-a7b2-1137db3a2807 --> bf37c5ee-cccd-4553-aca2-0135d49a8956
    36aa2fbf-8713-4e9b-a488-be7910c864c7 --> c4b385da-040d-475a-a7b2-1137db3a2807
    d357476a-e6c5-4730-b37d-ba872cd1d4cf --> out
    0383b40b-0e6d-4bad-878a-0687e7ce2b94 --> d357476a-e6c5-4730-b37d-ba872cd1d4cf
```
