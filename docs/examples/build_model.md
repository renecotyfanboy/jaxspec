# Model building made easy

## Nesting components

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
    b23e9dcc-d80f-41ea-ba67-69cb15a8bd3f{"**x**"}
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
    a7e5876d-e812-4abb-af8b-c5f73c3958fb{"**x**"}
    bdbbcb80-01c5-4ebc-9b1b-0f42984b42b5("Powerlaw (1)")
    bf37c5ee-cccd-4553-aca2-0135d49a8956{"**+**"}
    27c4e284-6238-4dd2-8f2a-6c8fc96a23f6("Phabs (1)")
    c4b385da-040d-475a-a7b2-1137db3a2807{"**x**"}
    36aa2fbf-8713-4e9b-a488-be7910c864c7("Blackbody (1)")
    d357476a-e6c5-4730-b37d-ba872cd1d4cf{"**+**"}
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

## Build a custom component

`jaxspec` enables the build of custom components. This is useful if you want to build a model with a component that is not implemented in `jaxspec`.

### Additive component

In this example, we will first build a component with a known analytical expression. Let's assume we want to model the following function:

$$
\begin{align}
\mathcal{M}_\text{add}( E ) &= K \sin (E/E_0) \exp (-E/E_1)
\end{align}
$$

Using `jaxspec`, this is fairly easy. The only thing required is that every function should be computable using `JAX` primitives. Since `JAX` implements
most of the `numpy` functions and a lot of `scipy` functions (see [here](https://jax.readthedocs.io/en/latest/jax.html)), this should not be a problem in the simplest cases.

```python
import jax.numpy as jnp
import flax.nnx as nnx
from jaxspec.model.abc import AdditiveComponent


class MyComponent(AdditiveComponent):
    def __init__(self):
        self.K = nnx.Param(0.5)
        self.E0 = nnx.Param(1.0)
        self.E1 = nnx.Param(1.0)

    def continuum(self, energy):
        return self.K * jnp.sin(energy / self.E0) * jnp.exp(-energy / self.E1)
```

Let's understand in depth this code snippet. First, we define a class that inherits from [`AdditiveComponent`][jaxspec.model.abc.AdditiveComponent].
This class is an abstract class that defines the interface of an additive component. This interface is composed of two methods: [`continuum`][jaxspec.model.abc.AdditiveComponent.continuum] and [`integrated_continuum`][jaxspec.model.abc.AdditiveComponent.integrated_continuum].
These functions will be called by the model to compute the defined continuum and integrate it, and add the integrated continuum.

To do a quick summary of what is required to build a custom component, we need to:

 1. Inherit from [`AdditiveComponent`][jaxspec.model.abc.AdditiveComponent]
 2. Implement the [`continuum`][jaxspec.model.abc.AdditiveComponent.continuum] method (optional)
 3. Implement the [`integrated_continuum`][jaxspec.model.abc.AdditiveComponent.integrated_continuum] method (optional)
 4. Ensure that the parameters to fit are defined using [`nnx.Param`][flax.nnx.Param]

And that's all. The newly created component is directly combinable with other components, and can be used to build more complex spectral model.

```python
from jaxspec.model.additive import Powerlaw
from jaxspec.model.multiplicative import Tbabs

model = Tbabs() * (Powerlaw() + MyComponent())
```

``` mermaid
graph LR
    f816ddff-ba64-4022-93d2-5d772b97a31c("Tbabs (1)")
    711387a2-7d95-4c1c-af7e-ae263e8fc049{**x**}
    261fdd93-cdb0-46f8-9006-84d9bc83bbcf("Powerlaw (1)")
    af00f991-37be-4598-b3bc-7e67c0e4ff3e{**+**}
    86346cd5-1c46-4fce-9c9c-b4f1d34cbce0("Mycomponent (1)")
    out("Output")
    f816ddff-ba64-4022-93d2-5d772b97a31c --> 711387a2-7d95-4c1c-af7e-ae263e8fc049
    711387a2-7d95-4c1c-af7e-ae263e8fc049 --> out
    261fdd93-cdb0-46f8-9006-84d9bc83bbcf --> af00f991-37be-4598-b3bc-7e67c0e4ff3e
    af00f991-37be-4598-b3bc-7e67c0e4ff3e --> 711387a2-7d95-4c1c-af7e-ae263e8fc049
    86346cd5-1c46-4fce-9c9c-b4f1d34cbce0 --> af00f991-37be-4598-b3bc-7e67c0e4ff3e
```

### Multiplicative component
Let's do the same implementation for a multiplicative component. In this example, we will use the following analytical expression:

$$
\begin{align}
\mathcal{M}_\text{mul}( E ) &= | \cos(E/E_0) |
\end{align}
$$

The same logic applies, you must inherit from the [`MultiplicativeComponent`][jaxspec.model.multiplicative.MultiplicativeComponent] and implement the [`factor`][jaxspec.model.abc.MultiplicativeComponent.factor] method.

```python

from jaxspec.model.abc import MultiplicativeComponent

class MyFactor(MultiplicativeComponent):
    def __init__(self):
        self.E0 = nnx.Param(1.0)

    def factor(self, energy):
        return jnp.abs(jnp.cos(energy / self.E0))
```