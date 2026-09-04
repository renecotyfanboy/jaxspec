# Flexible model setting

Most fits get by with a flat prior dict mapping every model parameter to a
single distribution. Once you have several observations, heterogeneous
instrument calibration, or correlated parameters, you might need more flexible
settings. This page walks through the richer prior-specification surface
`jaxspec` exposes: per-observation scoping baked into the dict key syntax,
tied parameters across observations, and a callable form for the cases the
dict can't express on its own.

## Splitting parameters for different observations

The prior dict accepts three key forms, distinguished by the optional
bracketed suffix as follows:

| Key form                  | Effect on the model                                                                                                                                        | Free parameters |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| `"prefix.path"` (vanilla) | The prior is used to draw a single parameter that is shared across all observations                                                                        | 1               |
| `"prefix.path[*]"`        | The same prior is used to draw multiple independent parameters for each observation                                                                        | N               |
| `"prefix.path[obs_name]"` | Single draw scoped to that observation. Use multiple `[obs_name]` entries for heterogeneous priors or for parameters that exist only on some observations. | 1 each          |

Here is a multi-observation fit that mixes all three, with independent parameters for some spectral model components
and instrument calibrations:

```python
import numpyro
import numpyro.distributions as dist
from jaxspec.fit import MCMCFitter, TiedParameter
from jaxspec.model.additive import Powerlaw, Blackbodyrad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.background import BackgroundWithError, SpectralModelBackground
from jaxspec.model.instrument import InstrumentModel, ConstantGain, ConstantShift

spectral_model = Tbabs() * (Powerlaw() + Powerlaw() + Blackbodyrad())

prior = {
    "spectrum.tbabs_1.nh":          dist.Uniform(0, 1),
    "spectrum.powerlaw_1.alpha":    dist.Uniform(0, 5),
    "spectrum.blackbodyrad_1.kT":   dist.Uniform(0, 5),
    "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
    "spectrum.powerlaw_1.norm[*]":  dist.LogUniform(1e-5, 1e-2),
    "spectrum.powerlaw_2.norm":     dist.LogUniform(1e-5, 1e-2),
    "instrument.gain.factor[*]":    dist.Uniform(0.5, 1.5),
    "instrument.shift.offset[*]":   dist.Uniform(-0.3, 0.3),
    "spectrum.powerlaw_2.alpha":    TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 0.5 * x),
}
```

The bracketed suffix only enters when you want per-observation
scoping. In the example above, the first four spectral parameters and `powerlaw_2.norm` are
shared, the `[*]` entries are per-observation, and `powerlaw_2.alpha` is tied
to `powerlaw_1.alpha`.

## Parameter names for posterior inspection

Sampled values land in `result.inference_data.posterior` under predictable
names:

| Prior key | Numpyro site |
|---|---|
| `"spectrum.powerlaw_1.alpha"` | `"spectrum.powerlaw_1.alpha"` |
| `"spectrum.powerlaw_1.norm[*]"` | `"forward.spectrum.<obs>.powerlaw_1.norm"` (one per obs) |
| `"instrument.gain.factor[MOS1]"` | `"forward.instrument.MOS1.gain.factor"` |
| `"background.countrate[PN]"` (auto from `BackgroundWithError`) | `"forward.background.PN.countrate"` |

The `"forward."` prefix on per-observation sites is just the conventional
scope under which the `ForwardModel`'s leaves are registered. Once you know
the convention, indexing the posterior is direct:

```python
mos1_gain = result.inference_data.posterior["forward.instrument.MOS1.gain.factor"]
mos2_gain = result.inference_data.posterior["forward.instrument.MOS2.gain.factor"]
```


## Example with `InstrumentModel`

A common calibration pattern is the following:

```python
instrument_model = {
    "PN":   None, # reference: no calibration applied
    "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
    "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
}
```

With this, a prior dict with `"instrument.gain.factor[*]"` samples MOS1 and MOS2 only,
PN flux is folded through the identity. Omitting `"PN"` from the dict
entirely has the same effect.

## Per-observation model dicts

`instrument_model` and `background_model` both accept a `{obs_name: model}`
dict, which allows you to specify different models for different observations.

A heterogeneous setup might apply pileup correction on PN only, and a
spectral background model on MOS1 with a per-bin background elsewhere:

```python
from jaxspec.model.instrument import PileupModel

instrument_model = {
    # PileupModel adds two parameters, alpha and psf_frac, to cover in the prior
    "PN":   PileupModel(gain=ConstantGain(), shift=ConstantShift(),
                        frame_time=73.4e-3, frac_expo=1.0),
    "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
    "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
}

background_model = {
    "PN":   BackgroundWithError(),
    "MOS1": SpectralModelBackground(Tbabs() * Powerlaw()),
    "MOS2": BackgroundWithError(),
}
```

The [`calibration`](calibration.ipynb) notebook walks through the
calibration-specific case end to end.

## Background models and `user_path`

`SpectralModelBackground(spec)` stores `spec` at `self.spectral_model`, so
the canonical path for a background parameter should be
`background.<obs>.spectral_model.powerlaw_1.alpha`, which is verbose.
The model exposes a `user_path` hook that strips its own wrapper segment,
letting you write the shorter:

```python
prior = {
    **prior,
    "background.tbabs_1.nh":       dist.Uniform(0, 1),
    "background.powerlaw_1.alpha": dist.Uniform(0, 5),
    "background.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
}
```

Custom `BackgroundModel` subclasses can override
`user_path(self, nnx_path: str) -> str` to expose their own user-facing
paths. The default is the identity.

!!! note "Only the dict form uses `user_path`"

    Callable priors (see below) receive the full path, e.g.
    `background.MOS1.spectral_model.powerlaw_1.alpha`. In particular, a
    `dict_prior` keyed with the short `background.powerlaw_1.alpha` form will
    not match a background parameter. Match on the path suffix instead.

## Tied parameters

`TiedParameter(tied_to, func)` makes one parameter a deterministic function
of another. The basic shared-to-shared tie is the common case:

```python
prior["spectrum.powerlaw_2.alpha"] = TiedParameter(
    "spectrum.powerlaw_1.alpha", lambda x: 0.5 * x
)
```

The `tied_to` key understands the same `[obs]` / `[*]` syntax as ordinary
prior keys, which unlocks three more patterns:

1. Specific-obs to specific-obs: MOS2's gain mirrors MOS1's draw.
  ```python
  prior["instrument.gain.factor[MOS1]"] = dist.Uniform(0.5, 1.5)
  prior["instrument.gain.factor[MOS2]"] = TiedParameter("instrument.gain.factor[MOS1]", lambda x: x)
  ```
2. Element-wise across observations with `[*]`: each per-obs draw of
   `blackbodyrad_1.norm` is 2x the corresponding per-obs `powerlaw_1.norm`.
  ```python
  prior["spectrum.blackbodyrad_1.norm[*]"] = TiedParameter("spectrum.powerlaw_1.norm[*]", lambda x: 2.0 * x)
  ```
3. Cross-prefix: a background powerlaw normalisation tied to the source.
  ```python
  prior["background.powerlaw_1.alpha[MOS1]"] = TiedParameter("spectrum.powerlaw_1.alpha", lambda x: x)
  ```

Resolved ties register as `numpyro.deterministic` sites, so they appear in
`result.inference_data.posterior` alongside the sampled sites and you can read
their posteriors directly. They are by default **excluded** from `to_chain`,
and therefore from `plot_corner` and `table`.

## Callable priors

The dict form covers most use cases, but a few things sit outside it:

- A prior structure that's defined programmatically (e.g. "apply the same
  prior to every parameter ending in `.alpha`").
- Drawing multiple parameters jointly from a multivariate distribution.
- Hierarchical priors where hyperparameters are themselves sampled.

For these, pass a callable as `prior=` instead of a dict. Two patterns are available
and automatically detected by argument count in the callable:

- **Leaf callable** `prior(path: str, shape: tuple) -> Distribution`
  invoked once per nnx leaf, close to the internal working of the prior dispatch.
- **Factory callable** `prior() -> leaf_callable` invoked once at the
  top of `numpyro_model` (inside the trace), so it can call
  `numpyro.sample` first to draw shared / joint / hierarchical parameters,
  then return the per-leaf lookup function.

!!! warning "A leaf callable draws *per observation*"

    The dict form and the callable form are **not** the same code path. A dict entry
    with a bare key emits one site shared across observations; a leaf callable is
    invoked once per per-observation parameter, so the equivalent-looking callable
    gives each observation its **own** draw and names every site
    `forward.<prefix>.<obs>.<path>`. Replacing a shared dict with a leaf callable
    silently increases the number of free parameters. To share a value from a
    callable, sample it once in a *factory* and return that value from the leaf
    lookup.

### Example 1: structural prior

When many parameters share the same prior shape, a leaf callable saves
typing:

```python
import numpyro.distributions as dist
import jax.numpy as jnp

def structural_prior(path, shape):
    if path.endswith(".nh"):           return dist.Uniform(0, 1)
    if path.endswith(".alpha"):        return dist.Uniform(0, 5)
    if path.endswith(".norm"):         return dist.LogUniform(1e-5, 1e-2)
    if path.endswith(".kT"):           return dist.Uniform(0, 5)
    if path.endswith(".gain.factor"):  return dist.Uniform(0.5, 1.5)
    if path.endswith(".shift.offset"): return dist.Uniform(-0.3, 0.3)
    if path.endswith(".countrate"):    return dist.Gamma(jnp.ones(shape), rate=1.0)
    raise KeyError(f"No prior defined for {path}")

fitter = MCMCFitter(spectral_model, structural_prior, observations)
```

The callable receives the post-replication nnx leaf path
(e.g. `"spectrum.MOS1.powerlaw_1.alpha"`), so you can dispatch on either
the parameter suffix or the observation segment. Remember the warning above:
every parameter here, including `nh` and `kT`, gets one draw per observation.

### Example 2: reuse a prior dict inside a callable via `dict_prior`

Most of the time you only need the callable form for one or two special
parameters and would rather keep the dict syntax for everything else.
[`dict_prior`][jaxspec.fit.dict_prior] packages a dict as a leaf callable
that returns `None` on miss:

```python
from jaxspec.fit import dict_prior

def hybrid_prior():
    covered = dict_prior({
        "spectrum.tbabs_1.nh":          dist.Uniform(0, 1),
        "spectrum.powerlaw_1.alpha":    dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm[*]":  dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT":   dist.Uniform(0, 5),
    })
    def prior(path, shape):
        # Try the dict first; fall back to your custom logic for the rest.
        # Test against None explicitly: a pre-sampled value is a traced array, and
        # `or` would force `bool()` on it (TracerBoolConversionError).
        d = covered(path, shape)
        return d if d is not None else structural_prior(path, shape)
    return prior
```

`dict_prior` does not accept `TiedParameter`. In a factory, sample the source
yourself, compute the tied value, and return both from the leaf callable. Wrapping
the tied value in `numpyro.deterministic` keeps it visible in the posterior:

```python
def tied_prior():
    alpha_1 = numpyro.sample("spectrum.powerlaw_1.alpha", dist.Uniform(0, 5))
    alpha_2 = numpyro.deterministic("spectrum.powerlaw_2.alpha", 0.5 * alpha_1)
    covered = dict_prior({
        "spectrum.tbabs_1.nh":          dist.Uniform(0, 1),
        "spectrum.blackbodyrad_1.kT":   dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.powerlaw_1.norm[*]":  dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_2.norm":     dist.LogUniform(1e-5, 1e-2),
    })
    def prior(path, shape):
        if path.endswith(".powerlaw_1.alpha"):
            return alpha_1
        if path.endswith(".powerlaw_2.alpha"):
            return alpha_2
        return covered(path, shape)
    return prior
```

Using the same site names as the dict form keeps the posterior identical to a
`TiedParameter` fit.

!!! warning "`dict_prior` is not validated"

    A *framework* prior dict is checked when the `BayesianModel` is constructed, and
    overlapping keys raise a `ValueError` naming both. A dict handed to `dict_prior`
    gets no such check &mdash; passing a callable skips validation entirely. On an overlap
    the lookup silently applies the precedence `[obs]` > `[*]` > shared, and the losing
    shared key has *already been sampled*, leaving a free site in the trace that
    influences nothing.

### Example 3: covariant priors via `joint_prior_factory`

`jaxspec` ships [`joint_prior_factory`][jaxspec.fit.joint_prior_factory] for
multivariate draws. It samples one multivariate site once and returns a per-leaf
lookup of the already-drawn components, which you can chain with a `dict_prior`
for the remaining parameters:

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxspec.fit import dict_prior, joint_prior_factory

def correlated_spectral_prior():
    # One Multivariate Normal draw shared across every per-obs spectrum replica.
    alpha_norm = joint_prior_factory(
        components=("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
        joint_dist=dist.MultivariateNormal(
            loc=jnp.array([2.0, 1e-4]),
            covariance_matrix=jnp.array([[0.5, 1e-5], [1e-5, 1e-8]]),
        ),
        name="spectrum.powerlaw_1.alpha_norm",
    )
    # Ordinary dict for everything else, shared across observations.
    rest = dict_prior({
        "spectrum.tbabs_1.nh":          dist.Uniform(0, 1),
        "spectrum.blackbodyrad_1.kT":   dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "spectrum.powerlaw_2.alpha":    dist.Uniform(0, 5),
        "spectrum.powerlaw_2.norm":     dist.LogUniform(1e-5, 1e-2),
    })
    def prior(path, shape):
        d = alpha_norm(path)
        return d if d is not None else rest(path, shape)
    return prior

fitter = MCMCFitter(spectral_model, correlated_spectral_prior, observations)
```

### Example 4: hierarchical prior

Hierarchical priors are a textbook factory-callable use case: sample the
hyper-parameters first, then return a leaf callable that consumes them:

```python
def hierarchical_prior():
    # Hyper-prior on the per-instrument gain mean and scale.
    mu    = numpyro.sample("hyper.gain.mu",    dist.Normal(1.0, 0.1))
    sigma = numpyro.sample("hyper.gain.sigma", dist.HalfNormal(0.1))
    rest = dict_prior({
        "spectrum.tbabs_1.nh":          dist.Uniform(0, 1),
        "spectrum.powerlaw_1.alpha":    dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm[*]":  dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_2.alpha":    dist.Uniform(0, 5),
        "spectrum.powerlaw_2.norm":     dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.kT":   dist.Uniform(0, 5),
        "spectrum.blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
        "instrument.shift.offset[*]":   dist.Uniform(-0.3, 0.3),
    })
    def prior(path, shape):
        if path.endswith(".gain.factor"):
            return dist.Normal(mu, sigma)
        return rest(path, shape)
    return prior

fitter = MCMCFitter(spectral_model, hierarchical_prior, observations,
                    instrument_model=instrument_model)
```

Each per-instrument gain is now drawn from a common
`Normal(mu, sigma)` where `mu` and `sigma` are themselves estimated from
the data.
