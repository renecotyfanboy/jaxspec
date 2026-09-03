# Flexible prior setting

Most fits get by with a flat prior dict mapping every model parameter to a
single distribution. Once you have several observations, heterogeneous
instrument calibration, or correlated parameters, that flat dict starts
fighting you. This page walks through the richer prior-specification surface
`jaxspec` exposes: per-observation scoping baked into the dict key syntax,
tied parameters across observations, and a callable form for the cases the
dict can't express on its own.

Everything below works with any of `MCMCFitter`, `VIFitter`, or `NSFitter`
— the `prior=` argument is normalised the same way before sampling.

## The prior dict: three key shapes

The prior dict accepts three key forms, distinguished by the optional
bracketed suffix:

| Key form | Semantics | Numpyro sites |
|---|---|---|
| `"prefix.path"` (bare) | Shared across every applicable observation: one draw, reused by identity for each per-obs replica. | 1 |
| `"prefix.path[*]"` | Independent draw per applicable observation. | N |
| `"prefix.path[obs_name]"` | Single draw scoped to that observation. Use multiple `[obs_name]` entries for heterogeneous priors or for parameters that exist only on some observations. | 1 each |

A typical multi-observation fit mixes all three:

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

Bare keys are the most common: they reproduce the familiar flat-prior
behaviour. The bracketed suffix only enters when you want per-observation
scoping. In the example above, the first four spectral keys and `powerlaw_2.norm` are
shared, the `[*]` entries are per-observation, and `powerlaw_2.alpha` is tied
to `powerlaw_1.alpha`.

## Applicable observations per prefix

Which observations a key applies to depends on the prefix:

- `spectrum.` &mdash; every observation in `observations`.
- `instrument.` &mdash; observations with a non-`None` `InstrumentModel`.
- `background.` &mdash; observations with a non-`None` `BackgroundModel`
  (after singleton expansion).

The wildcard `[*]` expands to exactly this set, and an explicit `[obs_name]`
must name a member of it. This is what lets you keep one observation as an
implicit "reference" for instrument calibration:

```python
instrument_model = {
    "PN":   None,                                                       # reference: no calibration applied
    "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
    "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
}
```

With this dict, `"instrument.gain.factor[*]"` samples MOS1 and MOS2 only;
PN's flux is folded through the identity. Omitting `"PN"` from the dict
entirely has the same effect.

## Per-observation model dicts

`instrument_model` and `background_model` both accept a `{obs_name: model | None}`
dict — the recipe for heterogeneous instrument or background treatments.
`background_model` additionally accepts a singleton `BackgroundModel`, which
is internally cloned per observation.

A heterogeneous setup might apply pileup correction on PN only, and a
spectral background model on MOS1 with a default Gamma-per-bin background
elsewhere:

```python
instrument_model = {
    "PN":   InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
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
the raw nnx leaf path for a background parameter would be
`background.<obs>.spectral_model.powerlaw_1.alpha`. That's noisy. The model
exposes a `user_path` hook that strips its own wrapper segment, letting you
write the shorter:

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

`BackgroundWithError` doesn't need a prior entry at all: its
`default_prior(observation, obs_name)` injects a per-bin `Gamma(observed + 1, 1)`
prior automatically. User entries override the default if present.

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
# Replaces the shared `instrument.gain.factor[*]` entry above — the scopes must stay
# disjoint, so drop it first.
del prior["instrument.gain.factor[*]"]
prior["instrument.gain.factor[MOS1]"] = dist.Uniform(0.5, 1.5)
prior["instrument.gain.factor[MOS2]"] = TiedParameter("instrument.gain.factor[MOS1]", lambda x: x)
```

2. Element-wise across observations with `[*]`: each per-obs draw of
   `blackbodyrad_1.norm` is 2x the corresponding per-obs `powerlaw_1.norm`.

```python
del prior["spectrum.blackbodyrad_1.norm"]  # replaced by the per-obs tie below
prior["spectrum.blackbodyrad_1.norm[*]"] = TiedParameter("spectrum.powerlaw_1.norm[*]", lambda x: 2.0 * x)
```

3. Cross-prefix: a background powerlaw normalisation tied to the source.

```python
prior["background.powerlaw_1.alpha[MOS1]"] = TiedParameter("spectrum.powerlaw_1.alpha", lambda x: x)
```

Resolved ties register as `numpyro.deterministic` sites, so they appear in
`result.inference_data.posterior` alongside the sampled sites and you can read
their posteriors directly. They are deliberately **excluded** from `to_chain`,
and therefore from `plot_corner` and `table`: a tied parameter is a deterministic
function of its source, so plotting it as a free dimension would be misleading.

## Callable priors: when the dict isn't enough

The dict form covers most use cases, but a few things sit outside it:

- A prior structure that's defined programmatically (e.g. "apply the same
  prior to every parameter ending in `.alpha`").
- Drawing multiple parameters jointly from a multivariate distribution.
- Hierarchical priors where a hyper-parameter is itself sampled.

For these, pass a callable as `prior=` instead of a dict. Two shapes are
auto-detected by argument count:

- **Leaf callable** `(path: str, shape: tuple) -> Distribution` &mdash;
  invoked once per nnx leaf. Use when no shared / hierarchical / joint
  setup is needed.
- **Factory callable** `() -> leaf_callable` &mdash; invoked once at the
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
the parameter suffix or the observation segment.

### Example 2: joint / covariant priors via `joint_prior_factory`

`jaxspec` ships [`joint_prior_factory`][jaxspec.fit.joint_prior_factory] for
multivariate draws. It samples one multivariate site once and returns a per-leaf
lookup of the already-drawn components, which you can chain with structural
defaults:

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxspec.fit import joint_prior_factory

def correlated_spectral_prior():
    # One MVN draw shared across every per-obs spectrum replica.
    alpha_norm = joint_prior_factory(
        components=("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
        joint_dist=dist.MultivariateNormal(
            loc=jnp.array([2.0, 1e-4]),
            covariance_matrix=jnp.array([[0.5, 1e-5], [1e-5, 1e-8]]),
        ),
        name="spectrum.powerlaw_1.alpha_norm",
    )
    def prior(path, shape):
        d = alpha_norm(path)
        return d if d is not None else structural_prior(path, shape)
    return prior

fitter = MCMCFitter(spectral_model, correlated_spectral_prior, observations)
```

For per-instrument joint draws (e.g. `(gain, shift)` correlated within each
instrument but independent across instruments), call the factory once per
observation inside the outer factory and chain the lookups.

The reason this needs the callable form is that the per-leaf prior contract
returns one `Distribution` per leaf. Multivariate sampling has to happen once,
outside that loop; each leaf then receives its already-sampled component as a
plain array, which the binder writes straight through without creating a second
site. `joint_prior_factory` is just a convenient packaging of that pattern.

### Example 3: hierarchical / partial-pooling prior

Hierarchical priors are a textbook factory-callable use case: sample the
hyper-parameters first, then return a leaf callable that consumes them:

```python
def hierarchical_prior():
    # Hyper-prior on the per-instrument gain mean and scale.
    mu    = numpyro.sample("hyper.gain.mu",    dist.Normal(1.0, 0.1))
    sigma = numpyro.sample("hyper.gain.sigma", dist.HalfNormal(0.1))
    def prior(path, shape):
        if path.endswith(".gain.factor"):
            return dist.Normal(mu, sigma)
        return structural_prior(path, shape)
    return prior

fitter = MCMCFitter(spectral_model, hierarchical_prior, observations,
                    instrument_model=instrument_model)
```

Each per-instrument gain is now drawn from a common
`Normal(mu, sigma)` where `mu` and `sigma` are themselves estimated from
the data.

## Hybrid: dict + callable via `dict_prior`

When you want the dict form's convenience for the easy parameters and a
custom callable for one or two special ones,
[`dict_prior`][jaxspec.fit.dict_prior] packages a dict as a leaf callable
that returns `None` on miss &mdash; perfect for chaining:

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

`dict_prior` reads the same key syntax as the framework and pre-samples shared entries
on first call, so the trace contains exactly one numpyro site per shared key.

Keep the scopes **disjoint**: a parameter should be covered by a shared key, by `[*]`,
or by an explicit `[obs]` key, never by two at once.

!!! warning "`dict_prior` is not validated"

    A *framework* prior dict is checked when the `BayesianModel` is constructed, and
    overlapping keys raise a `ValueError` naming both. A dict handed to `dict_prior`
    gets no such check — passing a callable skips validation entirely. On an overlap
    the lookup silently applies the precedence `[obs]` > `[*]` > shared, and the losing
    shared key has *already been sampled*, leaving a free site in the trace that
    influences nothing.

## Site names and posterior inspection

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

## Failure modes

A few common mistakes surface as build-time or sample-time errors so you
don't silently fit the wrong model:

!!! warning "Loud failures, by design"

    - **Missing prior for a leaf** raises `KeyError` at sample time
      &mdash; there's no silent fallback to the model's default `nnx.Param`
      value. Every nnx leaf must be covered by some entry (or default like
      `BackgroundWithError`'s).
    - **Unknown observation in `[obs]`** is caught at `BayesianModel`
      construction with a `ValueError` listing the applicable set.
    - **`[obs]` referencing an observation without an applicable model**
      (e.g. `"instrument.gain.factor[PN]"` when PN's `instrument_model`
      entry is `None`) errors at construction.
    - **`TiedParameter` source not found** is reported at sample time
      with the resolved `(path, obs)` lookup key, so it's clear which
      scope was expected.
