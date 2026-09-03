"""Derived quantities declared by model components.

A component may publish quantities derived from its parameters — `Diskbb`'s XSPEC
normalization is the shipped example. The forward model collects them and
`BayesianModel` registers one `numpyro.deterministic` site per draw.

Covered here: the hook, the numpyro-free collection, shared versus per-observation
naming, and that published values come from the sampled parameters.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from flax import nnx
from helpers import dict_of_obsconf

from jaxspec.fit import BayesianModel
from jaxspec.model._additive.disk import STANDARD_RADIAL_EXPONENT, _disk_band_photon_flux
from jaxspec.model.additive import Blackbodyrad, Diskbb, Powerlaw
from jaxspec.model.background import BackgroundModel
from jaxspec.model.instrument import (
    ConstantGain,
    ConstantShift,
    InstrumentModel,
)
from jaxspec.model.multiplicative import Tbabs

BAND = (0.5, 10.0)


def _leaf_values(forward_model, overrides=None):
    """Every nnx parameter leaf path mapped to a concrete value, for `bind_inputs`."""
    _, params, _ = nnx.split(forward_model, nnx.Param, nnx.Not(nnx.Param))
    flat: dict[str, float] = {}

    def walk(node, prefix=""):
        for key, value in node.items():
            path = f"{prefix}{key}"
            walk(value, f"{path}.") if isinstance(value, dict) else flat.setdefault(path, value)

    walk(nnx.to_pure_dict(params))
    for path in flat:
        for suffix, value in (overrides or {}).items():
            if path.endswith(suffix):
                flat[path] = value
    return {path: jnp.asarray(float(value)) for path, value in flat.items()}


@pytest.fixture(scope="module")
def obsconfs():
    return dict_of_obsconf


# --- The hook itself -------------------------------------------------------------


def test_components_publish_nothing_by_default():
    """An ordinary component adds no site, so existing models keep their exact posterior."""
    assert Powerlaw().derived_quantities() == {}
    assert Tbabs().derived_quantities() == {}
    assert ConstantGain().derived_quantities() == {}
    assert ConstantShift().derived_quantities() == {}
    assert InstrumentModel().derived_quantities() == {}


# --- Collection, without numpyro -------------------------------------------------


def test_forward_model_collects_one_entry_per_replica(obsconfs):
    from jaxspec.fit._forward_model import ForwardModel

    forward = ForwardModel(Tbabs() * (Diskbb() + Powerlaw()), obsconfs)
    collected = forward.derived_quantities(_leaf_values(forward))

    assert {(item.prefix, item.observation, item.owner_path, item.name) for item in collected} == {
        ("spectrum", obs, "diskbb_1", "norm_xspec") for obs in obsconfs
    }


def test_forward_model_collects_nothing_without_a_publishing_component(obsconfs):
    """No publishing component means no new posterior variable."""
    from jaxspec.fit._forward_model import ForwardModel

    forward = ForwardModel(Tbabs() * (Powerlaw() + Blackbodyrad()), obsconfs)

    assert forward.derived_quantities(_leaf_values(forward)) == ()


def test_collection_uses_the_bound_values_not_the_defaults(obsconfs):
    """Hooks receive values bound for the current draw rather than module defaults."""
    from jaxspec.fit._forward_model import ForwardModel

    forward = ForwardModel(Tbabs() * Diskbb(), obsconfs)
    inputs = _leaf_values(forward, overrides={"diskbb_1.Tin": 1.7, "diskbb_1.norm": 2.5e-3})

    collected = forward.derived_quantities(inputs)
    expected = 2.5e-3 / float(_disk_band_photon_flux(1.7, STANDARD_RADIAL_EXPONENT, *BAND))

    assert all(np.isclose(float(item.value), expected, rtol=1e-10) for item in collected)
    default = 1e-4 / float(_disk_band_photon_flux(1.0, STANDARD_RADIAL_EXPONENT, *BAND))
    assert not np.isclose(expected, default)


# --- Site naming -----------------------------------------------------------------


def _disk_prior(norm_key="spectrum.diskbb_1.norm"):
    return {
        "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
        "spectrum.diskbb_1.Tin": dist.Uniform(0.1, 5.0),
        norm_key: dist.LogUniform(1e-6, 1e-2),
    }


def test_shared_component_registers_a_single_bare_site(obsconfs):
    """All parameters shared means one value for the whole fit, hence one site."""
    model = BayesianModel(Tbabs() * Diskbb(), _disk_prior(), obsconfs)

    sites = {key for key in model.prior_samples(num_samples=1) if key.startswith("derived.")}

    assert sites == {"derived.spectrum.diskbb_1.norm_xspec"}


def test_scoped_component_registers_one_site_per_observation(obsconfs):
    model = BayesianModel(Tbabs() * Diskbb(), _disk_prior("spectrum.diskbb_1.norm[*]"), obsconfs)

    sites = {key for key in model.prior_samples(num_samples=1) if key.startswith("derived.")}

    assert sites == {f"derived.forward.spectrum.{obs}.diskbb_1.norm_xspec" for obs in obsconfs}


def test_published_value_matches_the_sampled_parameters(obsconfs):
    """The site is the XSPEC normalization of the parameters drawn for that sample."""
    model = BayesianModel(Tbabs() * Diskbb(), _disk_prior(), obsconfs)
    samples = model.prior_samples(num_samples=1)

    def scalar(name):
        return float(np.asarray(samples[name]).ravel()[0])

    tin, norm = scalar("spectrum.diskbb_1.Tin"), scalar("spectrum.diskbb_1.norm")
    expected = norm / float(_disk_band_photon_flux(tin, STANDARD_RADIAL_EXPONENT, *BAND))

    assert np.isclose(scalar("derived.spectrum.diskbb_1.norm_xspec"), expected, rtol=1e-10)


def test_derived_sites_are_not_free_parameters(obsconfs):
    """Deterministic sites are not coordinates of the parameter vector."""
    prior = _disk_prior()
    model = BayesianModel(Tbabs() * Diskbb(), prior, obsconfs)

    assert set(model.parameter_names) == set(prior)
    assert model.dict_to_array({name: 1.0 for name in prior}).shape == (len(prior),)


def test_callable_priors_register_per_observation_sites(obsconfs):
    """A callable prior binds every leaf per observation, so nothing it feeds is shared."""

    def prior_factory():
        def leaf_prior(path, shape):
            return dist.Uniform(0.1, 5.0) if path.endswith("Tin") else dist.LogUniform(1e-6, 1e-2)

        return leaf_prior

    model = BayesianModel(Tbabs() * Diskbb(), prior_factory, obsconfs)

    sites = {key for key in model.prior_samples(num_samples=1) if key.startswith("derived.")}

    assert sites == {f"derived.forward.spectrum.{obs}.diskbb_1.norm_xspec" for obs in obsconfs}


# --- Reach beyond spectral components --------------------------------------------


class _PublishingInstrument(InstrumentModel):
    def derived_quantities(self):
        return {"probe": jnp.asarray(2.0)}


class _PublishingBackground(BackgroundModel):
    is_stochastic = False

    def __call__(self, observation):
        return jnp.zeros_like(observation.folded_background.data)

    def derived_quantities(self):
        return {"probe": jnp.asarray(3.0)}


def test_instrument_and_background_layers_can_publish(obsconfs):
    """The hook is not spectral-only: the collection walks the whole forward model."""
    from jaxspec.fit._forward_model import ForwardModel

    obs_name = next(iter(obsconfs))
    forward = ForwardModel(
        Tbabs() * Powerlaw(),
        obsconfs,
        instrument_model={obs_name: _PublishingInstrument()},
        background_model={obs_name: _PublishingBackground()},
    )

    collected = forward.derived_quantities(_leaf_values(forward))

    by_location = {
        (item.prefix, item.observation, item.owner_path, item.name): item.value
        for item in collected
    }
    assert by_location[("instrument", obs_name, "", "probe")] == 2.0
    assert by_location[("background", obs_name, "", "probe")] == 3.0


def test_nested_gain_and_shift_publishers_use_bound_values(obsconfs):
    class _Gain(ConstantGain):
        def derived_quantities(self):
            return {"factor_squared": self.factor[...] ** 2}

    class _Shift(ConstantShift):
        def derived_quantities(self):
            return {"offset_squared": self.offset[...] ** 2}

    model = BayesianModel(
        Tbabs() * Powerlaw(),
        {
            "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
            "spectrum.powerlaw_1.alpha": dist.Uniform(0.0, 5.0),
            "spectrum.powerlaw_1.norm": dist.LogUniform(1e-6, 1e-2),
            "instrument.gain.factor[*]": dist.Uniform(0.9, 1.1),
            "instrument.shift.offset[*]": dist.Uniform(-0.5, 0.5),
        },
        obsconfs,
        instrument_model={obs: InstrumentModel(gain=_Gain(), shift=_Shift()) for obs in obsconfs},
    )
    inputs = _leaf_values(
        model.forward_model,
        overrides={"gain.factor": 1.2, "shift.offset": 0.3},
    )

    collected = model.forward_model.derived_quantities(inputs)
    values = {(item.owner_path, item.name): float(item.value) for item in collected}
    sites = model._derived_sites(model.forward_model, inputs)

    assert np.isclose(values[("gain", "factor_squared")], 1.2**2)
    assert np.isclose(values[("shift", "offset_squared")], 0.3**2)
    assert set(sites) == {
        f"derived.forward.instrument.{obs}.{owner}.{name}"
        for obs in obsconfs
        for owner, name in (("gain", "factor_squared"), ("shift", "offset_squared"))
    }


def test_instrument_replicas_publish_per_observation_even_when_parameters_are_shared(obsconfs):
    """Instrument replicas are distinct user-supplied objects, so a shared parameter
    does not make their published values equal: they publish per observation."""

    class _Calibrated(InstrumentModel):
        def __init__(self, deadtime):
            super().__init__(gain=ConstantGain())
            self._deadtime = deadtime

        def derived_quantities(self):
            return {"eff_rate": jnp.asarray(self._deadtime) * self.gain.factor[...]}

    model = BayesianModel(
        Tbabs() * Powerlaw(),
        {
            "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
            "spectrum.powerlaw_1.alpha": dist.Uniform(0.0, 5.0),
            "spectrum.powerlaw_1.norm": dist.LogUniform(1e-6, 1e-2),
            "instrument.gain.factor": dist.Uniform(0.9, 1.1),  # shared
        },
        obsconfs,
        instrument_model={name: _Calibrated(100.0 * (i + 1)) for i, name in enumerate(obsconfs)},
    )
    samples = model.prior_samples(num_samples=1)

    published = {key for key in samples if key.endswith(".eff_rate")}

    assert published == {f"derived.forward.instrument.{obs}.eff_rate" for obs in obsconfs}
    # Each site carries its own replica's constant, not one replica's for all.
    factor = float(np.asarray(samples["instrument.gain.factor"]).ravel()[0])
    for i, obs in enumerate(obsconfs):
        value = float(np.asarray(samples[f"derived.forward.instrument.{obs}.eff_rate"]).ravel()[0])
        assert np.isclose(value, 100.0 * (i + 1) * factor)


def test_a_component_without_parameters_publishes_per_observation(obsconfs):
    """A component without parameters is not shared: its quantities can still differ
    per observation."""

    class _Probe(InstrumentModel):
        def __init__(self, value):
            super().__init__()
            self._value = value

        def derived_quantities(self):
            return {"probe": jnp.asarray(self._value)}

    model = BayesianModel(
        Tbabs() * Powerlaw(),
        {
            "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
            "spectrum.powerlaw_1.alpha": dist.Uniform(0.0, 5.0),
            "spectrum.powerlaw_1.norm": dist.LogUniform(1e-6, 1e-2),
        },
        obsconfs,
        instrument_model={name: _Probe(float(i + 7)) for i, name in enumerate(obsconfs)},
    )
    samples = model.prior_samples(num_samples=1)

    published = {
        key: float(np.asarray(value).ravel()[0])
        for key, value in samples.items()
        if key.endswith(".probe")
    }

    assert set(published) == {f"derived.forward.instrument.{obs}.probe" for obs in obsconfs}
    assert sorted(published.values()) == [7.0, 8.0, 9.0]


def test_a_background_wrapping_a_bare_component_keeps_its_observation(obsconfs):
    """`SpectralModelBackground(Diskbb())` makes the component *be* the wrapper: the
    internal `spectral_model` segment resolves to the background's own path."""
    from jaxspec.model.background import SpectralModelBackground

    model = BayesianModel(
        Tbabs() * Powerlaw(),
        {
            "spectrum.tbabs_1.nh": dist.Uniform(0.0, 1.0),
            "spectrum.powerlaw_1.alpha": dist.Uniform(0.0, 5.0),
            "spectrum.powerlaw_1.norm": dist.LogUniform(1e-6, 1e-2),
            "background.Tin[*]": dist.Uniform(0.5, 3.0),
            "background.norm": dist.LogUniform(1e-6, 1e-2),
        },
        obsconfs,
        background_model=SpectralModelBackground(Diskbb()),
    )
    samples = model.prior_samples(num_samples=1)
    sites = {key for key in samples if key.startswith("derived.")}

    assert sites == {f"derived.forward.background.{obs}.norm_xspec" for obs in obsconfs}

    def scalar(name):
        return float(np.asarray(samples[name]).ravel()[0])

    for obs in obsconfs:
        expected = scalar("background.norm") / float(
            _disk_band_photon_flux(
                scalar(f"forward.background.{obs}.Tin"),
                STANDARD_RADIAL_EXPONENT,
                *BAND,
            )
        )
        assert np.isclose(scalar(f"derived.forward.background.{obs}.norm_xspec"), expected)


@pytest.mark.parametrize("name", ["", "a.b"])
def test_a_dotted_or_empty_quantity_name_is_rejected(obsconfs, name):
    """Site names are decomposed on dots, so a dotted quantity name is ambiguous."""
    from jaxspec.fit._forward_model import ForwardModel

    class _BadName(InstrumentModel):
        def derived_quantities(self):
            return {name: jnp.asarray(1.0)}

    obs_name = next(iter(obsconfs))
    forward = ForwardModel(Tbabs() * Powerlaw(), obsconfs, instrument_model={obs_name: _BadName()})

    with pytest.raises(ValueError, match="free of dots"):
        forward.derived_quantities(_leaf_values(forward))


def test_a_per_obs_derived_column_keeps_its_observation_label():
    """A scoped quantity is labelled like its parameters: `diskbb_1.norm_xspec [PN]`."""
    import xarray as xr

    from jaxspec.analysis.results import FitResult

    name = "derived.forward.spectrum.PN.diskbb_1.norm_xspec"
    array = xr.DataArray(np.zeros(4), dims=["sample"], name=name)

    (frame,) = FitResult._var_to_dataframes(None, name, array, ["PN"])

    assert frame.name == "diskbb_1.norm_xspec\n[PN]"
