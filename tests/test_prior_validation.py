"""Build-time validation: malformed / unmatched / out-of-scope prior keys and
mis-keyed instrument/background dicts must raise loudly when the
:class:`BayesianModel` is constructed (or at first sampling), never silently."""

import numpyro.distributions as dist
import pytest

from helpers import (
    dict_of_obsconf,
    list_of_obsconf,
    prior_shared_pars,
    single_obsconf,
    spectral_model,
)
from jaxspec.fit import BayesianModel
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import BackgroundWithError, SpectralModelBackground
from jaxspec.model.instrument import ConstantGain, ConstantShift, InstrumentModel
from jaxspec.model.multiplicative import Tbabs


def test_unmatched_prior_key_raises_at_build_time():
    """A typo'd parameter path (valid prefix, bogus tail) must raise at build
    time instead of being silently dropped."""
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alph": dist.Uniform(0, 5),
    }
    with pytest.raises(KeyError, match="does not match any model parameter"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.parametrize(
    "bad_key", ["spectrum.powerlaw_1.nrom[*]", "spectrum.powerlaw_1.nrom[data_0]"]
)
def test_unmatched_scoped_prior_key_raises_at_build_time(bad_key):
    """Same strictness for [*] / [obs] scoped keys with a typo'd parameter path."""
    prior = {**prior_shared_pars, bad_key: dist.LogUniform(1e-5, 1e-2)}
    with pytest.raises(KeyError, match="does not match any model parameter"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_unknown_obs_in_scope_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm[not_an_obs]": dist.LogUniform(1e-5, 1e-2),
    }

    with pytest.raises(ValueError, match="not in the 'spectrum' applicable set"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_instrument_scope_without_instrument_model_raises():
    prior = {
        **prior_shared_pars,
        "instrument.gain.factor[*]": dist.Uniform(0.5, 1.5),
    }

    # No instrument_model passed → no observations are 'applicable' for the
    # instrument prefix; the build should error early.
    with pytest.raises(ValueError, match="no observations are attached"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_prior_key_without_module_prefix_raises():
    """A flat key like 'tbabs_1_nh' (no dot, no module prefix) raises clearly."""
    prior = {
        **prior_shared_pars,
        "tbabs_1_nh": dist.Uniform(0, 1),
    }
    with pytest.raises(ValueError, match="no module prefix"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_prior_key_with_unknown_module_prefix_raises():
    """A key starting with an unrecognised module (e.g. 'foo.bar.baz') raises clearly."""
    prior = {
        **prior_shared_pars,
        "foo.tbabs_1.nh": dist.Uniform(0, 1),
    }
    with pytest.raises(ValueError, match="unknown module 'foo'"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_instrument_model_with_unknown_obs_key_raises():
    """instrument_model dict with a key that doesn't match any obs raises clearly."""
    with pytest.raises(ValueError, match="not in the observation set"):
        BayesianModel(
            spectral_model,
            prior_shared_pars,
            dict_of_obsconf,
            instrument_model={
                "typo_obs": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            },
        )


def test_background_model_dict_with_unknown_obs_key_raises():
    """background_model dict with a key that doesn't match any obs raises clearly."""
    with pytest.raises(ValueError, match="not in the observation set"):
        BayesianModel(
            spectral_model,
            prior_shared_pars,
            dict_of_obsconf,
            background_model={"typo_obs": BackgroundWithError()},
        )


def test_invalid_fixed_prior_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alpha": object(),
    }

    with pytest.raises(TypeError, match="Invalid fixed prior value"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_invalid_split_fixed_prior_raises_at_build_time():
    class NotCastableAsArray:
        def __array__(self, dtype=None):
            raise TypeError("cannot convert to array")

    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm[*]": NotCastableAsArray(),
    }

    with pytest.raises(TypeError, match="Invalid fixed prior value"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


def test_unmatched_key_only_exists_on_observation():
    """A scoped key whose parameter exists only on *other* observations reports
    which observation(s) actually own it."""
    prior = {
        **prior_shared_pars,
        # shift.offset exists only on MOS1 (MOS2 carries a gain) — requesting it
        # for MOS2 resolves to zero leaves.
        "instrument.shift.offset[MOS2]": dist.Uniform(-0.1, 0.1),
    }
    with pytest.raises(KeyError, match="only exists on observation"):
        BayesianModel(
            spectral_model,
            prior,
            dict_of_obsconf,
            instrument_model={
                "MOS1": InstrumentModel(shift=ConstantShift()),
                "MOS2": InstrumentModel(gain=ConstantGain()),
            },
        )


def test_background_model_without_background_raises_on_sampling():
    obsconf_without_background = single_obsconf.copy(deep=True)
    del obsconf_without_background["folded_background"]
    bkg_spec = Tbabs() * Powerlaw()
    prior = {
        **prior_shared_pars,
        "background.tbabs_1.nh": dist.Uniform(0, 1),
        "background.powerlaw_1.alpha": dist.Uniform(0, 5),
        "background.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
    }

    bayesian_model = BayesianModel(
        spectral_model,
        prior,
        obsconf_without_background,
        background_model=SpectralModelBackground(bkg_spec),
    )

    with pytest.raises(ValueError, match="Trying to fit a background model but no background"):
        bayesian_model.prior_samples(num_samples=1)
