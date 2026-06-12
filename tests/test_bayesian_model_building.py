from contextlib import nullcontext as does_not_raise

import jax
import numpyro.distributions as dist
import pytest

from conftest import (
    dict_of_obsconf,
    list_of_obsconf,
    prior_shared_pars,
    prior_split_pars,
    single_obsconf,
    spectral_model,
)
from jaxspec.fit import BayesianModel, MCMCFitter, NSFitter, TiedParameter, VIFitter
from jaxspec.model.additive import Powerlaw
from jaxspec.model.background import (
    BackgroundWithError,
    SpectralModelBackground,
)
from jaxspec.model.instrument import ConstantGain, ConstantShift, InstrumentModel
from jaxspec.model.multiplicative import Tbabs
from numpyro.optim import optax_to_numpyro
from optax import adamw

sparsify_marker = pytest.mark.parametrize(
    "sparse",
    [
        pytest.param(True, id="sparse matrix"),
        pytest.param(False, id="dense matrix"),
    ],
)


data_prior_marker = pytest.mark.parametrize(
    "model, prior, obsconf, expectation",
    [
        pytest.param(
            spectral_model,
            prior_shared_pars,
            single_obsconf,
            does_not_raise(),
            id="single observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            single_obsconf,
            does_not_raise(),
            id="single observation-split parameters",
        ),
        pytest.param(
            spectral_model,
            prior_shared_pars,
            list_of_obsconf,
            does_not_raise(),
            id="list of observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_shared_pars,
            dict_of_obsconf,
            does_not_raise(),
            id="dict of observation-shared parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            list_of_obsconf,
            does_not_raise(),
            id="list of observation-split parameters",
        ),
        pytest.param(
            spectral_model,
            prior_split_pars,
            dict_of_obsconf,
            does_not_raise(),
            id="dict of observation-split parameters",
        ),
    ],
)

mcmc_marker = pytest.mark.parametrize(
    "sampler",
    [
        pytest.param("nuts", id="NUTS"),
        pytest.param("aies", id="AIES"),
        pytest.param("ess", id="ESS"),
    ],
)


@pytest.mark.fast
@sparsify_marker
@data_prior_marker
def test_build_model(model, prior, obsconf, expectation, sparse):
    """Try to build a model from the given combination of observation and priors"""
    with expectation:
        BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)


@pytest.mark.slow
@sparsify_marker
@data_prior_marker
def test_mock_obs(model, prior, obsconf, expectation, sparse):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        bayesian_model = BayesianModel(model, prior, obsconf, sparsify_matrix=sparse)
        bayesian_model.mock_observations(bayesian_model.prior_samples())


@pytest.mark.fast
def test_prior_samples_with_ragged_background_default_prior():
    fitter = MCMCFitter(
        spectral_model,
        prior_shared_pars,
        list_of_obsconf,
        background_model=BackgroundWithError(),
    )
    prior_samples = fitter.prior_samples(num_samples=1)

    for i, obsconf in enumerate(list_of_obsconf):
        # Per-obs sites are registered under the literal "forward." prefix.
        site_name = f"forward.background.data_{i}.countrate"
        assert site_name in prior_samples
        assert prior_samples[site_name].shape == (1, len(obsconf.folded_background))


@pytest.mark.slow
@mcmc_marker
@data_prior_marker
def test_run_mcmc(model, prior, obsconf, expectation, sampler):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = MCMCFitter(model, prior, obsconf, background_model=BackgroundWithError())
        result = forward_model.fit(
            num_chains=4,
            num_warmup=10,
            num_samples=10,
            sampler=sampler,
            mcmc_kwargs={"progress_bar": False},
        )

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@data_prior_marker
def test_run_ns(model, prior, obsconf, expectation):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        bkg_spectral_model = Tbabs() * Powerlaw()
        # SpectralModelBackground exposes its inner spec's components directly
        # under the ``background.`` prefix via the ``user_path`` hook — users
        # write the same keys they would for the source spectrum.
        prior_with_backgrounds = {
            **prior,
            "background.powerlaw_1.alpha": dist.Uniform(0, 5),
            "background.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
            "background.tbabs_1.nh": dist.Uniform(0, 1),
        }
        forward_model = NSFitter(
            model,
            prior_with_backgrounds,
            obsconf,
            background_model=SpectralModelBackground(bkg_spectral_model),
        )
        # Aggressive stopping criterion: keep this on par with the MCMC/VI tests.
        result = forward_model.fit(
            num_samples=10,
            num_live_points=50,
            termination_kwargs={"max_samples": 200},
            verbose=False,
        )

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.slow
@data_prior_marker
def test_run_vi(model, prior, obsconf, expectation):
    """Try to generate mock observations from the given combination of observation and priors"""
    with expectation:
        forward_model = VIFitter(model, prior, obsconf, background_model=BackgroundWithError())
        optim = optax_to_numpyro(adamw(3e-4))

        result = forward_model.fit(
            num_steps=100,
            num_samples=10,
            optimizer=optim,
            plot_diagnostics=True,
        )

        result.photon_flux(0.7, 1.2, register=True)
        result.energy_flux(0.7, 1.2, register=True)
        result.luminosity(0.7, 1.2, redshift=0.01, register=True)
        result.c_stat
        [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
        result.to_chain("test")


@pytest.mark.fast
def test_energy_grid_shift_consistent_with_native_folding():
    """Regression: the user-grid folding path must apply the instrument energy
    shift in the same frame as the native path. With a fine grid and a smooth
    spectrum both paths must produce (nearly) identical folded counts for a
    shifted instrument — the old code shifted the *source* grid labels in
    ``InstrumentModel.fold``, inverting the shift's sign."""
    import jax.numpy as jnp
    import numpy as np

    from jaxspec.fit._forward_model import ForwardModel

    obs = single_obsconf
    model = Powerlaw() + Powerlaw()
    offset = 0.2

    native_energies = np.asarray(obs.in_energies)
    grid = jnp.geomspace(
        max(native_energies.min() - offset, 1e-2),
        native_energies.max() + 2 * offset,
        6000,
    )

    def build(energy_grid):
        return ForwardModel(
            model,
            obs,
            instrument_model={"data": InstrumentModel(shift=ConstantShift())},
            energy_grid=energy_grid,
        )

    inputs = {
        "spectrum.data.powerlaw_1.alpha": jnp.asarray(1.7),
        "spectrum.data.powerlaw_1.norm": jnp.asarray(1e-3),
        "spectrum.data.powerlaw_2.alpha": jnp.asarray(2.5),
        "spectrum.data.powerlaw_2.norm": jnp.asarray(5e-4),
        "instrument.data.shift.offset": jnp.asarray(offset),
    }

    native = build(None).evaluate(inputs)["data"]["source"]
    gridded = build(grid).evaluate(inputs)["data"]["source"]

    np.testing.assert_allclose(np.asarray(gridded), np.asarray(native), rtol=1e-3)


@pytest.mark.fast
def test_unmatched_prior_key_raises_at_build_time():
    """A typo'd parameter path (valid prefix, bogus tail) must raise at build
    time instead of being silently dropped."""
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alph": dist.Uniform(0, 5),
    }
    with pytest.raises(KeyError, match="does not match any model parameter"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad_key", ["spectrum.powerlaw_1.nrom[*]", "spectrum.powerlaw_1.nrom[data_0]"]
)
def test_unmatched_scoped_prior_key_raises_at_build_time(bad_key):
    """Same strictness for [*] / [obs] scoped keys with a typo'd parameter path."""
    prior = {**prior_shared_pars, bad_key: dist.LogUniform(1e-5, 1e-2)}
    with pytest.raises(KeyError, match="does not match any model parameter"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_unknown_obs_in_scope_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.norm[not_an_obs]": dist.LogUniform(1e-5, 1e-2),
    }

    with pytest.raises(ValueError, match="not in the 'spectrum' applicable set"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_instrument_scope_without_instrument_model_raises():
    prior = {
        **prior_shared_pars,
        "instrument.gain.factor[*]": dist.Uniform(0.5, 1.5),
    }

    # No instrument_model passed → no observations are 'applicable' for the
    # instrument prefix; the build should error early.
    with pytest.raises(ValueError, match="no observations are attached"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_prior_key_without_module_prefix_raises():
    """A flat key like 'tbabs_1_nh' (no dot, no module prefix) raises clearly."""
    prior = {
        **prior_shared_pars,
        "tbabs_1_nh": dist.Uniform(0, 1),
    }
    with pytest.raises(ValueError, match="no module prefix"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
def test_prior_key_with_unknown_module_prefix_raises():
    """A key starting with an unrecognised module (e.g. 'foo.bar.baz') raises clearly."""
    prior = {
        **prior_shared_pars,
        "foo.tbabs_1.nh": dist.Uniform(0, 1),
    }
    with pytest.raises(ValueError, match="unknown module 'foo'"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
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


@pytest.mark.fast
def test_background_model_dict_with_unknown_obs_key_raises():
    """background_model dict with a key that doesn't match any obs raises clearly."""
    with pytest.raises(ValueError, match="not in the observation set"):
        BayesianModel(
            spectral_model,
            prior_shared_pars,
            dict_of_obsconf,
            background_model={"typo_obs": BackgroundWithError()},
        )


@pytest.mark.slow
@mcmc_marker
def test_instrument_model_building(sampler):
    prior_with_instruments = {
        **prior_shared_pars,
        "instrument.gain.factor[*]": dist.Uniform(0.8, 1.2),
        "instrument.shift.offset[*]": dist.Uniform(-0.1, +0.1),
    }
    forward_model = MCMCFitter(
        spectral_model,
        prior_with_instruments,
        dict_of_obsconf,
        background_model=None,
        instrument_model={
            "PN": None,  # explicit reference
            "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
        },
    )

    result = forward_model.fit(
        num_chains=4,
        num_warmup=10,
        num_samples=10,
        sampler=sampler,
        mcmc_kwargs={"progress_bar": False},
    )

    result.photon_flux(0.7, 1.2, register=True)
    result.energy_flux(0.7, 1.2, register=True)
    result.luminosity(0.7, 1.2, redshift=0.01, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    result.to_chain("test")


@pytest.mark.slow
@mcmc_marker
def test_tied_parameters(sampler):
    spectral_model = Tbabs() * (Powerlaw() + Powerlaw())
    prior = {
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        "spectrum.powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_2.alpha": TiedParameter("spectrum.powerlaw_1.alpha", lambda x: 0.5 * x),
        "spectrum.powerlaw_2.norm": dist.LogUniform(1e-5, 1e-2),
        "spectrum.tbabs_1.nh": 0.6,
    }

    forward_model = MCMCFitter(spectral_model, prior, dict_of_obsconf)

    result = forward_model.fit(
        num_chains=4,
        num_warmup=10,
        num_samples=10,
        sampler=sampler,
        mcmc_kwargs={"progress_bar": False},
    )

    result.photon_flux(0.7, 1.2, register=True)
    result.energy_flux(0.7, 1.2, register=True)
    result.luminosity(0.7, 1.2, redshift=0.01, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    chain = result.to_chain("test")

    # Tied destinations are deterministic functions of their source; the user
    # convention is to exclude them from corner plots so the displayed
    # parameter set matches the genuinely sampled posterior.
    cols = list(chain.samples.columns)
    assert not any(
        "powerlaw_2.alpha" in c for c in cols
    ), f"tied destination 'powerlaw_2.alpha' should not be in chain columns: {cols}"
    assert any(
        "powerlaw_1.alpha" in c for c in cols
    ), f"tied source 'powerlaw_1.alpha' should remain in chain columns: {cols}"


@pytest.mark.slow
def test_scoped_tied_parameters_fit_and_extraction():
    """Scoped TiedParameter entries must survive the full fit → posterior
    extraction path (``input_parameters`` / fluxes / PPC), not just prior
    sampling — extraction used to look up ``tied_to`` without parsing its
    ``[obs]`` scope and drop the direct entries of a mixed direct+tied base."""
    import numpy as np

    prior = {
        **{k: v for k, v in prior_shared_pars.items() if k != "spectrum.powerlaw_1.norm"},
        "spectrum.powerlaw_1.norm[MOS1]": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_1.norm[MOS2]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.5 * x
        ),
        "spectrum.powerlaw_1.norm[PN]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.25 * x
        ),
    }

    fitter = MCMCFitter(spectral_model, prior, dict_of_obsconf)
    result = fitter.fit(
        num_chains=2,
        num_warmup=5,
        num_samples=5,
        sampler="nuts",
        mcmc_kwargs={"progress_bar": False},
    )

    norm = np.asarray(result.input_parameters["spectrum.powerlaw_1.norm"])
    obs_order = list(dict_of_obsconf.keys())
    mos1 = norm[..., obs_order.index("MOS1")]
    np.testing.assert_allclose(norm[..., obs_order.index("MOS2")], 0.5 * mos1, rtol=1e-6)
    np.testing.assert_allclose(norm[..., obs_order.index("PN")], 0.25 * mos1, rtol=1e-6)

    result.photon_flux(0.7, 1.2, register=True)
    [result._ppc_folded_branches(obs_id) for obs_id in result.obsconfs.keys()]
    result.to_chain("test")


@pytest.mark.slow
def test_to_chain_includes_per_obs_columns():
    """[*] per-obs parameters should appear in to_chain output, one column per obs.

    Pinned to NUTS — the column structure is sampler-independent and ensemble
    samplers (AIES / ESS) want n_chains ≥ 2·n_params which we'd need to scale
    up for, defeating the point of a fast smoke test.
    """
    prior = {
        **{k: v for k, v in prior_shared_pars.items() if k != "spectrum.powerlaw_1.norm"},
        "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
    }
    forward = MCMCFitter(spectral_model, prior, dict_of_obsconf)
    result = forward.fit(
        num_chains=2,
        num_warmup=5,
        num_samples=5,
        sampler="nuts",
        mcmc_kwargs={"progress_bar": False},
    )
    chain = result.to_chain("test")
    cols = list(chain.samples.columns)

    # One per-obs column per observation, labeled "<param>\n[<obs>]"
    per_obs_norm_cols = [c for c in cols if "powerlaw_1.norm" in c and "\n[" in c]
    assert len(per_obs_norm_cols) == len(
        dict_of_obsconf
    ), f"expected {len(dict_of_obsconf)} per-obs norm columns, got {per_obs_norm_cols}"
    obs_names_in_cols = {c.split("\n[")[1].rstrip("]") for c in per_obs_norm_cols}
    assert obs_names_in_cols == set(dict_of_obsconf.keys())


@pytest.mark.fast
def test_invalid_fixed_prior_raises_at_build_time():
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alpha": object(),
    }

    with pytest.raises(TypeError, match="Invalid fixed prior value"):
        BayesianModel(spectral_model, prior, list_of_obsconf)


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
@pytest.mark.parametrize(
    "key, expected",
    [
        ("spectrum.powerlaw_1.alpha", ("spectrum.powerlaw_1.alpha", None)),
        ("instrument.gain.factor[*]", ("instrument.gain.factor", "*")),
        ("spectrum.tbabs_1.nh[PN]", ("spectrum.tbabs_1.nh", "PN")),
        ("background.countrate[data_0]", ("background.countrate", "data_0")),
    ],
)
def test_parse_prior_key_valid(key, expected):
    from jaxspec.fit._prior_resolution import parse_prior_key

    assert parse_prior_key(key) == expected


@pytest.mark.fast
@pytest.mark.parametrize("bad", ["malformed[", "key[]", "[obs]", "a[b][c]", "[", "]"])
def test_parse_prior_key_malformed_raises(bad):
    from jaxspec.fit._prior_resolution import parse_prior_key

    with pytest.raises(ValueError, match="Malformed prior key"):
        parse_prior_key(bad)


@pytest.mark.fast
def test_joint_prior_factory_matches_components():
    """joint_prior_factory returns Delta for matching paths, None for misses."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from jaxspec.fit import joint_prior_factory

    captured = {}

    def model():
        lookup = joint_prior_factory(
            components=("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
            joint_dist=dist.MultivariateNormal(
                loc=jnp.array([2.0, -8.0]),
                covariance_matrix=jnp.eye(2),
            ),
            name="joint_alpha_norm",
        )
        # Match through the per-obs leaf-path convention
        captured["alpha"] = lookup("spectrum.PN.powerlaw_1.alpha")
        captured["norm"] = lookup("spectrum.MOS1.powerlaw_1.norm")
        captured["miss"] = lookup("spectrum.MOS1.tbabs_1.nh")

    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        model()

    assert "joint_alpha_norm" in tr
    # New leaf-callable contract: array-like returns are pre-sampled values
    # (no Delta wrapper). Joint slice picks out a scalar from the 2-vector draw.
    assert isinstance(captured["alpha"], jax.Array)
    assert isinstance(captured["norm"], jax.Array)
    assert captured["alpha"].shape == ()
    assert captured["norm"].shape == ()
    assert captured["miss"] is None


@pytest.mark.fast
def test_per_obs_tied_parameter_resolution():
    """TiedParameter with [obs]-scoped tied_to ties one obs's draw to another's."""
    import numpyro.distributions as dist

    from jaxspec.fit import BayesianModel

    spec_only = spectral_model
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
        # MOS1 gets an independent draw; MOS2 is tied to MOS1's draw via λ x: 0.5 * x
        "spectrum.powerlaw_1.norm[MOS1]": dist.LogUniform(1e-5, 1e-2),
        "spectrum.powerlaw_1.norm[MOS2]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.5 * x
        ),
        # PN also tied — same pattern
        "spectrum.powerlaw_1.norm[PN]": TiedParameter(
            "spectrum.powerlaw_1.norm[MOS1]", lambda x: 0.25 * x
        ),
    }
    # Strip the redundant entry from prior_shared_pars (it had spectrum.powerlaw_1.norm shared,
    # which conflicts with our per-obs entries above for the same path).
    prior = {k: v for k, v in prior.items() if k != "spectrum.powerlaw_1.norm"}

    bm = BayesianModel(spec_only, prior, dict_of_obsconf)

    samples = bm.prior_samples(num_samples=1)

    def site(obs):
        return f"forward.spectrum.{obs}.powerlaw_1.norm"

    mos1 = float(samples[site("MOS1")][0])
    mos2 = float(samples[site("MOS2")][0])
    pn = float(samples[site("PN")][0])
    assert abs(mos2 - 0.5 * mos1) < 1e-6
    assert abs(pn - 0.25 * mos1) < 1e-6


@pytest.mark.fast
def test_per_obs_tied_parameter_wildcard_source():
    """tied_to with [*] derives each dest obs from its same-obs source draw."""
    import numpyro.distributions as dist

    from jaxspec.fit import BayesianModel

    prior = {
        **{
            k: v
            for k, v in prior_shared_pars.items()
            if k not in ("spectrum.powerlaw_1.norm", "spectrum.blackbodyrad_1.norm")
        },
        "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),
        "spectrum.blackbodyrad_1.norm[*]": TiedParameter(
            "spectrum.powerlaw_1.norm[*]", lambda x: 2.0 * x
        ),
    }

    bm = BayesianModel(spectral_model, prior, dict_of_obsconf)
    samples = bm.prior_samples(num_samples=1)
    for obs in dict_of_obsconf:
        pw = float(samples[f"forward.spectrum.{obs}.powerlaw_1.norm"][0])
        bb = float(samples[f"forward.spectrum.{obs}.blackbodyrad_1.norm"][0])
        assert abs(bb - 2.0 * pw) < 1e-6


@pytest.mark.fast
def test_dict_prior_resolves_shared_split_and_specific():
    """dict_prior returns a pre-sampled array for shared / fixed entries,
    a Distribution for unresolved [*] / [obs] entries, None on miss."""
    import numpyro
    import numpyro.distributions as dist

    from jaxspec.fit import dict_prior

    user_prior = {
        "spectrum.tbabs_1.nh": dist.Uniform(0, 1),  # shared → pre-sampled array
        "spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2),  # [*]    → Distribution
        "spectrum.powerlaw_1.alpha[MOS1]": dist.Uniform(0, 5),  # [obs]  → Distribution
        "spectrum.blackbodyrad_1.kT": 0.5,  # fixed shared → pre-sampled array
    }
    captured = {}

    def model():
        lookup = dict_prior(user_prior)
        captured["shared_dist"] = lookup("spectrum.MOS1.tbabs_1.nh", ())
        captured["shared_fixed"] = lookup("spectrum.PN.blackbodyrad_1.kT", ())
        captured["split"] = lookup("spectrum.PN.powerlaw_1.norm", ())
        captured["specific_match"] = lookup("spectrum.MOS1.powerlaw_1.alpha", ())
        captured["specific_miss"] = lookup("spectrum.PN.powerlaw_1.alpha", ())
        captured["unknown"] = lookup("spectrum.MOS1.unknown.path", ())

    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        model()

    assert "spectrum.tbabs_1.nh" in tr  # shared sampled once at top
    # Shared / fixed entries return pre-sampled arrays (new leaf-callable contract).
    assert isinstance(captured["shared_dist"], jax.Array)
    assert isinstance(captured["shared_fixed"], jax.Array)
    # Per-obs / [*] entries still return Distributions (sampled at leaf time).
    assert isinstance(captured["split"], dist.LogUniform)
    assert isinstance(captured["specific_match"], dist.Uniform)
    assert captured["specific_miss"] is None  # only MOS1 has the [obs] entry
    assert captured["unknown"] is None


@pytest.mark.fast
def test_resolve_per_obs_entry_partial_coverage_stays_dict():
    """Regression: a per-obs entry covering only a subset of the applicable obs
    must NOT be collapsed into a compacted trailing axis — otherwise
    ``_leaf_inputs_from_input_parameters`` misindexes it by full-obs-order
    position and a leaf silently gets another obs's parameters. Partial coverage
    must stay a ``{obs: array}`` dict; full coverage still stacks."""
    from jaxspec.analysis.results import _resolve_per_obs_entry

    obs_axis = ["A", "B", "C"]
    chain_draw = (2, 5)

    # Leaf present only on B and C (e.g. gain on the non-reference instruments).
    partial = _resolve_per_obs_entry(
        {"B": 1.0, "C": 2.0},
        "instrument.gain.factor",
        obs_axis,
        {},
        chain_draw,
        [],
    )
    assert isinstance(partial, dict)
    assert set(partial) == {"B", "C"}

    # Full coverage still collapses into a trailing obs axis (len == n_obs).
    full = _resolve_per_obs_entry(
        {"A": 0.0, "B": 1.0, "C": 2.0},
        "instrument.gain.factor",
        obs_axis,
        {},
        chain_draw,
        [],
    )
    assert not isinstance(full, dict)
    assert full.shape == (*chain_draw, len(obs_axis))
