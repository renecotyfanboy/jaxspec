"""Unit tests for the prior-resolution machinery in ``jaxspec.fit._prior_resolution``
(plus the ``joint_prior_factory`` / ``dict_prior`` leaf callables and
``results._resolve_per_obs_entry``). These exercise the resolution helpers and
prior-sampling behaviour directly, without running inference."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from helpers import (
    dict_of_obsconf,
    list_of_obsconf,
    prior_shared_pars,
    single_obsconf,
    spectral_model,
)
from jaxspec.analysis.results import _resolve_per_obs_entry
from jaxspec.fit import BayesianModel, TiedParameter, dict_prior, joint_prior_factory
from jaxspec.fit._prior_resolution import (
    _KNOWN_PREFIXES,
    _missing_prior_message,
    _normalise_callable_prior,
    _split_nnx_leaf,
    bind_inputs,
    parse_prior_key,
    sample_prior,
)
from jaxspec.model.instrument import ConstantGain, ConstantShift, InstrumentModel

# --- parse_prior_key ---------------------------------------------------------


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
    assert parse_prior_key(key) == expected


@pytest.mark.parametrize("bad", ["malformed[", "key[]", "[obs]", "a[b][c]", "[", "]"])
def test_parse_prior_key_malformed_raises(bad):
    with pytest.raises(ValueError, match="Malformed prior key"):
        parse_prior_key(bad)


def test_split_nnx_leaf_too_short_raises():
    """_split_nnx_leaf rejects paths with fewer than three segments."""
    with pytest.raises(ValueError, match="Unexpected nnx leaf path"):
        _split_nnx_leaf("spectrum.alpha")


# --- joint_prior_factory -----------------------------------------------------


def test_joint_prior_factory_matches_components():
    """joint_prior_factory returns Delta for matching paths, None for misses."""
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


def test_joint_prior_factory_bare_component_path():
    """A lookup with a bare component path (no obs segment) hits the direct-match
    branch in joint_prior_factory's lookup."""
    captured = {}

    def model():
        lookup = joint_prior_factory(
            components=("spectrum.powerlaw_1.alpha", "spectrum.powerlaw_1.norm"),
            joint_dist=dist.MultivariateNormal(
                loc=jnp.array([2.0, -8.0]), covariance_matrix=jnp.eye(2)
            ),
            name="joint_bare",
        )
        captured["bare"] = lookup("spectrum.powerlaw_1.alpha")  # no obs segment

    with numpyro.handlers.trace(), numpyro.handlers.seed(rng_seed=0):
        model()

    assert isinstance(captured["bare"], jax.Array)
    assert captured["bare"].shape == ()


# --- dict_prior --------------------------------------------------------------


def test_dict_prior_resolves_shared_split_and_specific():
    """dict_prior returns a pre-sampled array for shared / fixed entries,
    a Distribution for unresolved [*] / [obs] entries, None on miss."""
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


def test_dict_prior_fixed_split_value():
    """A ``[*]`` dict_prior entry with a plain value is materialised to an array."""
    captured = {}

    def model():
        lookup = dict_prior({"spectrum.powerlaw_1.norm[*]": 1e-3})  # fixed, not a Distribution
        captured["v"] = lookup("spectrum.PN.powerlaw_1.norm", ())

    with numpyro.handlers.trace(), numpyro.handlers.seed(rng_seed=0):
        model()

    assert isinstance(captured["v"], jax.Array)


def test_dict_prior_short_path_returns_none():
    """dict_prior's lookup returns None for a leaf path with < 3 segments."""
    captured = {}

    def model():
        lookup = dict_prior({"spectrum.powerlaw_1.norm[*]": dist.LogUniform(1e-5, 1e-2)})
        captured["v"] = lookup("too.short", ())

    with numpyro.handlers.trace(), numpyro.handlers.seed(rng_seed=0):
        model()

    assert captured["v"] is None


# --- _resolve_per_obs_entry --------------------------------------------------


def test_resolve_per_obs_entry_partial_coverage_stays_dict():
    """Regression: a per-obs entry covering only a subset of the applicable obs
    must NOT be collapsed into a compacted trailing axis — otherwise
    ``_leaf_inputs_from_input_parameters`` misindexes it by full-obs-order
    position and a leaf silently gets another obs's parameters. Partial coverage
    must stay a ``{obs: array}`` dict; full coverage still stacks."""
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


# --- TiedParameter resolution ------------------------------------------------


def test_per_obs_tied_parameter_resolution():
    """TiedParameter with [obs]-scoped tied_to ties one obs's draw to another's."""
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
    # Strip the redundant shared entry that conflicts with the per-obs entries above.
    prior = {k: v for k, v in prior.items() if k != "spectrum.powerlaw_1.norm"}

    bm = BayesianModel(spectral_model, prior, dict_of_obsconf)

    samples = bm.prior_samples(num_samples=1)

    def site(obs):
        return f"forward.spectrum.{obs}.powerlaw_1.norm"

    mos1 = float(samples[site("MOS1")][0])
    mos2 = float(samples[site("MOS2")][0])
    pn = float(samples[site("PN")][0])
    assert abs(mos2 - 0.5 * mos1) < 1e-6
    assert abs(pn - 0.25 * mos1) < 1e-6


def test_per_obs_tied_parameter_wildcard_source():
    """tied_to with [*] derives each dest obs from its same-obs source draw."""
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


def test_tied_parameter_unknown_source_raises():
    """A TiedParameter whose source path matches no leaf raises at sample time."""
    prior = {
        **prior_shared_pars,
        "spectrum.powerlaw_1.alpha": TiedParameter("spectrum.nonexistent.param", lambda x: x),
    }
    bm = BayesianModel(spectral_model, prior, list_of_obsconf)
    with pytest.raises(ValueError, match="references unknown source"):
        bm.prior_samples(num_samples=1)


def test_tied_parameter_per_obs_cannot_match_source_raises():
    """A per-obs dest tied to a ``[*]`` source that doesn't cover that obs raises.

    Heterogeneous instruments: MOS1 has only a shift, MOS2 only a gain. The
    ``[*]`` source ``instrument.shift.offset`` therefore covers MOS1 alone,
    while the dest ``instrument.gain.factor[MOS2]`` lives on MOS2 — so the
    same-obs pairing finds no source value for MOS2."""
    prior = {
        **prior_shared_pars,
        "instrument.shift.offset[*]": dist.Uniform(-0.1, 0.1),
        "instrument.gain.factor[MOS2]": TiedParameter(
            "instrument.shift.offset[*]", lambda x: 1.0 + x
        ),
    }
    bm = BayesianModel(
        spectral_model,
        prior,
        dict_of_obsconf,
        instrument_model={
            "MOS1": InstrumentModel(shift=ConstantShift()),
            "MOS2": InstrumentModel(gain=ConstantGain()),
        },
    )
    with pytest.raises(ValueError, match="cannot match a source value"):
        bm.prior_samples(num_samples=1)


# --- Fixed-value entries -----------------------------------------------------


def test_shared_fixed_value_entry():
    """A shared (scope=None) fixed value is broadcast to every leaf as a constant."""
    prior = {k: v for k, v in prior_shared_pars.items() if k != "spectrum.blackbodyrad_1.kT"}
    prior["spectrum.blackbodyrad_1.kT"] = 0.5  # shared fixed scalar, not a Distribution

    bm = BayesianModel(spectral_model, prior, dict_of_obsconf)
    bm.prior_samples(num_samples=1)  # must not raise
    assert "spectrum.blackbodyrad_1.kT" not in bm.parameter_names


def test_per_obs_fixed_value_entry():
    """A ``[*]`` entry with a plain fixed value is bound as a constant (no site)."""
    prior = {k: v for k, v in prior_shared_pars.items() if k != "spectrum.powerlaw_1.norm"}
    prior["spectrum.powerlaw_1.norm[*]"] = 1e-3  # fixed scalar, not a Distribution

    bm = BayesianModel(spectral_model, prior, dict_of_obsconf)
    bm.prior_samples(num_samples=1)  # must not raise
    for obs in dict_of_obsconf:
        # Fixed values are deterministic — no free sample site is created.
        assert f"forward.spectrum.{obs}.powerlaw_1.norm" not in bm.parameter_names


# --- Callable-prior path -----------------------------------------------------

#: Leaf distributions keyed by ``"<component>.<param>"`` for the ``spectral_model``
#: (Tbabs * (Powerlaw + Blackbodyrad)). Used by the callable prior tests below to
#: resolve every nnx.Param leaf to a valid Distribution.
_CALLABLE_LEAF_DISTS = {
    "tbabs_1.nh": dist.Uniform(0, 1),
    "powerlaw_1.alpha": dist.Uniform(0, 5),
    "powerlaw_1.norm": dist.LogUniform(1e-5, 1e-2),
    "blackbodyrad_1.kT": dist.Uniform(0, 5),
    "blackbodyrad_1.norm": dist.LogUniform(1e-2, 1e2),
}


def _leaf_callable(path, shape):
    """2-arg leaf callable: ``path`` is ``"spectrum.<obs>.<component>.<param>"``."""
    return _CALLABLE_LEAF_DISTS[".".join(path.split(".")[2:])]


def test_callable_factory_prior_runs_through_model():
    """A 0-arg factory prior returning a 2-arg leaf callable runs end-to-end:
    exercises _normalise_callable_prior (0-arg branch), _sample_callable_prior,
    _sample_leaves' Distribution branch and _iter_pure_dict_values."""

    def factory():
        return _leaf_callable

    bm = BayesianModel(spectral_model, factory, dict_of_obsconf)
    samples = bm.prior_samples(num_samples=1)
    for obs in dict_of_obsconf:
        assert f"forward.spectrum.{obs}.powerlaw_1.alpha" in samples
        assert f"forward.spectrum.{obs}.tbabs_1.nh" in samples


def test_leaf_callable_prior_two_arg_form():
    """A bare 2-arg ``(path, shape) -> Distribution`` callable is used as-is by
    _normalise_callable_prior (n == 2 branch)."""
    bm = BayesianModel(spectral_model, _leaf_callable, single_obsconf)
    samples = bm.prior_samples(num_samples=1)
    assert "forward.spectrum.data.powerlaw_1.alpha" in samples


def test_callable_prior_wrong_arg_count_raises():
    """A callable prior taking neither 0 nor 2 args raises TypeError."""
    with pytest.raises(TypeError, match="Callable prior must take"):
        _normalise_callable_prior(lambda only_one_arg: only_one_arg)


# --- Error message / lookup helpers ------------------------------------------


def test_missing_prior_message_styles():
    """_missing_prior_message tailors its advice to the calling flow."""
    leaf = "spectrum.PN.powerlaw_1.alpha"
    prior_msg = _missing_prior_message(leaf, style="prior")
    assert "Add an entry like" in prior_msg
    assert "[*]" in prior_msg

    inputs_msg = _missing_prior_message(leaf, style="inputs")
    assert "evaluate() takes a flat" in inputs_msg

    # Short leaf paths (< 3 segments) take the terse fallback branch.
    assert "No prior provided" in _missing_prior_message("alpha", style="prior")
    assert "No value provided" in _missing_prior_message("alpha", style="inputs")


def test_sample_prior_unmatched_entries_raise():
    """Defensive guards in _sample_entry / _resolve_tied_entry (normally
    pre-empted by BayesianModel validation) raise on keys resolving to no leaves."""
    fm = BayesianModel(spectral_model, prior_shared_pars, single_obsconf).forward_model
    applicable = {prefix: set() for prefix in _KNOWN_PREFIXES}
    applicable["spectrum"] = {"data"}

    # Direct entry whose path matches no leaf.
    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(KeyError, match="does not match any model parameter"):
            sample_prior(fm, {"spectrum.powerlaw_1.bogus": dist.Uniform(0, 1)}, applicable)

    # Tied entry whose *destination* path matches no leaf (source is valid).
    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(KeyError, match="does not match any model parameter"):
            sample_prior(
                fm,
                {
                    "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
                    "spectrum.powerlaw_1.bogus": TiedParameter(
                        "spectrum.powerlaw_1.alpha", lambda x: x
                    ),
                },
                applicable,
            )


def test_bind_inputs_missing_leaf_raises():
    """bind_inputs with an empty inputs dict raises the rich missing-value error."""
    bm = BayesianModel(spectral_model, prior_shared_pars, single_obsconf)
    with pytest.raises(KeyError, match="No value provided for parameter"):
        bind_inputs(bm.forward_model, {})
