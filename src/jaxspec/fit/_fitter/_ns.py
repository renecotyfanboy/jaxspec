import importlib.util
import re

import jax

from jax import random
from numpyro.primitives import Messenger

from ...analysis.results import FitResult
from ._base import BayesianModelFitter


def _sanitise_name(name: str) -> str:
    """Replace every character that is not valid in a Python identifier with ``_``."""
    _INVALID_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
    sanitised = _INVALID_IDENT_RE.sub("_", name)
    if sanitised and sanitised[0].isdigit():
        sanitised = "_" + sanitised
    return sanitised


def _build_name_mapping(names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Build bijective mappings *original → sanitised* and *sanitised → original*.

    Raises ``ValueError`` if two distinct original names would collide after
    sanitisation.
    """
    original_to_safe: dict[str, str] = {}
    safe_to_original: dict[str, str] = {}

    for name in names:
        safe = _sanitise_name(name)
        if safe in safe_to_original and safe_to_original[safe] != name:
            raise ValueError(
                f"Name collision after sanitisation: both '{safe_to_original[safe]}' "
                f"and '{name}' map to '{safe}'."
            )
        original_to_safe[name] = safe
        safe_to_original[safe] = name

    return original_to_safe, safe_to_original


def _collect_site_names(model) -> list[str]:
    """Trace a numpyro model and return the names of all sample / plate sites."""
    from numpyro.handlers import seed, trace

    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
    return [site["name"] for site in model_trace.values() if site["type"] in ("sample", "plate")]


class _RenameSites(Messenger):
    """Numpyro effect handler that renames sample / plate / deterministic sites.

    Only names present in *name_map* are renamed; all other sites pass through
    unchanged.
    """

    def __init__(self, fn, name_map: dict[str, str]):
        super().__init__(fn)
        self._name_map = name_map

    def process_message(self, msg):
        if msg["type"] in ("sample", "plate", "deterministic"):
            name = msg["name"]
            if name in self._name_map:
                msg["name"] = self._name_map[name]


class NSFitter(BayesianModelFitter):
    r"""
    Fit a model using the Nested Sampling algorithm via
    [`jaxns`](https://jaxns.readthedocs.io/en/latest/) through the
    [`numpyro.contrib.nested_sampling.NestedSampler`][] wrapper.

    Because `jaxns` requires sample-site names to be valid Python identifiers,
    this fitter transparently renames sites that contain characters such as
    ``.`` (e.g. ``spectral_model.components.powerlaw_1.alpha``) before passing
    the model to the nested sampler, and maps them back afterwards.

    !!! info
        Ensure the prior distributions cover a large enough volume for the
        algorithm to yield proper results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            if importlib.util.find_spec("jaxns") is None:
                raise ModuleNotFoundError
        except (ModuleNotFoundError, ValueError):
            raise ImportError("jaxns is not installed. Please install it with `pip install jaxns`.")

    def fit(
        self,
        rng_key: int = 0,
        num_samples: int = 1000,
        num_live_points: int = 1000,
        plot_diagnostics: bool = False,
        termination_kwargs: dict | None = None,
        constructor_kwargs: dict | None = None,
        verbose: bool = True,
        use_transformed_model: bool = True,
    ) -> FitResult:
        """
        Fit the model to the data using the Phantom-Powered nested sampling algorithm.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_samples: the number of posterior samples to draw.
            num_live_points: the number of live points used by the NS algorithm.
            plot_diagnostics: whether to plot the diagnostics of the NS algorithm.
            termination_kwargs: keyword arguments forwarded to the jaxns termination
                criterion.
            constructor_kwargs: extra keyword arguments forwarded to
                ``jaxns.DefaultNestedSampler``.
            verbose: whether to print progress.
            use_transformed_model: whether to use the reparameterised model.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance.
        """
        from numpyro.contrib.nested_sampling import NestedSampler

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        all_site_names = _collect_site_names(numpyro_model)
        names_needing_rename = [n for n in all_site_names if n != _sanitise_name(n)]

        if names_needing_rename:
            original_to_safe, safe_to_original = _build_name_mapping(all_site_names)
            wrapped_model = _RenameSites(numpyro_model, original_to_safe)
        else:
            wrapped_model = numpyro_model
            safe_to_original = {}

        keys = random.split(random.PRNGKey(rng_key), 4)

        # Default to a single device: jaxns shards the per-bin likelihood across
        # devices, which fails with an IndivisibleError when the number of bins
        # is not a multiple of the number of devices. Users can override via
        # ``constructor_kwargs={"devices": jax.devices()}``.
        default_constructor = dict(
            verbose=verbose,
            difficult_model=True,
            max_samples=1e5,
            parameter_estimation=True,
            gradient_guided=False,
            devices=jax.devices()[:1],
            num_live_points=num_live_points,
        )
        if constructor_kwargs:
            default_constructor.update(constructor_kwargs)

        ns = NestedSampler(
            wrapped_model,
            constructor_kwargs=default_constructor,
            termination_kwargs=termination_kwargs or {},
        )

        ns.run(keys[0])

        if plot_diagnostics:
            ns.diagnostics()

        posterior = ns.get_samples(keys[1], num_samples=num_samples)

        if safe_to_original:
            posterior = {safe_to_original.get(k, k): v for k, v in posterior.items()}

        inference_data = self.build_inference_data(
            posterior, num_chains=1, use_transformed_model=use_transformed_model
        )

        results = ns._results
        if results is not None:
            inference_data.attrs["log_Z_mean"] = float(results.log_Z_mean)
            inference_data.attrs["log_Z_uncert"] = float(results.log_Z_uncert)
            inference_data.attrs["ESS"] = float(results.ESS)
            inference_data.attrs["H_mean"] = float(results.H_mean)
            inference_data.attrs["total_num_samples"] = int(results.total_num_samples)
            inference_data.attrs["total_num_likelihood_evaluations"] = int(
                results.total_num_likelihood_evaluations
            )

        return FitResult(
            self,
            inference_data,
            background_model=self.background_model,
        )
