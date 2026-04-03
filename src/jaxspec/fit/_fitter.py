import importlib.util
import re
import warnings

from abc import ABC, abstractmethod
from typing import Literal

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpyro

from jax import random
from jax.numpy import concatenate
from jax.random import PRNGKey
from numpyro.infer import AIES, ESS, MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.primitives import Messenger

from ..analysis.results import FitResult
from ..model.background import SubtractedBackground
from ._bayesian_model import BayesianModel

# ---------------------------------------------------------------------------
# Helpers for sanitising numpyro sample-site names so that they are valid
# Python identifiers (required by jaxns / numpyro nested sampling).
# ---------------------------------------------------------------------------

_INVALID_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitise_name(name: str) -> str:
    """Replace every character that is not valid in a Python identifier with ``_``."""
    sanitised = _INVALID_IDENT_RE.sub("_", name)
    # Ensure the name does not start with a digit
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


class BayesianModelFitter(BayesianModel, ABC):
    def build_inference_data(
        self,
        posterior_samples,
        num_chains: int = 1,
        num_predictive_samples: int = 1000,
        key: PRNGKey = PRNGKey(42),
        use_transformed_model: bool = False,
        filter_inference_data: bool = True,
    ) -> az.InferenceData:
        """
        Build an [InferenceData][arviz.InferenceData] object from posterior samples.

        Parameters:
            posterior_samples: the samples from the posterior distribution.
            num_chains: the number of chains used to sample the posterior.
            num_predictive_samples: the number of samples to draw from the prior.
            key: the random key used to initialize the sampler.
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            filter_inference_data: whether to filter the InferenceData to keep only the relevant parameters.
        """

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        keys = random.split(key, 3)

        posterior_predictive = Predictive(numpyro_model, posterior_samples)(keys[0], observed=False)

        prior = Predictive(numpyro_model, num_samples=num_predictive_samples * num_chains)(
            keys[1], observed=False
        )

        log_likelihood = numpyro.infer.log_likelihood(numpyro_model, posterior_samples)
        if len(log_likelihood.keys()) > 1:
            log_likelihood["full"] = concatenate([ll for _, ll in log_likelihood.items()], axis=1)
            log_likelihood["obs/~/all"] = concatenate(
                [ll for k, ll in log_likelihood.items() if "obs" in k], axis=1
            )

            # Subtracted background is not fitted so there is no likelihood
            if self.background_model is not None and not isinstance(
                self.background_model, SubtractedBackground
            ):
                log_likelihood["bkg/~/all"] = concatenate(
                    [ll for k, ll in log_likelihood.items() if "bkg" in k], axis=1
                )

        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(numpyro_model, keys[2]),
            substitute_fn=numpyro.infer.init_to_sample,
        )

        observations = {
            name: site["value"]
            for name, site in numpyro.handlers.trace(seeded_model).get_trace().items()
            if site["type"] == "sample" and site["is_observed"]
        }

        def reshape_first_dimension(arr):
            new_dim = arr.shape[0] // num_chains
            new_shape = (num_chains, new_dim) + arr.shape[1:]
            reshaped_array = arr.reshape(new_shape)

            return reshaped_array

        posterior_samples = {
            key: reshape_first_dimension(value) for key, value in posterior_samples.items()
        }
        prior = {key: value[None, :] for key, value in prior.items()}
        posterior_predictive = {
            key: reshape_first_dimension(value) for key, value in posterior_predictive.items()
        }
        log_likelihood = {
            key: reshape_first_dimension(value) for key, value in log_likelihood.items()
        }

        inference_data = az.from_dict(
            posterior_samples,
            prior=prior,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            observed_data=observations,
        )

        return (
            self.filter_inference_data(inference_data) if filter_inference_data else inference_data
        )

    def filter_inference_data(
        self,
        inference_data: az.InferenceData,
    ) -> az.InferenceData:
        """
        Filter the inference data to keep only the relevant parameters for the observations.

        - Removes predictive parameters from deterministic random variables (e.g. kernel of background GP)
        - Removes parameters build from reparametrised variables (e.g. ending with `"_base"`)
        """

        predictive_parameters = []

        for key, value in self._observation_container.items():
            predictive_parameters.append(f"obs/~/{key}")
            if self.background_model is not None:
                predictive_parameters.append(f"bkg/~/{key}")
            #                predictive_parameters.append(f"ins/~/{key}")
        #                predictive_parameters.append(f"ins/~/{key}")

        inference_data.posterior_predictive = inference_data.posterior_predictive[
            predictive_parameters
        ]

        parameters = [
            x
            for x in inference_data.posterior.keys()
            if not (x.endswith("_base") or x.startswith("_"))
        ]
        inference_data.posterior = inference_data.posterior[parameters]
        inference_data.prior = inference_data.prior[parameters]

        return inference_data

    @abstractmethod
    def fit(self, **kwargs) -> FitResult: ...


class MCMCFitter(BayesianModelFitter):
    """
    A class to fit a model to a given set of observation using a Bayesian approach. This class uses samplers
    from numpyro to perform the inference on the model parameters.
    """

    kernel_dict = {
        "nuts": NUTS,
        "aies": AIES,
        "ess": ESS,
    }

    def fit(
        self,
        rng_key: int = 0,
        num_chains: int = len(jax.devices()),
        num_warmup: int = 1000,
        num_samples: int = 1000,
        sampler: Literal["nuts", "aies", "ess"] = "nuts",
        use_transformed_model: bool = True,
        kernel_kwargs: dict = {},
        mcmc_kwargs: dict = {},
    ) -> FitResult:
        """
        Fit the model to the data using a MCMC sampler from numpyro.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_chains: the number of chains to run.
            num_warmup: the number of warmup steps.
            num_samples: the number of samples to draw.
            sampler: the sampler to use. Can be one of "nuts", "aies" or "ess".
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            kernel_kwargs: additional arguments to pass to the kernel. See [`NUTS`][numpyro.infer.mcmc.MCMCKernel] for more details.
            mcmc_kwargs: additional arguments to pass to the MCMC sampler. See [`MCMC`][numpyro.infer.mcmc.MCMC] for more details.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        kernel = self.kernel_dict[sampler](bayesian_model, **kernel_kwargs)

        mcmc_kwargs = chain_kwargs | mcmc_kwargs

        if sampler in ["aies", "ess"] and mcmc_kwargs.get("chain_method", None) != "vectorized":
            mcmc_kwargs["chain_method"] = "vectorized"
            warnings.warn("The chain_method is set to 'vectorized' for AIES and ESS samplers")

        mcmc = MCMC(kernel, **mcmc_kwargs)
        keys = random.split(random.PRNGKey(rng_key), 3)

        mcmc.run(keys[0])

        posterior = mcmc.get_samples()

        inference_data = self.build_inference_data(
            posterior, num_chains=num_chains, use_transformed_model=use_transformed_model
        )

        return FitResult(
            self,
            inference_data,
            background_model=self.background_model,
        )


class VIFitter(BayesianModelFitter):
    def fit(
        self,
        rng_key: int = 0,
        num_steps: int = 10_000,
        optimizer: numpyro.optim._NumPyroOptim = numpyro.optim.Adam(step_size=0.0005),
        loss: numpyro.infer.elbo.ELBO = Trace_ELBO(),
        num_samples: int = 1000,
        guide: numpyro.infer.autoguide.AutoGuide | None = None,
        use_transformed_model: bool = True,
        plot_diagnostics: bool = False,
    ) -> FitResult:
        """
        Fit the model to the data using a variational inference approach from numpyro.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_steps: the number of steps for VI.
            optimizer: the optimizer to use.
            num_samples: the number of samples to draw.
            loss: the loss function to use.
            guide: the guide to use.
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            plot_diagnostics: plot the loss during VI.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """
        bayesian_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        if guide is None:
            guide = AutoMultivariateNormal(bayesian_model)

        svi = SVI(bayesian_model, guide, optimizer, loss=loss)

        keys = random.split(random.PRNGKey(rng_key), 2)
        svi_result = svi.run(keys[0], num_steps)
        params = svi_result.params

        if plot_diagnostics:
            plt.plot(svi_result.losses)
            plt.xlabel("Steps")
            plt.ylabel("ELBO loss")
            plt.semilogy()

        predictive = Predictive(guide, params=params, num_samples=num_samples)
        posterior = predictive(keys[1])

        inference_data = self.build_inference_data(
            posterior, num_chains=1, use_transformed_model=use_transformed_model
        )

        return FitResult(
            self,
            inference_data,
            background_model=self.background_model,
        )


class NSFitter(BayesianModelFitter):
    r"""
    Fit a model using the Nested Sampling algorithm via
    [`jaxns`](https://jaxns.readthedocs.io/en/latest/) through the
    [`numpyro.contrib.nested_sampling.NestedSampler`][] wrapper.

    Because `jaxns` requires sample-site names to be valid Python identifiers,
    this fitter transparently renames sites that contain characters such as
    ``/`` or ``~`` (e.g. ``mod/~/powerlaw_1_alpha``) before passing the model
    to the nested sampler, and maps them back afterwards.

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

        bayesian_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        # --- Discover all site names that need renaming ----------------------
        all_site_names = _collect_site_names(bayesian_model)
        names_needing_rename = [n for n in all_site_names if n != _sanitise_name(n)]

        if names_needing_rename:
            original_to_safe, safe_to_original = _build_name_mapping(all_site_names)
            wrapped_model = _RenameSites(bayesian_model, original_to_safe)
        else:
            wrapped_model = bayesian_model
            safe_to_original = {}

        # --- Run the nested sampler ------------------------------------------
        keys = random.split(random.PRNGKey(rng_key), 4)

        default_constructor = dict(
            verbose=verbose,
            difficult_model=True,
            max_samples=1e5,
            parameter_estimation=True,
            gradient_guided=False,
            devices=jax.devices(),
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

        # --- Rename samples back to original names ---------------------------
        if safe_to_original:
            posterior = {safe_to_original.get(k, k): v for k, v in posterior.items()}

        inference_data = self.build_inference_data(
            posterior, num_chains=1, use_transformed_model=use_transformed_model
        )

        # --- Store nested sampling evidence in the InferenceData attrs -------
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


def _collect_site_names(model) -> list[str]:
    """Trace a numpyro model and return the names of all sample / plate sites."""
    from numpyro.handlers import seed, trace

    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
    return [site["name"] for site in model_trace.values() if site["type"] in ("sample", "plate")]
