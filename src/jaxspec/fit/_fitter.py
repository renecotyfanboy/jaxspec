import warnings

from abc import ABC, abstractmethod
from typing import Literal

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpyro

from jax import random
from jax.random import PRNGKey
from numpyro.infer import AIES, ESS, MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal

from ..analysis.results import FitResult
from ._bayesian_model import BayesianModel


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

        seeded_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(numpyro_model, keys[3]),
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
            if self.background_model is not None:
                predictive_parameters.append(f"obs/~/{key}")
                predictive_parameters.append(f"bkg/~/{key}")
            #                predictive_parameters.append(f"ins/~/{key}")
            else:
                predictive_parameters.append(f"obs/~/{key}")
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
        optimizer=numpyro.optim.Adam(step_size=0.0005),
        loss=Trace_ELBO(),
        num_samples: int = 1000,
        guide=None,
        use_transformed_model: bool = True,
        plot_diagnostics: bool = False,
    ) -> FitResult:
        bayesian_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        if guide is None:
            guide = AutoMultivariateNormal(bayesian_model)

        svi = SVI(bayesian_model, guide, optimizer, loss=loss)

        keys = random.split(random.PRNGKey(rng_key), 3)
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
