from abc import ABC, abstractmethod

import arviz as az
import jax
import numpyro

from jax import Array, random
from jax.numpy import concatenate
from numpyro.infer import Predictive

from ...analysis.results import FitResult
from .._bayesian_model import BayesianModel


class BayesianModelFitter(BayesianModel, ABC):
    def build_inference_data(
        self,
        posterior_samples,
        num_chains: int = 1,
        num_predictive_samples: int = 1000,
        key: Array = random.key(42),
        use_transformed_model: bool = False,
        filter_inference_data: bool = True,
        consolidate_samples: bool = True,
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
            consolidate_samples: whether to gather the (possibly device-sharded) posterior samples onto a single device before post-processing. Multi-chain MCMC shards samples across all devices, which inflates peak memory during the predictive/log-likelihood passes; gathering is a no-op on a single device and does not change the results.
        """

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        # Multi-chain MCMC returns samples sharded across every device (num_chains defaults to
        # len(jax.devices())). The post-processing passes below do not benefit from that
        # sharding, and mapping over a sharded leading axis inflates peak memory (XLA
        # materializes per-device temporaries). Gather onto a single device — a no-op when
        # already single-device, and numerically identical either way.
        if consolidate_samples:
            posterior_samples = jax.device_put(posterior_samples, jax.devices()[0])

        keys = random.split(key, 3)

        posterior_predictive = Predictive(numpyro_model, posterior_samples)(keys[0], observed=False)

        prior = Predictive(numpyro_model, num_samples=num_predictive_samples * num_chains)(
            keys[1], observed=False
        )

        log_likelihood = numpyro.infer.log_likelihood(numpyro_model, posterior_samples)
        if len(log_likelihood.keys()) > 1:
            log_likelihood["full"] = concatenate([ll for _, ll in log_likelihood.items()], axis=1)
            log_likelihood["observed.all"] = concatenate(
                [ll for k, ll in log_likelihood.items() if k.startswith("observed.")], axis=1
            )

            # TODO : should we really track the likelihood on the background model?
            has_stochastic_bg = any(
                bg.is_stochastic for bg in self.forward_model.background.values()
            )
            if has_stochastic_bg:
                log_likelihood["observed_background.all"] = concatenate(
                    [
                        ll
                        for k, ll in log_likelihood.items()
                        if k.startswith("observed_background.")
                    ],
                    axis=1,
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
            return arr.reshape(new_shape)

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

        - Removes private parameters (e.g. starting with "_")s
        - Removes parameters build from reparametrised variables (e.g. ending with `"_base"`)
        """

        predictive_parameters = []

        for key in self.forward_model.observations:
            predictive_parameters.append(f"observed.{key}")
            if self.forward_model.background.get(key) is not None:
                predictive_parameters.append(f"observed_background.{key}")

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
