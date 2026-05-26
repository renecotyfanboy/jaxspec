import warnings

from typing import Literal

import jax

from jax import random
from numpyro.infer import AIES, ESS, MCMC, NUTS

from ...analysis.results import FitResult
from ._base import BayesianModelFitter


class MCMCFitter(BayesianModelFitter):
    """Fit a spectral model via MCMC sampling (NUTS, AIES, or ESS).

    Inherits from :class:`BayesianModel` and accepts the same constructor
    arguments (spectral model, prior dict with dotted-path keys, observations,
    optional background/instrument models).
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
        kernel_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
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

        kernel_kwargs: dict = kernel_kwargs or {}
        mcmc_kwargs: dict = mcmc_kwargs or {}

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        chain_kwargs = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        kernel = self.kernel_dict[sampler](numpyro_model, **kernel_kwargs)

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

        return FitResult(self, inference_data)
