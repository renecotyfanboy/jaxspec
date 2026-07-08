import matplotlib.pyplot as plt
import numpyro

from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal

from ...analysis.results import FitResult
from ._base import BayesianModelFitter


class VIFitter(BayesianModelFitter):
    def fit(
        self,
        rng_key: int = 0,
        num_steps: int = 10_000,
        optimizer: numpyro.optim._NumPyroOptim | None = None,
        loss: numpyro.infer.elbo.ELBO | None = None,
        num_samples: int = 1000,
        guide: numpyro.infer.autoguide.AutoGuide | None = None,
        use_transformed_model: bool = True,
        plot_diagnostics: bool = False,
        svi_kwargs: dict | None = None,
    ) -> FitResult:
        """
        Fit the model to the data using a variational inference approach from numpyro.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_steps: the number of steps for VI.
            optimizer: the optimizer to use. Defaults to `Adam(step_size=0.0005)`.
            num_samples: the number of samples to draw.
            loss: the loss function to use. Defaults to `Trace_ELBO()`.
            guide: the guide to use. Defaults to an `AutoMultivariateNormal` guide.
            use_transformed_model: whether to use the transformed model to build the InferenceData.
            plot_diagnostics: plot the loss during VI.
            svi_kwargs: additional arguments to pass to the SVI runner. See [`SVI.run`][numpyro.infer.svi.SVI.run] for more details.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        svi_kwargs: dict = svi_kwargs or {}
        optimizer = numpyro.optim.Adam(step_size=0.0005) if optimizer is None else optimizer
        loss = Trace_ELBO() if loss is None else loss

        numpyro_model = (
            self.transformed_numpyro_model if use_transformed_model else self.numpyro_model
        )

        if guide is None:
            guide = AutoMultivariateNormal(numpyro_model)

        svi = SVI(numpyro_model, guide, optimizer, loss=loss)

        keys = random.split(random.PRNGKey(rng_key), 2)
        svi_result = svi.run(keys[0], num_steps, **svi_kwargs)
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

        return FitResult(self, inference_data)
