import jax

from jax import random
from numpyro.contrib.nested_sampling import NestedSampler

from ..analysis.results import FitResult
from ..fit._fitter import BayesianModelFitter


class NSFitter(BayesianModelFitter):
    r"""
    A class to fit a model to a given set of observation using the Nested Sampling algorithm. This class uses the
    [`DefaultNestedSampler`][jaxns.DefaultNestedSampler] from [`jaxns`](https://jaxns.readthedocs.io/en/latest/) which
    implements the [Phantom-Powered Nested Sampling](https://arxiv.org/abs/2312.11330) algorithm.

    !!! info
        Ensure large prior volume is covered by the prior distributions to ensure the algorithm yield proper results.

    """

    def fit(
        self,
        rng_key: int = 0,
        num_samples: int = 1000,
        num_live_points: int = 1000,
        plot_diagnostics=False,
        termination_kwargs: dict | None = None,
        verbose=True,
        use_transformed_model: bool = True,
    ) -> FitResult:
        """
        Fit the model to the data using the Phantom-Powered nested sampling algorithm.

        Parameters:
            rng_key: the random key used to initialize the sampler.
            num_samples: the number of samples to draw.
            num_live_points: the number of live points to use at the start of the NS algorithm.
            plot_diagnostics: whether to plot the diagnostics of the NS algorithm.
            termination_kwargs: additional arguments to pass to the termination criterion of the NS algorithm.
            verbose: whether to print the progress of the NS algorithm.

        Returns:
            A [`FitResult`][jaxspec.analysis.results.FitResult] instance containing the results of the fit.
        """

        bayesian_model = self.transformed_numpyro_model
        keys = random.split(random.PRNGKey(rng_key), 4)

        ns = NestedSampler(
            bayesian_model,
            constructor_kwargs=dict(
                verbose=verbose,
                difficult_model=True,
                max_samples=1e5,
                parameter_estimation=True,
                gradient_guided=True,
                devices=jax.devices(),
                # init_efficiency_threshold=0.01,
                num_live_points=num_live_points,
            ),
            termination_kwargs=termination_kwargs if termination_kwargs else dict(),
        )

        ns.run(keys[0])

        if plot_diagnostics:
            ns.diagnostics()

        posterior = ns.get_samples(keys[1], num_samples=num_samples)
        inference_data = self.build_inference_data(
            posterior, num_chains=1, use_transformed_model=use_transformed_model
        )

        return FitResult(
            self,
            inference_data,
            background_model=self.background_model,
        )
