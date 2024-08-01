from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import arviz as az
import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from astropy.cosmology import Cosmology, Planck18
from astropy.units import Unit
from chainconsumer import Chain, ChainConsumer, PlotConfig
from haiku.data_structures import traverse
from jax.typing import ArrayLike
from numpyro.handlers import seed
from scipy.integrate import trapezoid
from scipy.special import gammaln
from scipy.stats import nbinom

if TYPE_CHECKING:
    from ..fit import BayesianModel
    from ..model.background import BackgroundModel

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class HaikuDict(dict[str, dict[str, T]]): ...


def _plot_binned_samples_with_error(
    ax: plt.Axes,
    x_bins: ArrayLike,
    denominator: ArrayLike | None = None,
    y_samples: ArrayLike | None = None,
    color=(0.15, 0.25, 0.45),
    percentile: tuple = (16, 84),
):
    """
    Helper function to plot the posterior predictive distribution of the model. The function
    computes the percentiles of the posterior predictive distribution and plot them as a shaded
    area. If the observed data is provided, it is also plotted as a step function.

    Parameters
    ----------
        x_bins: The bin edges of the data (2 x N).
        y_samples: The samples of the posterior predictive distribution (Samples X N).
        denominator: Values used to divided the samples, i.e. to get energy flux (N).
        ax: The matplotlib axes object.
        color: The color of the posterior predictive distribution.
        y_observed: The observed data (N).
        label: The label of the observed data.
        percentile: The percentile of the posterior predictive distribution to plot.
    """

    mean, envelope = None, None

    if denominator is None:
        denominator = np.ones_like(x_bins[0])

    mean = ax.stairs(
        list(np.median(y_samples, axis=0) / denominator),
        edges=[*list(x_bins[0]), x_bins[1][-1]],
        color=color,
        alpha=0.7,
    )

    if y_samples is not None:
        if denominator is None:
            denominator = np.ones_like(x_bins[0])

        percentiles = np.percentile(y_samples, percentile, axis=0)

        # The legend cannot handle fill_between, so we pass a fill to get a fancy icon
        (envelope,) = ax.fill(np.nan, np.nan, alpha=0.3, facecolor=color)

        ax.stairs(
            percentiles[1] / denominator,
            edges=[*list(x_bins[0]), x_bins[1][-1]],
            baseline=percentiles[0] / denominator,
            alpha=0.3,
            fill=True,
            color=color,
        )

    return [(mean, envelope)]


class FitResult:
    """
    Container for the result of a fit using any ModelFitter class.
    """

    # TODO : Add type hints
    def __init__(
        self,
        bayesian_fitter: BayesianModel,
        inference_data: az.InferenceData,
        structure: Mapping[K, V],
        background_model: BackgroundModel = None,
    ):
        self.model = bayesian_fitter.model
        self.bayesian_fitter = bayesian_fitter
        self.inference_data = inference_data
        self.obsconfs = bayesian_fitter.observation_container
        self.background_model = background_model
        self._structure = structure

        # Add the model used in fit to the metadata
        for group in self.inference_data.groups():
            group_name = group.split("/")[-1]
            metadata = getattr(self.inference_data, group_name).attrs
            metadata["model"] = str(self.model)
            # TODO : Store metadata about observations used in the fitting process

    @property
    def converged(self) -> bool:
        r"""
        Convergence of the chain as computed by the $\hat{R}$ statistic.
        """

        return all(az.rhat(self.inference_data) < 1.01)

    @property
    def _structured_samples(self):
        """
        Get samples from the parameter posterior distribution but keep their shape in terms of draw and chains.
        """

        samples_flat = self._structured_samples_flat

        samples_haiku = {}

        for module, parameter, value in traverse(self._structure):
            if samples_haiku.get(module, None) is None:
                samples_haiku[module] = {}
            samples_haiku[module][parameter] = samples_flat[f"{module}_{parameter}"]

        return samples_haiku

    @property
    def _structured_samples_flat(self):
        """
        Get samples from the parameter posterior distribution but keep their shape in terms of draw and chains.
        """

        var_names = [f"{m}_{n}" for m, n, _ in traverse(self._structure)]
        posterior = az.extract(self.inference_data, var_names=var_names, combined=False)
        samples_flat = {key: posterior[key].data for key in var_names}

        return samples_flat

    @property
    def input_parameters(self) -> HaikuDict[ArrayLike]:
        """
        The input parameters of the model.
        """

        posterior = az.extract(self.inference_data, combined=False)

        samples_shape = (len(posterior.coords["chain"]), len(posterior.coords["draw"]))

        total_shape = tuple(posterior.sizes[d] for d in posterior.coords)

        posterior = {key: posterior[key].data for key in posterior.data_vars}

        with seed(rng_seed=0):
            input_parameters = self.bayesian_fitter.prior_distributions_func()

        for module, parameter, value in traverse(input_parameters):
            if f"{module}_{parameter}" in posterior.keys():
                # We add as extra dimension as there might be different values per observation
                if posterior[f"{module}_{parameter}"].shape == samples_shape:
                    to_set = posterior[f"{module}_{parameter}"][..., None]
                else:
                    to_set = posterior[f"{module}_{parameter}"]

                input_parameters[module][parameter] = to_set

            else:
                # The parameter is fixed in this case, so we just broadcast is over chain and draws
                input_parameters[module][parameter] = value[None, None, ...]

            if len(total_shape) < len(input_parameters[module][parameter].shape):
                # If there are only chains and draws, we reduce
                input_parameters[module][parameter] = input_parameters[module][parameter][..., 0]

            else:
                input_parameters[module][parameter] = jnp.broadcast_to(
                    input_parameters[module][parameter], total_shape
                )

        return input_parameters

    def photon_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.photon / u.cm**2 / u.s,
        register: bool = False,
    ) -> ArrayLike:
        """
        Compute the unfolded photon flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the photon flux.
            register: Whether to register the flux with the other posterior parameters.

        !!! warning
            Computation of the folded flux is not implemented yet. Feel free to open an
            [issue](https://github.com/renecotyfanboy/jaxspec/issues) in the GitHub repository.
        """

        @jax.jit
        @jnp.vectorize
        def vectorized_flux(*pars):
            parameters_pytree = jax.tree.unflatten(pytree_def, pars)
            return self.model.photon_flux(
                parameters_pytree, jnp.asarray([e_min]), jnp.asarray([e_max]), n_points=100
            )[0]

        flat_tree, pytree_def = jax.tree.flatten(self.input_parameters)
        flux = vectorized_flux(*flat_tree)
        conversion_factor = (u.photon / u.cm**2 / u.s).to(unit)
        value = flux * conversion_factor

        if register:
            self.inference_data.posterior[f"photon_flux_{e_min:.1f}_{e_max:.1f}"] = (
                list(self.inference_data.posterior.coords),
                value,
            )

        return value

    def energy_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.erg / u.cm**2 / u.s,
        register: bool = False,
    ) -> ArrayLike:
        """
        Compute the unfolded energy flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the energy flux.
            register: Whether to register the flux with the other posterior parameters.

        !!! warning
            Computation of the folded flux is not implemented yet. Feel free to open an
            [issue](https://github.com/renecotyfanboy/jaxspec/issues) in the GitHub repository.
        """

        @jax.jit
        @jnp.vectorize
        def vectorized_flux(*pars):
            parameters_pytree = jax.tree.unflatten(pytree_def, pars)
            return self.model.energy_flux(
                parameters_pytree, jnp.asarray([e_min]), jnp.asarray([e_max]), n_points=100
            )[0]

        flat_tree, pytree_def = jax.tree.flatten(self.input_parameters)
        flux = vectorized_flux(*flat_tree)
        conversion_factor = (u.keV / u.cm**2 / u.s).to(unit)
        value = flux * conversion_factor

        if register:
            self.inference_data.posterior[f"energy_flux_{e_min:.1f}_{e_max:.1f}"] = (
                list(self.inference_data.posterior.coords),
                value,
            )

        return value

    def luminosity(
        self,
        e_min: float,
        e_max: float,
        redshift: float | ArrayLike = 0.1,
        observer_frame: bool = True,
        cosmology: Cosmology = Planck18,
        unit: Unit = u.erg / u.s,
        register: bool = False,
    ) -> ArrayLike:
        """
        Compute the luminosity of the source specifying its redshift. The luminosity is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band.
            e_max: The upper bound of the energy band.
            redshift: The redshift of the source. It can be a distribution of redshifts.
            observer_frame: Whether the input bands are defined in observer frame or not.
            cosmology: Chosen cosmology.
            unit: The unit of the luminosity.
            register: Whether to register the flux with the other posterior parameters.
        """

        if not observer_frame:
            raise NotImplementedError()

        @jax.jit
        @jnp.vectorize
        def vectorized_flux(*pars):
            parameters_pytree = jax.tree.unflatten(pytree_def, pars)
            return self.model.energy_flux(
                parameters_pytree,
                jnp.asarray([e_min]) * (1 + redshift),
                jnp.asarray([e_max]) * (1 + redshift),
                n_points=100,
            )[0]

        flat_tree, pytree_def = jax.tree.flatten(self.input_parameters)
        flux = vectorized_flux(*flat_tree) * (u.keV / u.cm**2 / u.s)
        value = (flux * (4 * np.pi * cosmology.luminosity_distance(redshift) ** 2)).to(unit)

        if register:
            self.inference_data.posterior[f"luminosity_{e_min:.1f}_{e_max:.1f}"] = (
                list(self.inference_data.posterior.coords),
                value,
            )

        return value

    def to_chain(self, name: str, parameters_type: Literal["model", "bkg"] = "model") -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters_type.

        Parameters:
            name: The name of the chain.
            parameters_type: The parameters_type to include in the chain.
        """

        obs_id = self.inference_data.copy()

        if parameters_type == "model":
            keys_to_drop = [
                key
                for key in obs_id.posterior.keys()
                if (key.startswith("_") or key.startswith("bkg"))
            ]
        elif parameters_type == "bkg":
            keys_to_drop = [key for key in obs_id.posterior.keys() if not key.startswith("bkg")]
        else:
            raise ValueError(f"Unknown value for parameters_type: {parameters_type}")

        obs_id.posterior = obs_id.posterior.drop_vars(keys_to_drop)
        chain = Chain.from_arviz(obs_id, name)

        """
        chain.samples.columns = [
            format_parameters(parameter) for parameter in chain.samples.columns
        ]
        """

        return chain

    @property
    def samples_haiku(self) -> HaikuDict[ArrayLike]:
        """
        Haiku-like structure for the samples e.g.

        ```
        {
            'powerlaw_1' :
            {
                'alpha': ...,
                'amplitude': ...
            },

            'blackbody_1':
            {
                'kT': ...,
                'norm': ...
            },

            'tbabs_1':
            {
                'nH': ...
            }
        }
        ```

        """

        params = {}

        for module, parameter, value in traverse(self._structure):
            if params.get(module, None) is None:
                params[module] = {}
            params[module][parameter] = self.samples_flat[f"{module}_{parameter}"]

        return params

    @property
    def samples_flat(self) -> dict[str, ArrayLike]:
        """
        Flat structure for the samples e.g.

        ```
        {
            'powerlaw_1_alpha': ...,
            'powerlaw_1_amplitude': ...,
            'blackbody_1_kT': ...,
            'blackbody_1_norm': ...,
            'tbabs_1_nH': ...,
        }
        ```
        """
        var_names = [f"{m}_{n}" for m, n, _ in traverse(self._structure)]
        posterior = az.extract(self.inference_data, var_names=var_names)
        return {key: posterior[key].data for key in var_names}

    @property
    def log_likelihood(self) -> xr.Dataset:
        """
        Return the log_likelihood of each observation
        """
        log_likelihood = az.extract(self.inference_data, group="log_likelihood")
        dimensions_to_reduce = [
            coord for coord in log_likelihood.coords if coord not in ["sample", "draw", "chain"]
        ]
        return log_likelihood.sum(dimensions_to_reduce)

    @property
    def c_stat(self):
        r"""
        Return the C-statistic of the model

        The C-statistic is defined as:

        $$ C = 2 \sum_{i} M - D*log(M) + D*log(D) - D $$
        or
        $$ C = 2 \sum_{i} M - D*log(M)$$
        for bins with no counts

        """

        exclude_dims = ["chain", "draw", "sample"]
        all_dims = list(self.inference_data.log_likelihood.dims)
        reduce_dims = [dim for dim in all_dims if dim not in exclude_dims]
        data = self.inference_data.observed_data
        c_stat = -2 * (
            self.log_likelihood
            + (gammaln(data + 1) - (xr.where(data > 0, data * (np.log(data) - 1), 0))).sum(
                dim=reduce_dims
            )
        )

        return c_stat

    def plot_ppc(
        self,
        percentile: tuple[int, int] = (16, 84),
        x_unit: str | u.Unit = "keV",
        y_type: Literal[
            "counts", "countrate", "photon_flux", "photon_flux_density"
        ] = "photon_flux_density",
    ) -> plt.Figure:
        r"""
        Plot the posterior predictive distribution of the model. It also features a residual plot, defined using the
        following formula:

        $$ \text{Residual} = \frac{\text{Observed counts} - \text{Posterior counts}}
        {(\text{Posterior counts})_{84\%}-(\text{Posterior counts})_{16\%}} $$

        Parameters:
            percentile: The percentile of the posterior predictive distribution to plot.
            x_unit: The units of the x-axis. It can be either a string (parsable by astropy.units) or an astropy unit. It must be homogeneous to either a length, a frequency or an energy.
            y_type: The type of the y-axis. It can be either "counts", "countrate", "photon_flux" or "photon_flux_density".

        Returns:
            The matplotlib figure.
        """

        obsconf_container = self.obsconfs
        x_unit = u.Unit(x_unit)

        match y_type:
            case "counts":
                y_units = u.photon
            case "countrate":
                y_units = u.photon / u.s
            case "photon_flux":
                y_units = u.photon / u.cm**2 / u.s
            case "photon_flux_density":
                y_units = u.photon / u.cm**2 / u.s / x_unit
            case _:
                raise ValueError(
                    f"Unknown y_type: {y_type}. Must be 'counts', 'countrate', 'photon_flux' or 'photon_flux_density'"
                )

        color = (0.15, 0.25, 0.45)

        with plt.style.context("default"):
            # Note to Simon : do not change xbins[1] - xbins[0] to
            # np.diff, you already did this twice and forgot that it does not work since diff keeps the dimensions
            # and enable weird broadcasting that makes the plot fail

            fig, axs = plt.subplots(
                2,
                len(obsconf_container),
                figsize=(6 * len(obsconf_container), 6),
                sharex=True,
                height_ratios=[0.7, 0.3],
            )

            plot_ylabels_once = True

            for name, obsconf, ax in zip(
                obsconf_container.keys(),
                obsconf_container.values(),
                axs.T if len(obsconf_container) > 1 else [axs],
            ):
                legend_plots = []
                legend_labels = []
                count = az.extract(
                    self.inference_data, var_names=f"obs_{name}", group="posterior_predictive"
                ).values.T
                bkg_count = (
                    None
                    if self.background_model is None
                    else az.extract(
                        self.inference_data, var_names=f"bkg_{name}", group="posterior_predictive"
                    ).values.T
                )

                xbins = obsconf.out_energies * u.keV
                xbins = xbins.to(x_unit, u.spectral())

                # This compute the total effective area within all bins
                # This is a bit weird since the following computation is equivalent to ignoring the RMF
                exposure = obsconf.exposure.data * u.s
                mid_bins_arf = obsconf.in_energies.mean(axis=0) * u.keV
                mid_bins_arf = mid_bins_arf.to(x_unit, u.spectral())
                e_grid = np.linspace(*xbins, 10)
                interpolated_arf = np.interp(e_grid, mid_bins_arf, obsconf.area)
                integrated_arf = (
                    trapezoid(interpolated_arf, x=e_grid, axis=0)
                    / (
                        np.abs(
                            xbins[1] - xbins[0]
                        )  # Must fold in abs because some units reverse the ordering of the bins
                    )
                    * u.cm**2
                )

                match y_type:
                    case "counts":
                        denominator = 1
                    case "countrate":
                        denominator = exposure
                    case "photon_flux":
                        denominator = integrated_arf * exposure
                    case "photon_flux_density":
                        denominator = (xbins[1] - xbins[0]) * integrated_arf * exposure

                y_samples = (count * u.photon / denominator).to(y_units)
                y_observed = (obsconf.folded_counts.data * u.photon / denominator).to(y_units)
                y_observed_low = (
                    nbinom.ppf(percentile[0] / 100, obsconf.folded_counts.data, 0.5)
                    * u.photon
                    / denominator
                ).to(y_units)
                y_observed_high = (
                    nbinom.ppf(percentile[1] / 100, obsconf.folded_counts.data, 0.5)
                    * u.photon
                    / denominator
                ).to(y_units)

                # Use the helper function to plot the data and posterior predictive
                legend_plots += _plot_binned_samples_with_error(
                    ax[0],
                    xbins.value,
                    y_samples=y_samples.value,
                    denominator=np.ones_like(y_observed).value,
                    color=color,
                    percentile=percentile,
                )

                legend_labels.append("Model")

                true_data_plot = ax[0].errorbar(
                    np.sqrt(xbins.value[0] * xbins.value[1]),
                    y_observed.value,
                    xerr=np.abs(xbins.value - np.sqrt(xbins.value[0] * xbins.value[1])),
                    yerr=[
                        y_observed.value - y_observed_low.value,
                        y_observed_high.value - y_observed.value,
                    ],
                    color="black",
                    linestyle="none",
                    alpha=0.3,
                    capsize=2,
                )

                legend_plots.append((true_data_plot,))
                legend_labels.append("Observed")

                if self.background_model is not None:
                    # We plot the background only if it is included in the fit, i.e. by subtracting
                    ratio = obsconf.folded_backratio.data
                    y_samples_bkg = (bkg_count * u.photon / (denominator * ratio)).to(y_units)
                    y_observed_bkg = (
                        obsconf.folded_background.data * u.photon / (denominator * ratio)
                    ).to(y_units)
                    legend_plots += _plot_binned_samples_with_error(
                        ax[0],
                        xbins.value,
                        y_samples=y_samples_bkg.value,
                        denominator=np.ones_like(y_observed).value,
                        color=(0.26787604, 0.60085972, 0.63302651),
                        percentile=percentile,
                    )

                    legend_labels.append("Model (bkg)")

                residual_samples = (obsconf.folded_counts.data - count) / np.diff(
                    np.percentile(count, percentile, axis=0), axis=0
                )

                residuals = np.percentile(
                    residual_samples,
                    percentile,
                    axis=0,
                )

                median_residuals = np.median(
                    residual_samples,
                    axis=0,
                )

                ax[1].stairs(
                    residuals[1],
                    edges=[*list(xbins.value[0]), xbins.value[1][-1]],
                    baseline=list(residuals[0]),
                    alpha=0.3,
                    facecolor=color,
                    fill=True,
                )

                ax[1].stairs(
                    median_residuals,
                    edges=[*list(xbins.value[0]), xbins.value[1][-1]],
                    color=color,
                    alpha=0.7,
                )

                max_residuals = np.max(np.abs(residuals))

                ax[0].loglog()
                ax[1].set_ylim(-max(3.5, max_residuals), +max(3.5, max_residuals))

                if plot_ylabels_once:
                    ax[0].set_ylabel(f"Folded spectrum\n [{y_units:latex_inline}]")
                    ax[1].set_ylabel("Residuals \n" + r"[$\sigma$]")
                    plot_ylabels_once = False

                match getattr(x_unit, "physical_type"):
                    case "length":
                        ax[1].set_xlabel(f"Wavelength \n[{x_unit:latex_inline}]")
                    case "energy":
                        ax[1].set_xlabel(f"Energy \n[{x_unit:latex_inline}]")
                    case "frequency":
                        ax[1].set_xlabel(f"Frequency \n[{x_unit:latex_inline}]")
                    case _:
                        RuntimeError(
                            f"Unknown physical type for x_units: {x_unit}. "
                            f"Must be 'length', 'energy' or 'frequency'"
                        )

                ax[1].axhline(0, color=color, ls="--")
                ax[1].axhline(-3, color=color, ls=":")
                ax[1].axhline(3, color=color, ls=":")

                # ax[1].set_xticks(xticks, labels=xticks)
                # ax[1].xaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(np.inf, np.inf)))
                ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
                ax[1].set_yticks(range(-3, 4), minor=True)

                ax[0].set_xlim(xbins.value.min(), xbins.value.max())

                ax[0].legend(legend_plots, legend_labels)
                fig.suptitle(self.model.to_string())
                fig.align_ylabels()
                plt.subplots_adjust(hspace=0.0)
                fig.tight_layout()

            return fig

    def table(self) -> str:
        r"""
        Return a formatted $\LaTeX$ table of the results of the fit.
        """

        consumer = ChainConsumer()
        consumer.add_chain(self.to_chain(self.model.to_string()))

        return consumer.analysis.get_latex_table(caption="Results of the fit", label="tab:results")

    def plot_corner(
        self,
        config: PlotConfig = PlotConfig(usetex=False, summarise=False, label_font_size=12),
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Plot the corner plot of the posterior distribution of the parameters_type. This method uses the ChainConsumer.

        Parameters:
            config: The configuration of the plot.
            **kwargs: Additional arguments passed to ChainConsumer.plotter.plot. Some useful parameters are :
                - columns : list of parameters to plot.
        """

        consumer = ChainConsumer()
        consumer.add_chain(self.to_chain(self.model.to_string()))
        consumer.set_plot_config(config)

        # Context for default mpl style
        with plt.style.context("default"):
            return consumer.plotter.plot(**kwargs)
