import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from ..data import ObsConfiguration
from ..model.abc import SpectralModel
from ..model.background import BackgroundModel
from collections.abc import Mapping
from typing import TypeVar, Tuple, Literal
from astropy.cosmology import Cosmology, Planck18
import astropy.units as u
from astropy.units import Unit
from haiku.data_structures import traverse
from chainconsumer import Chain, PlotConfig, ChainConsumer
import jax
from jax.typing import ArrayLike
from scipy.integrate import trapezoid

K = TypeVar("K")
V = TypeVar("V")


def _plot_binned_samples_with_error(
    ax: plt.Axes,
    x_bins: ArrayLike,
    denominator: ArrayLike | None = None,
    y_samples: ArrayLike | None = None,
    y_observed: ArrayLike | None = None,
    color=(0.15, 0.25, 0.45),
    percentile: tuple = (16, 84),
):
    """
    Helper function to plot the posterior predictive distribution of the model. The function
    computes the percentiles of the posterior predictive distribution and plot them as a shaded
    area. If the observed data is provided, it is also plotted as a step function.

    Parameters:
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

    if x_bins is None:
        raise ValueError("x_bins cannot be None.")

    if (y_samples is None) and (y_observed is None):
        raise ValueError("Either a y_samples or y_observed must be provided.")

    if y_observed is not None:
        if denominator is None:
            denominator = np.ones_like(x_bins[0])

        (mean,) = ax.step(
            list(x_bins[0]) + [x_bins[1][-1]],  # x_bins[1][-1]+1],
            list(y_observed / denominator) + [np.nan],  # + [np.nan, np.nan],
            where="pre",
            c=color,
        )

    if y_samples is not None:
        if denominator is None:
            denominator = np.ones_like(x_bins[0])

        percentiles = np.percentile(y_samples, percentile, axis=0)

        # The legend cannot handle fill_between, so we pass a fill to get a fancy icon
        (envelope,) = ax.fill(np.nan, np.nan, alpha=0.3, facecolor=color)

        ax.fill_between(
            list(x_bins[0]) + [x_bins[1][-1]],  # + [x_bins[1][-1], x_bins[1][-1] + 1],
            list(percentiles[0] / denominator) + [np.nan],  # + [np.nan, np.nan],
            list(percentiles[1] / denominator) + [np.nan],  # + [np.nan, np.nan],
            alpha=0.3,
            step="pre",
            facecolor=color,
        )

    return [(mean, envelope)]


def format_parameters(parameter_name):
    computed_parameters = ["Photon flux", "Energy flux", "Luminosity"]

    if parameter_name == "weight":
        # ChainConsumer add a weight column to the samples
        return parameter_name

    for parameter in computed_parameters:
        if parameter in parameter_name:
            return parameter_name

    # Find second occurrence of the character '_'
    first_occurrence = parameter_name.find("_")
    second_occurrence = parameter_name.find("_", first_occurrence + 1)
    module = parameter_name[:second_occurrence]
    parameter = parameter_name[second_occurrence + 1 :]

    name, number = module.split("_")
    module = rf"[{name.capitalize()} ({number})]"

    if parameter == "norm":
        return r"Norm " + module

    else:
        return rf"${parameter}$" + module


class ChainResult:
    """
    This class is the container for the result of a fit using the BayesianModel class.
    """

    # TODO : Add type hints
    def __init__(
        self,
        model: SpectralModel,
        obsconf: ObsConfiguration | dict[str, ObsConfiguration],
        inference_data: az.InferenceData,
        structure: Mapping[K, V],
        background_model: BackgroundModel = None,
    ):
        self.model = model
        self._structure = structure
        self.inference_data = inference_data
        self.obsconfs = {"Observation": obsconf} if isinstance(obsconf, ObsConfiguration) else obsconf
        self.background_model = background_model
        self._structure = structure

        # Add the model used in fit to the metadata
        for group in self.inference_data.groups():
            group_name = group.split("/")[-1]
            metadata = getattr(self.inference_data, group_name).attrs
            metadata["model"] = str(model)
            # TODO : Store metadata about observations used in the fitting process

    @property
    def converged(self):
        """
        Convergence of the chain as computed by the $\hat{R}$ statistic.
        """

        return all(az.rhat(self.inference_data) < 1.01)

    def photon_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.photon / u.cm**2 / u.s,
    ) -> ArrayLike:
        """
        Compute the unfolded photon flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the photon flux.

        !!! warning
            Computation of the folded flux is not implemented yet. Feel free to open an
            [issue](https://github.com/renecotyfanboy/jaxspec/issues) in the GitHub repository.
        """

        flux = jax.vmap(lambda p: self.model.photon_flux(p, np.asarray([e_min]), np.asarray([e_max])))(self.params)

        conversion_factor = (u.photon / u.cm**2 / u.s).to(unit)

        value = flux * conversion_factor

        self.samples[rf"Photon flux ({e_min:.1f}-{e_max:.1f} keV)"] = value

        return value

    def energy_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.erg / u.cm**2 / u.s,
    ) -> ArrayLike:
        """
        Compute the unfolded energy flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the energy flux.

        !!! warning
            Computation of the folded flux is not implemented yet. Feel free to open an
            [issue](https://github.com/renecotyfanboy/jaxspec/issues) in the GitHub repository.
        """

        flux = jax.vmap(lambda p: self.model.energy_flux(p, np.asarray([e_min]), np.asarray([e_max])))(self.params)

        conversion_factor = (u.keV / u.cm**2 / u.s).to(unit)

        value = flux * conversion_factor

        self.samples[rf"Energy flux ({e_min:.1f}-{e_max:.1f} keV)"] = value

        return value

    def luminosity(
        self,
        e_min: float,
        e_max: float,
        redshift: float | ArrayLike = 0,
        observer_frame: bool = True,
        cosmology: Cosmology = Planck18,
        unit: Unit = u.erg / u.s,
    ):
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
        """

        if not observer_frame:
            raise NotImplementedError()

        flux = self.energy_flux(e_min * (1 + redshift), e_max * (1 + redshift)) * (u.erg / u.cm**2 / u.s)

        value = (flux * (4 * np.pi * cosmology.luminosity_distance(redshift) ** 2)).to(unit)

        self.samples[rf"Luminosity ({e_min:.1f}-{e_max:.1f} keV)"] = value

        return value

    def to_chain(self, name: str, parameters: Literal["model", "bkg"] = "model") -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters.

        Parameters:
            name: The name of the chain.
            parameters: The parameters to include in the chain.
        """

        obs_id = self.inference_data.copy()

        if parameters == "model":
            keys_to_drop = [key for key in obs_id.posterior.keys() if (key.startswith("_") or key.startswith("bkg"))]
        elif parameters == "bkg":
            keys_to_drop = [key for key in obs_id.posterior.keys() if not key.startswith("bkg")]
        else:
            raise ValueError(f"Unknown value for parameters: {parameters}")

        obs_id.posterior = obs_id.posterior.drop_vars(keys_to_drop)
        chain = Chain.from_arviz(obs_id, name)
        chain.samples.columns = [format_parameters(parameter) for parameter in chain.samples.columns]

        return chain

    @property
    def samples_haiku(self):
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
            params[module][parameter] = self.samples[f"{module}_{parameter}"]

        return params

    @property
    def samples_flat(self):
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

    def plot_ppc(
        self,
        percentile: Tuple[int, int] = (14, 86),
        x_units: str | u.Unit = "keV",
        y_type: Literal["counts", "countrate", "photon_flux", "photon_flux_density"] = "photon_flux_density",
    ) -> plt.Figure:
        r"""
        Plot the posterior predictive distribution of the model. It also features a residual plot, defined using the
        following formula:

        $$ \text{Residual} = \frac{\text{Observed counts} - \text{Posterior counts}}
        {(\text{Posterior counts})_{84\%}-(\text{Posterior counts})_{16\%}} $$

        Parameters:
            percentile: The percentile of the posterior predictive distribution to plot.
            x_units: The units of the x-axis. It can be either a string (parsable by astropy.units) or an astropy unit.
            It must be homogeneous to either a length, a frequency or an energy.
            y_type: The type of the y-axis. It can be either "counts", "countrate", "photon_flux" or "photon_flux_density".

        Returns:
            The matplotlib figure.
        """

        obsconf_container = self.obsconfs
        x_units = u.Unit(x_units)

        match y_type:
            case "counts":
                y_units = u.photon
            case "countrate":
                y_units = u.photon / u.s
            case "photon_flux":
                y_units = u.photon / u.cm**2 / u.s
            case "photon_flux_density":
                y_units = u.photon / u.cm**2 / u.s / x_units
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
                2, len(obsconf_container), figsize=(6 * len(obsconf_container), 6), sharex=True, height_ratios=[0.7, 0.3]
            )

            plot_ylabels_once = True

            for name, obsconf, ax in zip(
                obsconf_container.keys(), obsconf_container.values(), axs.T if len(obsconf_container) > 1 else [axs]
            ):
                legend_plots = []
                legend_labels = []
                count = az.extract(self.inference_data, var_names=f"{name}_obs", group="posterior_predictive").values.T
                bkg_count = (
                    None
                    if self.background_model is None
                    else az.extract(self.inference_data, var_names=f"{name}_bkg", group="posterior_predictive").values.T
                )

                xbins = obsconf.out_energies * u.keV
                xbins = xbins.to(x_units, u.spectral())

                # This compute the total effective area within all bins
                # This is a bit weird since the following computation is equivalent to ignoring the RMF
                exposure = obsconf.exposure.data * u.s
                mid_bins_arf = obsconf.in_energies.mean(axis=0) * u.keV
                mid_bins_arf = mid_bins_arf.to(x_units, u.spectral())
                e_grid = np.linspace(*xbins, 10)
                interpolated_arf = np.interp(e_grid, mid_bins_arf, obsconf.area)
                integrated_arf = (
                    trapezoid(interpolated_arf, x=e_grid, axis=0)
                    / (
                        np.abs(xbins[1] - xbins[0])  # Must fold in abs because some units reverse the ordering of the bins
                    )
                    * u.cm**2
                )

                """
                if xbins[0][0] < 1 < xbins[1][-1]:
                    xticks = [np.floor(xbins[0][0] * 10) / 10, 1.0, np.floor(xbins[1][-1])]
                else:
                    xticks = [np.floor(xbins[0][0] * 10) / 10, np.floor(xbins[1][-1])]
                """

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

                # Use the helper function to plot the data and posterior predictive
                legend_plots += _plot_binned_samples_with_error(
                    ax[0],
                    xbins.value,
                    y_samples=y_samples.value,
                    y_observed=y_observed.value,
                    denominator=np.ones_like(y_observed).value,
                    color=color,
                    percentile=percentile,
                )

                legend_labels.append("Source + Background")

                if self.background_model is not None:
                    # We plot the background only if it is included in the fit, i.e. by subtracting
                    ratio = obsconf.folded_backratio.data
                    y_samples_bkg = (bkg_count * u.photon / (denominator * ratio)).to(y_units)
                    y_observed_bkg = (obsconf.folded_background.data * u.photon / (denominator * ratio)).to(y_units)
                    legend_plots += _plot_binned_samples_with_error(
                        ax[0],
                        xbins.value,
                        y_samples=y_samples_bkg.value,
                        y_observed=y_observed_bkg.value,
                        denominator=np.ones_like(y_observed).value,
                        color=(0.26787604, 0.60085972, 0.63302651),
                        percentile=percentile,
                    )

                    legend_labels.append("Background")

                residuals = np.percentile(
                    (obsconf.folded_counts.data - count) / np.diff(np.percentile(count, percentile, axis=0), axis=0),
                    percentile,
                    axis=0,
                )

                ax[1].fill_between(
                    list(xbins.value[0]) + [xbins.value[1][-1]],
                    list(residuals[0]) + [residuals[0][-1]],
                    list(residuals[1]) + [residuals[1][-1]],
                    alpha=0.3,
                    step="post",
                    facecolor=color,
                )

                max_residuals = np.max(np.abs(residuals))

                ax[0].loglog()
                ax[1].set_ylim(-max(3.5, max_residuals), +max(3.5, max_residuals))

                if plot_ylabels_once:
                    ax[0].set_ylabel(f"Folded spectrum\n [{y_units:latex_inline}]")
                    ax[1].set_ylabel("Residuals \n" + r"[$\sigma$]")
                    plot_ylabels_once = False

                match getattr(x_units, "physical_type"):
                    case "length":
                        ax[1].set_xlabel(f"Wavelength \n[{x_units:latex_inline}]")
                    case "energy":
                        ax[1].set_xlabel(f"Energy \n[{x_units:latex_inline}]")
                    case "frequency":
                        ax[1].set_xlabel(f"Frequency \n[{x_units:latex_inline}]")
                    case _:
                        RuntimeError(
                            f"Unknown physical type for x_units: {x_units}. " f"Must be 'length', 'energy' or 'frequency'"
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
        config: PlotConfig = PlotConfig(usetex=False, summarise=False, label_font_size=6),
        style="default",
        **kwargs,
    ):
        """
        Plot the corner plot of the posterior distribution of the parameters. This method uses the ChainConsumer.

        Parameters:
            config: The configuration of the plot.
            style: The matplotlib style of the plot.
            **kwargs: Additional arguments passed to ChainConsumer.plotter.plot.
        """

        consumer = ChainConsumer()
        consumer.add_chain(self.to_chain(self.model.to_string()))
        consumer.set_plot_config(config)

        # Context for default mpl style
        with plt.style.context(style):
            return consumer.plotter.plot(**kwargs)
