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

        # The legend cannot handle fill_between, so we pass a fill to get a fancy icone
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
    # TODO : Add docstring
    # TODO : Add type hints
    # TODO : Add proper separation between params and samples, cf from haiku and numpyro
    def __init__(
        self,
        model: SpectralModel,
        folding_model: ObsConfiguration,
        inference_data: az.InferenceData,
        samples,
        structure: Mapping[K, V],
        background_model: BackgroundModel = None,
    ):
        self.model = model
        self._structure = structure
        self.inference_data = inference_data
        self.folding_model = folding_model
        self.samples = samples
        self.background_model = background_model
        self._structure = structure

        # Add the model used in fit to the metadata
        for group in self.inference_data.groups():
            group_name = group.split("/")[-1]
            metadata = getattr(self.inference_data, group_name).attrs
            metadata["model"] = str(model)
            # TODO : Store metadata about observations used in the fitting process

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

    def chain(self, name: str, parameters: Literal["model", "bkg"] = "model") -> Chain:
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
    def params(self):
        """
        Haiku-like structure for the parameters
        """

        params = {}

        for module, parameter, value in traverse(self._structure):
            if params.get(module, None) is None:
                params[module] = {}
            params[module][parameter] = self.samples[f"{module}_{parameter}"]

        return params

    def plot_ppc(self, percentile: Tuple[int, int] = (14, 86)) -> plt.Figure:
        r"""
        Plot the posterior predictive distribution of the model. It also features a residual plot, defined using the
        following formula:

        $$ \text{Residual} = \frac{\text{Observed counts} - \text{Posterior counts}}
        {(\text{Posterior counts})_{84\%}-(\text{Posterior counts})_{16\%}} $$

        Parameters:
            percentile: The percentile of the posterior predictive distribution to plot.

        Returns:
            The matplotlib two panel figure.
        """

        folding_model = self.folding_model
        count = az.extract(self.inference_data, var_names="obs", group="posterior_predictive").values.T
        bkg_count = (
            None
            if self.background_model is None
            else az.extract(self.inference_data, var_names="bkg", group="posterior_predictive").values.T
        )

        legend_plots = []
        legend_labels = []

        color = (0.15, 0.25, 0.45)

        with plt.style.context("default"):
            # Note to Simon : do not change folding_model.out_energies[1] - folding_model.out_energies[0] to
            # np.diff, you already did this twice and forgot that it does not work since diff keeps the dimensions
            # and enable weird broadcasting that makes the plot fail

            fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[0.7, 0.3])

            mid_bins_arf = folding_model.in_energies.mean(axis=0)

            e_grid = np.linspace(*folding_model.out_energies, 10)
            interpolated_arf = np.interp(e_grid, mid_bins_arf, folding_model.area)
            integrated_arf = trapezoid(interpolated_arf, x=e_grid, axis=0) / (
                folding_model.out_energies[1] - folding_model.out_energies[0]
            )

            if folding_model.out_energies[0][0] < 1 < folding_model.out_energies[1][-1]:
                xticks = [np.floor(folding_model.out_energies[0][0] * 10) / 10, 1.0, np.floor(folding_model.out_energies[1][-1])]
            else:
                xticks = [np.floor(folding_model.out_energies[0][0] * 10) / 10, np.floor(folding_model.out_energies[1][-1])]

            denominator = (
                (folding_model.out_energies[1] - folding_model.out_energies[0]) * folding_model.exposure.data * integrated_arf
            )

            # Use the helper function to plot the data and posterior predictive
            legend_plots += _plot_binned_samples_with_error(
                axs[0],
                folding_model.out_energies,
                y_samples=count,
                y_observed=folding_model.folded_counts.data,
                denominator=denominator,
                color=color,
                percentile=percentile,
            )

            legend_labels.append("Source + Background")

            if self.background_model is not None:
                # We plot the background only if it is included in the fit, i.e. by subtracting
                legend_plots += _plot_binned_samples_with_error(
                    axs[0],
                    folding_model.out_energies,
                    y_observed=folding_model.folded_background.data,
                    y_samples=bkg_count,
                    denominator=denominator * folding_model.folded_backratio.data,
                    color=(0.26787604, 0.60085972, 0.63302651),
                    percentile=percentile,
                )

                legend_labels.append("Background")

            residuals = np.percentile(
                (folding_model.folded_counts.data - count) / np.diff(np.percentile(count, percentile, axis=0), axis=0),
                percentile,
                axis=0,
            )

            axs[1].fill_between(
                list(folding_model.out_energies[0]) + [folding_model.out_energies[1][-1]],
                list(residuals[0]) + [residuals[0][-1]],
                list(residuals[1]) + [residuals[1][-1]],
                alpha=0.3,
                step="post",
                facecolor=color,
            )

            max_residuals = np.max(np.abs(residuals))

            axs[0].loglog()
            axs[0].set_ylabel("Folded spectrum\n" + r"[Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$]")

            axs[1].set_ylim(-max(3.5, max_residuals), +max(3.5, max_residuals))
            axs[1].set_ylabel("Residuals \n" + r"[$\sigma$]")
            axs[1].set_xlabel("Energy \n[keV]")

            axs[1].axhline(0, color=color, ls="--")
            axs[1].axhline(-3, color=color, ls=":")
            axs[1].axhline(3, color=color, ls=":")

            axs[1].set_xticks(xticks, labels=xticks)
            axs[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
            axs[1].set_yticks(range(-3, 4), minor=True)

            axs[0].set_xlim(folding_model.out_energies.min(), folding_model.out_energies.max())

            axs[0].legend(legend_plots, legend_labels)
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
        consumer.add_chain(self.chain(self.model.to_string()))

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
        consumer.add_chain(self.chain(self.model.to_string()))
        consumer.set_plot_config(config)

        # Context for default mpl style
        with plt.style.context(style):
            return consumer.plotter.plot(**kwargs)
