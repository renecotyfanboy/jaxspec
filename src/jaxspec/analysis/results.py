import arviz as az
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from ..data.observation import Observation
from ..model.abc import SpectralModel
from collections.abc import Mapping
from typing import TypeVar, Tuple
from astropy.cosmology import Cosmology, Planck18
from numpyro.infer import MCMC
import astropy.units as u
from astropy.units import Unit
from haiku.data_structures import traverse
from chainconsumer import ChainConsumer, Chain, PlotConfig
import jax
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

K = TypeVar("K")
V = TypeVar("V")


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
        observations: list[Observation],
        mcmc: MCMC,
        structure: Mapping[K, V],
    ):
        self.model = model
        self._structure = structure
        self.inference_data = az.from_numpyro(mcmc)
        self.observations = observations
        self.samples = mcmc.get_samples()
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

    @property
    def chain(self) -> Chain:
        df = pd.DataFrame.from_dict({key: np.ravel(value) for key, value in self.samples.items()})
        chain = Chain(samples=df, name="Model", color=(0.15, 0.35, 0.55))
        chain.samples.columns = [format_parameters(parameter) for parameter in chain.samples.columns]

        return chain

    @property
    def consumer(self) -> ChainConsumer:
        consumer = ChainConsumer()
        consumer.add_chain(self.chain)

        return consumer

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

    def plot_ppc(self, index: int, percentile: Tuple[int, int] = (14, 86)):
        from ..data.util import fakeit_for_multiple_parameters

        observation = self.observations[index]

        count = fakeit_for_multiple_parameters(observation, self.model, self.params)

        color = (0.15, 0.25, 0.45)

        with plt.style.context("default"):
            fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[0.7, 0.3])

            mid_bins_arf = (observation.arf.energ_hi + observation.arf.energ_lo) / 2

            interpolated_arf = interp1d(mid_bins_arf, observation.arf.specresp)
            integrated_arf = np.array(
                [
                    trapezoid(interpolated_arf(np.linspace(bin_low, bin_up, 10)), np.linspace(bin_low, bin_up, 10))
                    for (bin_low, bin_up) in zip(observation.out_energies[0], observation.out_energies[1])
                ]
            ) / (observation.out_energies[1] - observation.out_energies[0])

            if observation.out_energies[0][0] < 1 < observation.out_energies[1][-1]:
                xticks = [np.floor(observation.out_energies[0][0] * 10) / 10, 1.0, np.floor(observation.out_energies[1][-1])]
            else:
                xticks = [np.floor(observation.out_energies[0][0] * 10) / 10, np.floor(observation.out_energies[1][-1])]

            denominator = (observation.out_energies[1] - observation.out_energies[0]) * observation.exposure * integrated_arf

            axs[0].step(
                list(observation.out_energies[0]) + [observation.out_energies[1][-1]],
                list(observation.observed_counts / denominator) + [(observation.observed_counts / denominator)[-1]],
                where="post",
                label="data",
                c=color,
            )

            percentiles = np.percentile(count, percentile, axis=0)
            axs[0].fill_between(
                list(observation.out_energies[0]) + [observation.out_energies[1][-1]],
                list(percentiles[0] / denominator) + [(percentiles[0] / denominator)[-1]],
                list(percentiles[1] / denominator) + [(percentiles[1] / denominator)[-1]],
                alpha=0.3,
                step="post",
                label="posterior predictive",
                facecolor=color,
            )

            axs[0].set_ylabel("Folded spectrum\n" + r"[Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$]")
            axs[0].loglog()
            axs[0].set_xlim(observation.out_energies[0][0], observation.out_energies[1][-1])

            residuals = np.percentile(
                (observation.observed_counts - count) / np.diff(np.percentile(count, percentile, axis=0), axis=0),
                percentile,
                axis=0,
            )

            max_residuals = np.max(np.abs(residuals))

            axs[1].fill_between(
                list(observation.out_energies[0]) + [observation.out_energies[1][-1]],
                list(residuals[0]) + [residuals[0][-1]],
                list(residuals[1]) + [residuals[1][-1]],
                alpha=0.3,
                step="post",
                label="posterior predictive",
                facecolor=color,
            )

            axs[1].set_xlim(observation.out_energies[0][0], observation.out_energies[1][-1])
            axs[1].set_ylim(-max(3.5, max_residuals), +max(3.5, max_residuals))
            axs[1].set_ylabel("Residuals \n" + r"[$\sigma$]")
            axs[1].set_xlabel("Energy \n[keV]")

            axs[1].axhline(0, color=color, ls="--")
            axs[1].axhline(-3, color=color, ls=":")
            axs[1].axhline(3, color=color, ls=":")

            axs[1].set_xticks(xticks, labels=xticks)
            axs[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
            axs[1].set_yticks(range(-3, 4), minor=True)

            fig.suptitle(self.model.to_string())

            fig.align_ylabels()
            plt.subplots_adjust(hspace=0.0)
            fig.tight_layout()

    def table(self):
        return self.consumer.analysis.get_latex_table(caption="Results of the fit", label="tab:results")

    def plot_corner(
        self,
        config: PlotConfig = PlotConfig(usetex=False, summarise=False, label_font_size=6),
        **kwargs,
    ):
        """
        Plot the corner plot of the posterior distribution of the parameters. This method uses the ChainConsumer.

        Parameters:
            config: The configuration of the plot.
            **kwargs: Additional arguments passed to ChainConsumer.plotter.plot.
        """

        consumer = self.consumer
        consumer.set_plot_config(config)

        # Context for default mpl style
        with plt.style.context("default"):
            consumer.plotter.plot(**kwargs)
