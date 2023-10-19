import arviz as az
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from ..data.observation import Observation
from ..model.abc import SpectralModel
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar, Tuple
from astropy.cosmology import Cosmology, Planck18
from numpyro.infer import MCMC
import astropy.units as u
from astropy.units import Unit
from haiku.data_structures import traverse
from chainconsumer import ChainConsumer, Chain, PlotConfig
import jax

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


class ResultContainer(ABC):
    """
    This class is a container for the results of a fit.

    TODO : Add flux, luminosity, etc.
    """

    model: SpectralModel

    def __init__(self, model: SpectralModel, structure: Mapping[K, V]):
        self.model = model
        self._structure = structure

    @abstractmethod
    def plot_ppc(self, index: int, percentile: Tuple[int, int] = (14, 86)):
        ...

    @property
    @abstractmethod
    def table(self):
        ...


class ChainResult(ResultContainer):
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
        super().__init__(model, structure)

        self.inference_data = az.from_numpyro(mcmc)
        self.observations = observations
        self.samples = mcmc.get_samples()
        self._structure = structure

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

        flux = jax.vmap(
            lambda p: self.model.photon_flux(
                p, np.asarray([e_min]), np.asarray([e_max])
            )
        )(self.params)

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

        flux = jax.vmap(
            lambda p: self.model.energy_flux(
                p, np.asarray([e_min]), np.asarray([e_max])
            )
        )(self.params)

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

        flux = self.energy_flux(e_min * (1 + redshift), e_max * (1 + redshift)) * (
            u.erg / u.cm**2 / u.s
        )

        value = (flux * (4 * np.pi * cosmology.luminosity_distance(redshift) ** 2)).to(
            unit
        )

        self.samples[rf"Luminosity ({e_min:.1f}-{e_max:.1f} keV)"] = value

        return value

    @property
    def chain(self) -> Chain:
        df = pd.DataFrame.from_dict(
            {key: np.ravel(value) for key, value in self.samples.items()}
        )
        chain = Chain(samples=df, name="Model")
        chain.samples.columns = [
            format_parameters(parameter) for parameter in chain.samples.columns
        ]

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

        count = fakeit_for_multiple_parameters(
            self.observations[0], self.model, self.params
        )

        with plt.style.context("default"):
            fig, axs = plt.subplots(
                2, 1, figsize=(8, 5), sharex=True, height_ratios=[0.6, 0.4]
            )

            observation = self.observations[index]

            axs[0].step(
                observation.out_energies[0],
                observation.observed_counts,
                where="post",
                label="data",
            )

            axs[0].fill_between(
                observation.out_energies[0],
                *np.percentile(count, percentile, axis=0),
                alpha=0.3,
                step="post",
                label="posterior predictive",
            )

            axs[0].set_ylabel("Counts")
            axs[0].loglog()

            residuals = np.percentile(
                (observation.observed_counts - count)
                / np.diff(np.percentile(count, percentile, axis=0), axis=0),
                percentile,
                axis=0,
            )

            max_residuals = np.max(np.abs(residuals))

            axs[1].fill_between(
                observation.out_energies[0],
                *residuals,
                alpha=0.3,
                step="post",
                label="posterior predictive",
            )

            axs[1].set_ylim(-max_residuals, +max_residuals)
            axs[1].set_ylabel("Residuals")
            axs[1].set_xlabel("Energy [keV]")
            axs[1].axhline(0, color="k", ls="--")

            axs[0].set_xlim(observation.low_energy, observation.high_energy)
            plt.subplots_adjust(hspace=0.0)

    def table(self):
        return self.consumer.analysis.get_latex_table(
            caption="Results of the fit", label="tab:results"
        )

    def plot_corner(
        self,
        config: PlotConfig = PlotConfig(
            usetex=False, summarise=False, label_font_size=6
        ),
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
