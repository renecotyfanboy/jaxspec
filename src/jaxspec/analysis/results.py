from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import arviz as az
import astropy.cosmology.units as cu
import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from astropy.cosmology import Cosmology, Planck18
from astropy.units import Unit
from chainconsumer import Chain, ChainConsumer, PlotConfig
from jax.experimental.sparse import BCOO
from jax.typing import ArrayLike
from numpyro.handlers import seed
from scipy.special import gammaln

from ._plot import (
    BACKGROUND_COLOR,
    BACKGROUND_DATA_COLOR,
    COLOR_CYCLE,
    SPECTRUM_COLOR,
    SPECTRUM_DATA_COLOR,
    _compute_effective_area,
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
)

if TYPE_CHECKING:
    from ..fit import BayesianModel
    from ..model.background import BackgroundModel

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class FitResult:
    """
    Container for the result of a fit using any ModelFitter class.
    """

    # TODO : Add type hints
    def __init__(
        self,
        bayesian_fitter: BayesianModel,
        inference_data: az.InferenceData,
        background_model: BackgroundModel = None,
    ):
        self.model = bayesian_fitter.model
        self.bayesian_fitter = bayesian_fitter
        self.inference_data = inference_data
        self.obsconfs = bayesian_fitter.observation_container
        self.background_model = background_model

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

    def _ppc_folded_branches(self, obs_id):
        obs = self.obsconfs[obs_id]

        if len(next(iter(self.input_parameters.values())).shape) > 2:
            idx = list(self.obsconfs.keys()).index(obs_id)
            obs_parameters = jax.tree.map(lambda x: x[..., idx], self.input_parameters)

        else:
            obs_parameters = self.input_parameters

        if self.bayesian_fitter.sparse:
            transfer_matrix = BCOO.from_scipy_sparse(
                obs.transfer_matrix.data.to_scipy_sparse().tocsr()
            )

        else:
            transfer_matrix = np.asarray(obs.transfer_matrix.data.todense())

        energies = obs.in_energies

        flux_func = jax.jit(
            jax.vmap(jax.vmap(lambda p: self.model.photon_flux(p, *energies, split_branches=True)))
        )
        convolve_func = jax.jit(
            jax.vmap(jax.vmap(lambda flux: jnp.clip(transfer_matrix @ flux, a_min=1e-6)))
        )
        return jax.tree.map(
            lambda flux: np.random.poisson(convolve_func(flux)), flux_func(obs_parameters)
        )

    @cached_property
    def input_parameters(self) -> dict[str, ArrayLike]:
        """
        The input parameters of the model.
        """

        posterior = az.extract(self.inference_data, combined=False)

        samples_shape = (len(posterior.coords["chain"]), len(posterior.coords["draw"]))

        total_shape = tuple(posterior.sizes[d] for d in posterior.coords)

        posterior = {key: posterior[key].data for key in posterior.data_vars}

        with seed(rng_seed=0):
            input_parameters = self.bayesian_fitter.prior_distributions_func()

        for key, value in input_parameters.items():
            module, parameter = key.rsplit("_", 1)

            if f"{module}_{parameter}" in posterior.keys():
                # We add as extra dimension as there might be different values per observation
                if posterior[f"{module}_{parameter}"].shape == samples_shape:
                    to_set = posterior[f"{module}_{parameter}"][..., None]
                else:
                    to_set = posterior[f"{module}_{parameter}"]

                input_parameters[f"{module}_{parameter}"] = to_set

            else:
                # The parameter is fixed in this case, so we just broadcast is over chain and draws
                input_parameters[f"{module}_{parameter}"] = value[None, None, ...]

            if len(total_shape) < len(input_parameters[f"{module}_{parameter}"].shape):
                # If there are only chains and draws, we reduce
                input_parameters[f"{module}_{parameter}"] = input_parameters[
                    f"{module}_{parameter}"
                ][..., 0]

            else:
                input_parameters[f"{module}_{parameter}"] = jnp.broadcast_to(
                    input_parameters[f"{module}_{parameter}"], total_shape
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
        redshift: float | ArrayLike = None,
        distance: float | ArrayLike = None,
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

        if redshift is None and distance is None:
            raise ValueError("Either redshift or distance must be specified.")

        if distance is not None:
            if redshift is not None:
                raise ValueError("Redshift must be None as a distance is specified.")
            else:
                redshift = distance.to(
                    cu.redshift, cu.redshift_distance(cosmology, kind="luminosity")
                ).value

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

    def to_chain(self, name: str) -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters_type.

        Parameters:
            name: The name of the chain.
        """

        keys_to_drop = [
            key
            for key in self.inference_data.posterior.keys()
            if (key.startswith("_") or key.startswith("bkg"))
        ]

        reduced_id = az.extract(
            self.inference_data,
            var_names=[f"~{key}" for key in keys_to_drop] if keys_to_drop else None,
            group="posterior",
        )

        df_list = []

        for var, array in reduced_id.data_vars.items():
            extra_dims = [dim for dim in array.dims if dim not in ["sample"]]

            if extra_dims:
                dim = extra_dims[
                    0
                ]  # We only support the case where the extra dimension comes from the observations

                for coord, obs_id in zip(array.coords[dim], self.obsconfs.keys()):
                    df = array.loc[{dim: coord}].to_pandas()
                    df.name += f"\n[{obs_id}]"
                    df_list.append(df)
            else:
                df_list.append(array.to_pandas())

        df = pd.concat(df_list, axis=1)

        return Chain(samples=df, name=name)

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
        n_sigmas: int = 1,
        x_unit: str | u.Unit = "keV",
        y_type: Literal[
            "counts", "countrate", "photon_flux", "photon_flux_density"
        ] = "photon_flux_density",
        plot_background: bool = True,
        plot_components: bool = False,
        scale: Literal["linear", "semilogx", "semilogy", "loglog"] = "loglog",
        alpha_envelope: (float, float) = (0.15, 0.25),
        style: str | Any = "default",
        title: str | None = None,
    ) -> list[plt.Figure]:
        r"""
        Plot the posterior predictive distribution of the model. It also features a residual plot, defined using the
        following formula:

        $$ \text{Residual} = \frac{\text{Observed counts} - \text{Posterior counts}}
        {(\text{Posterior counts})_{84\%}-(\text{Posterior counts})_{16\%}} $$

        Parameters:
            percentile: The percentile of the posterior predictive distribution to plot.
            x_unit: The units of the x-axis. It can be either a string (parsable by astropy.units) or an astropy unit. It must be homogeneous to either a length, a frequency or an energy.
            y_type: The type of the y-axis. It can be either "counts", "countrate", "photon_flux" or "photon_flux_density".
            plot_background: Whether to plot the background model if it is included in the fit.
            plot_components: Whether to plot the components of the model separately.
            scale: The axes scaling
            alpha_envelope: The transparency range for envelops
            style: The style of the plot. It can be either a string or a matplotlib style context.

        Returns:
            A list of matplotlib figures for each observation in the model.
        """

        obsconf_container = self.obsconfs
        figure_list = []
        x_unit = u.Unit(x_unit)

        match y_type:
            case "counts":
                y_units = u.ct
            case "countrate":
                y_units = u.ct / u.s
            case "photon_flux":
                y_units = u.ct / u.cm**2 / u.s
            case "photon_flux_density":
                y_units = u.ct / u.cm**2 / u.s / x_unit
            case _:
                raise ValueError(
                    f"Unknown y_type: {y_type}. Must be 'counts', 'countrate', 'photon_flux' or 'photon_flux_density'"
                )

        with plt.style.context(style):
            for obs_id, obsconf in obsconf_container.items():
                fig, ax = plt.subplots(
                    2,
                    1,
                    figsize=(6, 6),
                    sharex="col",
                    height_ratios=[0.7, 0.3],
                )

                legend_plots = []
                legend_labels = []

                count = az.extract(
                    self.inference_data, var_names=f"obs_{obs_id}", group="posterior_predictive"
                ).values.T

                xbins, exposure, integrated_arf = _compute_effective_area(obsconf, x_unit)

                match y_type:
                    case "counts":
                        denominator = 1
                    case "countrate":
                        denominator = exposure
                    case "photon_flux":
                        denominator = integrated_arf * exposure
                    case "photon_flux_density":
                        denominator = (xbins[1] - xbins[0]) * integrated_arf * exposure

                y_samples = (count * u.ct / denominator).to(y_units)

                y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                    obsconf.folded_counts.data, denominator, y_units
                )

                # Use the helper function to plot the data and posterior predictive
                model_plot = _plot_binned_samples_with_error(
                    ax[0],
                    xbins.value,
                    y_samples.value,
                    color=SPECTRUM_COLOR,
                    n_sigmas=n_sigmas,
                    alpha_envelope=alpha_envelope,
                )

                true_data_plot = _plot_poisson_data_with_error(
                    ax[0],
                    xbins.value,
                    y_observed.value,
                    y_observed_low.value,
                    y_observed_high.value,
                    color=SPECTRUM_DATA_COLOR,
                    alpha=0.7,
                )

                legend_plots.append((true_data_plot,))
                legend_labels.append("Observed")
                legend_plots += model_plot
                legend_labels.append("Model")

                # Plot the residuals
                residual_samples = (obsconf.folded_counts.data - count) / np.diff(
                    np.percentile(count, [16, 84], axis=0), axis=0
                )

                _plot_binned_samples_with_error(
                    ax[1],
                    xbins.value,
                    residual_samples,
                    color=SPECTRUM_COLOR,
                    n_sigmas=n_sigmas,
                    alpha_envelope=alpha_envelope,
                )

                if plot_components:
                    for (component_name, count), color in zip(
                        self._ppc_folded_branches(obs_id).items(), COLOR_CYCLE
                    ):
                        # _ppc_folded_branches returns (n_chains, n_draws, n_bins) shaped arrays so we must flatten it
                        y_samples = (
                            count.reshape((count.shape[0] * count.shape[1], -1))
                            * u.ct
                            / denominator
                        ).to(y_units)
                        component_plot = _plot_binned_samples_with_error(
                            ax[0],
                            xbins.value,
                            y_samples.value,
                            color=color,
                            linestyle="dashdot",
                            n_sigmas=n_sigmas,
                            alpha_envelope=alpha_envelope,
                        )

                        legend_plots += component_plot
                        legend_labels.append(component_name)

                if self.background_model is not None and plot_background:
                    # We plot the background only if it is included in the fit, i.e. by subtracting
                    bkg_count = (
                        None
                        if self.background_model is None
                        else az.extract(
                            self.inference_data,
                            var_names=f"bkg_{obs_id}",
                            group="posterior_predictive",
                        ).values.T
                    )

                    y_samples_bkg = (bkg_count * u.ct / denominator).to(y_units)

                    y_observed_bkg, y_observed_bkg_low, y_observed_bkg_high = (
                        _error_bars_for_observed_data(
                            obsconf.folded_background.data, denominator, y_units
                        )
                    )

                    model_bkg_plot = _plot_binned_samples_with_error(
                        ax[0],
                        xbins.value,
                        y_samples_bkg.value,
                        color=BACKGROUND_COLOR,
                        alpha_envelope=alpha_envelope,
                        n_sigmas=n_sigmas,
                    )

                    true_bkg_plot = _plot_poisson_data_with_error(
                        ax[0],
                        xbins.value,
                        y_observed_bkg.value,
                        y_observed_bkg_low.value,
                        y_observed_bkg_high.value,
                        color=BACKGROUND_DATA_COLOR,
                        alpha=0.7,
                    )

                    legend_plots.append((true_bkg_plot,))
                    legend_labels.append("Observed (bkg)")
                    legend_plots += model_bkg_plot
                    legend_labels.append("Model (bkg)")

                max_residuals = np.max(np.abs(residual_samples))

                ax[0].loglog()
                ax[1].set_ylim(-max(3.5, max_residuals), +max(3.5, max_residuals))
                ax[0].set_ylabel(f"Folded spectrum\n [{y_units:latex_inline}]")
                ax[1].set_ylabel("Residuals \n" + r"[$\sigma$]")

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

                ax[1].axhline(0, color=SPECTRUM_DATA_COLOR, ls="--")
                ax[1].axhline(-3, color=SPECTRUM_DATA_COLOR, ls=":")
                ax[1].axhline(3, color=SPECTRUM_DATA_COLOR, ls=":")

                ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
                ax[1].set_yticks(range(-3, 4), minor=True)

                ax[0].set_xlim(xbins.value.min(), xbins.value.max())

                ax[0].legend(legend_plots, legend_labels)

                match scale:
                    case "linear":
                        ax[0].set_xscale("linear")
                        ax[0].set_yscale("linear")
                    case "semilogx":
                        ax[0].set_xscale("log")
                        ax[0].set_yscale("linear")
                    case "semilogy":
                        ax[0].set_xscale("linear")
                        ax[0].set_yscale("log")
                    case "loglog":
                        ax[0].set_xscale("log")
                        ax[0].set_yscale("log")

                fig.align_ylabels()
                plt.subplots_adjust(hspace=0.0)
                fig.tight_layout()
                figure_list.append(fig)
                fig.suptitle(f"Posterior predictive - {obs_id}" if title is None else title)
                # fig.show()

        plt.tight_layout()
        plt.show()

        return figure_list

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
