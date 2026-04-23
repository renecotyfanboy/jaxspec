from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

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
    _rebin_xbins,
    adaptive_bin_1d,
    rebin_counts,
)

if TYPE_CHECKING:
    from ..fit import BayesianModel
    from ..model.background import BackgroundModel


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
        self.model = bayesian_fitter.spectral_model
        self.bayesian_fitter = bayesian_fitter
        self.inference_data = inference_data
        self.obsconfs = bayesian_fitter.forward_model.observations
        self.background_model = background_model

        # Add the model used in fit to the metadata
        for group in self.inference_data.groups():
            group_name = group.split("/")[-1]
            metadata = getattr(self.inference_data, group_name).attrs
            # metadata["model"] = str(self.model)
            # TODO : Store metadata about observations used in the fitting process

    @property
    def converged(self) -> bool:
        r"""
        Convergence of the chain as computed by the $\hat{R}$ statistic.
        """
        rhat = az.rhat(self.inference_data)

        return bool((rhat.to_array() < 1.01).all())

    def _ppc_folded_branches(self, obs_id):
        # TODO : move this to the forward model class, it has nothing to do in here
        obs = self.obsconfs[obs_id]

        idx = list(self.obsconfs.keys()).index(obs_id)
        obs_parameters = jax.tree.map(lambda x: x[..., idx], self.spectrum_parameters)

        if self.bayesian_fitter.settings.get("sparse", False):
            transfer_matrix = BCOO.from_scipy_sparse(
                obs.transfer_matrix.data.to_scipy_sparse().tocsr()
            )

        else:
            transfer_matrix = np.asarray(obs.transfer_matrix.data.todense())

        energies = obs.in_energies

        flux_func = jax.jit(
            jax.vmap(
                jax.vmap(lambda p: self.model.photon_flux(*energies, params=p, split_branches=True))
            )
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
        The input parameters of the model, keyed by full dotted-path numpyro
        site names (e.g. ``"spectrum.powerlaw_1.alpha"``,
        ``"instrument.gain.factor"``, ``"background.countrate"``).

        Shared parameters are broadcast along a trailing observation axis and
        per-observation samples are stacked on that same axis when every
        observation leaf has the same shape. Ragged per-observation entries are
        kept as ``{observation_name: array}``.
        """
        fm = self.bayesian_fitter.forward_model
        obs_names = list(fm.observations.keys())
        effective_prior = self.bayesian_fitter._effective_prior

        out: dict[str, ArrayLike] = {}
        # TODO : check wether this should be simplified under a single call ?
        out.update(
            fm.spectrum.extract_posterior_samples(self.inference_data, effective_prior, obs_names)
        )
        if fm.instrument_model is not None:
            out.update(
                fm.instrument_model.extract_posterior_samples(
                    self.inference_data, effective_prior, obs_names
                )
            )
        if fm.background_model is not None:
            out.update(
                fm.background_model.extract_posterior_samples(
                    self.inference_data, effective_prior, obs_names
                )
            )
        return out

    @cached_property
    def spectrum_parameters(self) -> dict[str, ArrayLike]:
        """Subset of :attr:`input_parameters` belonging to the spectral model."""
        prefix = self.bayesian_fitter.forward_model.spectrum.prior_prefix
        return {k: v for k, v in self.input_parameters.items() if k.startswith(prefix)}

    def _register_derived_parameter(
        self, name: str, value: ArrayLike, prefix: str | None = None
    ) -> None:
        posterior = self.inference_data.posterior
        value = np.asarray(value)

        # TODO : why do we need a prefix here ?
        if prefix is None:
            prefix = self.bayesian_fitter.forward_model.spectrum.prior_prefix

        dims = ("chain", "draw")
        if value.ndim > 2:
            for var_name, data_array in posterior.data_vars.items():
                if not var_name.startswith(prefix):
                    continue
                if data_array.shape == value.shape:
                    dims = data_array.dims
                    break
                if data_array.ndim == value.ndim and data_array.shape[2:] == value.shape[2:]:
                    dims = data_array.dims
                    break
            else:
                extra_dims = tuple(f"derived_dim_{i}" for i in range(value.ndim - 2))
                dims = ("chain", "draw", *extra_dims)

        posterior[name] = (dims, value)

    def photon_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.photon / u.cm**2 / u.s,
        register: bool = False,
        n_points: int = 5,
        n_grid: int = 1_000,
    ) -> ArrayLike:
        """
        Compute the unfolded photon flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the photon flux.
            register: Whether to register the flux with the other posterior parameters.
            n_points: The number of points per bin to use for computing the unfolded spectrum.
            n_grid: The number of grid points to use for computing the unfolded spectrum.
        """
        flux = self.model.integrated_flux(
            e_min,
            e_max,
            params=self.spectrum_parameters,
            energy=False,
            n_points=n_points,
            n_grid=n_grid,
        )
        value = np.asarray(flux * float((u.photon / u.cm**2 / u.s).to(unit)))

        if register:
            self._register_derived_parameter(
                f"derived.photon_flux_{e_min:.1f}_{e_max:.1f}",
                value,
            )

        return value

    def energy_flux(
        self,
        e_min: float,
        e_max: float,
        unit: Unit = u.erg / u.cm**2 / u.s,
        register: bool = False,
        n_points: int = 5,
        n_grid: int = 1_000,
    ) -> ArrayLike:
        """
        Compute the unfolded energy flux in a given energy band. The flux is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band in observer frame.
            e_max: The upper bound of the energy band in observer frame.
            unit: The unit of the energy flux.
            register: Whether to register the flux with the other posterior parameters.
            n_points: The number of points per bin to use for computing the unfolded spectrum.
            n_grid: The number of grid points to use for computing the unfolded spectrum.
        """
        flux = self.model.integrated_flux(
            e_min,
            e_max,
            params=self.spectrum_parameters,
            energy=True,
            n_points=n_points,
            n_grid=n_grid,
        )
        value = np.asarray(flux * float((u.keV / u.cm**2 / u.s).to(unit)))

        if register:
            self._register_derived_parameter(
                f"derived.energy_flux_{e_min:.1f}_{e_max:.1f}",
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
        n_points: int = 5,
        n_grid: int = 1_000,
    ) -> ArrayLike:
        """
        Compute the luminosity of the source specifying its redshift. The luminosity is then added to
        the result parameters so covariance can be plotted.

        Parameters:
            e_min: The lower bound of the energy band.
            e_max: The upper bound of the energy band.
            redshift: The redshift of the source. Incompatible with distance.
            distance: The distance of the source (multiplied by an astropy.unit). Incompatible with redshift.
            observer_frame: Whether the input bands are defined in the observer frame or not.
            cosmology: Chosen cosmology.
            unit: The unit of the luminosity.
            register: Whether to register the flux with the other posterior parameters.
            n_points: The number of points per bin to use for computing the unfolded spectrum.
            n_grid: The number of grid points to use for computing the unfolded spectrum.
        """
        if not observer_frame:
            raise NotImplementedError()

        if redshift is None and distance is None:
            raise ValueError("Either redshift or distance must be specified.")

        if distance is not None:
            if redshift is not None:
                raise ValueError("Redshift must be None as a distance is specified.")
            redshift = distance.to(
                cu.redshift, cu.redshift_distance(cosmology, kind="luminosity")
            ).value

        flux = self.model.integrated_flux(
            e_min * (1 + redshift),
            e_max * (1 + redshift),
            params=self.spectrum_parameters,
            energy=True,
            n_points=n_points,
            n_grid=n_grid,
        ) * (u.keV / u.cm**2 / u.s)
        value = np.asarray(
            (flux * (4 * np.pi * cosmology.luminosity_distance(redshift) ** 2)).to(unit)
        )

        if register:
            self._register_derived_parameter(
                f"derived.luminosity_{e_min:.1f}_{e_max:.1f}",
                value,
            )

        return value

    def to_chain(self, name: str) -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters_type.

        Parameters:
            name: The name of the chain.
        """

        # TODO : we should be able to get parameter from instrument model or background model
        fm = self.bayesian_fitter.forward_model

        keys_to_drop = [
            key
            for key in self.inference_data.posterior.keys()
            if not (
                str(key).startswith(fm.spectrum.prior_prefix) or str(key).startswith("derived.")
            )
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

        # Remove the prefix "spectrum" / "instrument" / "background"
        df = df.rename(columns=lambda colname: colname.split(".", maxsplit=1)[1])

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
        # TODO : add a test against XSPEC to check for this. There will be a hard time handling and determining wether or not the background should be accounted for here
        observed_data = self.inference_data.observed_data
        log_likelihood = self.log_likelihood
        c_stat_data_vars: dict[str, xr.DataArray] = {}

        for var_name, data in observed_data.data_vars.items():
            safe_data = xr.where(data > 0, data, 1)
            saturated = gammaln(data + 1) - xr.where(data > 0, data * (np.log(safe_data) - 1), 0)
            constant = saturated.sum(dim=list(data.dims)) if data.dims else saturated
            c_stat_data_vars[var_name] = -2 * (log_likelihood[var_name] + constant)

        all_c_stat_vars = dict(c_stat_data_vars)

        if len(c_stat_data_vars) > 1:
            all_c_stat_vars["full"] = xr.concat(
                list(c_stat_data_vars.values()), dim="_cstat_component"
            ).sum("_cstat_component")

            observed_terms = [
                value for key, value in c_stat_data_vars.items() if key.startswith("observed.")
            ]
            if observed_terms:
                all_c_stat_vars["observed.all"] = xr.concat(
                    observed_terms, dim="_cstat_component"
                ).sum("_cstat_component")

            background_terms = [
                value
                for key, value in c_stat_data_vars.items()
                if key.startswith("observed_background.")
            ]
            if background_terms:
                all_c_stat_vars["observed_background.all"] = xr.concat(
                    background_terms, dim="_cstat_component"
                ).sum("_cstat_component")

        return xr.Dataset(all_c_stat_vars)

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
        figsize: tuple[float, float] = (6, 6),
        x_lims: tuple[float, float] | None = None,
        rescale_background: bool = False,
        min_counts: int | None = None,
        grouping: int | None = None,
    ) -> list[plt.Figure]:
        r"""
        Plot the posterior predictive distribution of the model. It also features a residual plot, defined using the
        following formula:

        $$ \text{Residual} = \frac{\text{Observed counts} - \text{Posterior counts}}
        {(\text{Posterior counts})_{84\%}-(\text{Posterior counts})_{16\%}} $$

        Parameters:
            n_sigmas: The number of sigmas to plot the envelops.
            x_unit: The units of the x-axis. It can be either a string (parsable by astropy.units) or an astropy unit. It must be homogeneous to either a length, a frequency or an energy.
            y_type: The type of the y-axis. It can be either "counts", "countrate", "photon_flux" or "photon_flux_density".
            plot_background: Whether to plot the background model if it is included in the fit.
            plot_components: Whether to plot the components of the model separately.
            scale: The axes scaling
            alpha_envelope: The transparency range for envelops
            style: The style of the plot. It can be either a string or a matplotlib style context.
            title: The title of the plot.
            figsize: The size of the figure.
            x_lims: The limits of the x-axis.
            rescale_background: Whether to rescale the background model to the data with backscal ratio.
            min_counts: Minimum number of observed counts per grouped bin. Adjacent bins are merged until the threshold is reached. Mutually exclusive with *grouping*.
            grouping: Number of consecutive bins to merge into each group. Mutually exclusive with *min_counts*.

        Returns:
            A list of matplotlib figures for each observation in the model.
        """

        if min_counts is not None and grouping is not None:
            raise ValueError("min_counts and grouping are mutually exclusive")

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
                    figsize=figsize,
                    sharex="col",
                    height_ratios=[0.7, 0.3],
                )

                legend_plots = []
                legend_labels = []

                count = az.extract(
                    self.inference_data,
                    var_names=f"observed.{obs_id}",
                    group="posterior_predictive",
                ).values.T

                xbins, exposure, integrated_arf = _compute_effective_area(obsconf, x_unit)
                observed_counts = obsconf.folded_counts.data

                # --- Adaptive rebinning ---
                if min_counts is not None:
                    bin_ids = adaptive_bin_1d(observed_counts, min_counts)
                elif grouping is not None:
                    n_bins = len(observed_counts)
                    bin_ids = np.arange(n_bins) // grouping
                else:
                    bin_ids = None

                if bin_ids is not None:
                    count = rebin_counts(count, bin_ids)
                    observed_counts = rebin_counts(observed_counts, bin_ids)
                    xbins = _rebin_xbins(xbins, bin_ids)
                    integrated_arf = (
                        rebin_counts(integrated_arf.value, bin_ids) * integrated_arf.unit
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

                y_samples = count * u.ct / denominator

                y_samples = y_samples.to(y_units)

                y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                    observed_counts, denominator, y_units
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

                lowest_y = np.nanmin(y_observed)
                highest_y = np.nanmax(y_observed)

                legend_plots.append((true_data_plot,))
                legend_labels.append("Observed")
                legend_plots += model_plot
                legend_labels.append("Model")

                # Plot the residuals
                residual_samples = (observed_counts - count) / np.diff(
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
                    for (component_name, comp_count), color in zip(
                        self._ppc_folded_branches(obs_id).items(), COLOR_CYCLE
                    ):
                        # _ppc_folded_branches returns (n_chains, n_draws, n_bins) shaped arrays so we must flatten it
                        comp_flat = comp_count.reshape(
                            (comp_count.shape[0] * comp_count.shape[1], -1)
                        )
                        if bin_ids is not None:
                            comp_flat = rebin_counts(comp_flat, bin_ids)
                        y_samples = comp_flat * u.ct / denominator

                        y_samples = y_samples.to(y_units)

                        component_plot = _plot_binned_samples_with_error(
                            ax[0],
                            xbins.value,
                            y_samples.value,
                            color=color,
                            linestyle="dashdot",
                            n_sigmas=n_sigmas,
                            alpha_envelope=alpha_envelope,
                        )

                        name = component_name.split("*")[-1]

                        legend_plots += component_plot
                        legend_labels.append(name)

                if self.background_model is not None and plot_background:
                    # We plot the background only if it is included in the fit, i.e. by subtracting
                    bkg_count = (
                        None
                        if self.background_model is None
                        else az.extract(
                            self.inference_data,
                            var_names=f"observed_background.{obs_id}",
                            group="posterior_predictive",
                        ).values.T
                    )

                    bkg_observed = obsconf.folded_background.data

                    if bin_ids is not None:
                        bkg_count = rebin_counts(bkg_count, bin_ids)
                        bkg_observed = rebin_counts(bkg_observed, bin_ids)
                        rescale_background_factor = (
                            rebin_counts(obsconf.folded_backratio.data, bin_ids)
                            / np.bincount(bin_ids)
                            if rescale_background
                            else 1.0
                        )
                    else:
                        rescale_background_factor = (
                            obsconf.folded_backratio.data if rescale_background else 1.0
                        )

                    y_samples_bkg = (bkg_count * u.ct / denominator).to(y_units)

                    y_observed_bkg, y_observed_bkg_low, y_observed_bkg_high = (
                        _error_bars_for_observed_data(bkg_observed, denominator, y_units)
                    )

                    model_bkg_plot = _plot_binned_samples_with_error(
                        ax[0],
                        xbins.value,
                        y_samples_bkg.value * rescale_background_factor,
                        color=BACKGROUND_COLOR,
                        alpha_envelope=alpha_envelope,
                        n_sigmas=n_sigmas,
                    )

                    true_bkg_plot = _plot_poisson_data_with_error(
                        ax[0],
                        xbins.value,
                        y_observed_bkg.value * rescale_background_factor,
                        y_observed_bkg_low.value * rescale_background_factor,
                        y_observed_bkg_high.value * rescale_background_factor,
                        color=BACKGROUND_DATA_COLOR,
                        alpha=0.7,
                    )

                    # lowest_y = np.nanmin(lowest_y.min, np.nanmin(y_observed_bkg.value).astype(float))
                    # highest_y = np.nanmax(highest_y.value.astype(float), np.nanmax(y_observed_bkg.value).astype(float))

                    legend_plots.append((true_bkg_plot,))
                    legend_labels.append("Observed (bkg)")
                    legend_plots += model_bkg_plot
                    legend_labels.append("Model (bkg)")

                max_residuals = min(3.5, np.nanmax(np.abs(residual_samples)))

                ax[0].loglog()
                ax[1].set_ylim(-np.nanmax([3.5, max_residuals]), +np.nanmax([3.5, max_residuals]))
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
                        raise RuntimeError(
                            f"Unknown physical type for x_units: {x_unit}. "
                            f"Must be 'length', 'energy' or 'frequency'"
                        )

                ax[1].axhline(0, color=SPECTRUM_DATA_COLOR, ls="--")
                ax[1].axhline(-3, color=SPECTRUM_DATA_COLOR, ls=":")
                ax[1].axhline(3, color=SPECTRUM_DATA_COLOR, ls=":")

                ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
                ax[1].set_yticks(range(-3, 4), minor=True)

                ax[0].set_xlim(xbins.value.min(), xbins.value.max())
                ax[0].set_ylim(lowest_y.value * 0.8, highest_y.value * 1.2)
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

                if x_lims is not None:
                    ax[0].set_xlim(*x_lims)

                fig.align_ylabels()
                plt.subplots_adjust(hspace=0.0)
                fig.suptitle(f"Posterior predictive - {obs_id}" if title is None else title)
                fig.tight_layout()
                figure_list.append(fig)
                # fig.show()

        plt.tight_layout()
        plt.show()

        return figure_list

    def table(self) -> str:
        r"""
        Return a formatted $\LaTeX$ table of the results of the fit.
        """

        consumer = ChainConsumer()
        consumer.add_chain(self.to_chain("Model"))

        return consumer.analysis.get_latex_table(caption="Fit result", label="tab:results")

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
        consumer.add_chain(self.to_chain("Results"))
        consumer.set_plot_config(config)

        # Context for default mpl style
        with plt.style.context("default"):
            return consumer.plotter.plot(**kwargs)
