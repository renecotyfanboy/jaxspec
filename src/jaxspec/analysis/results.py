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
import numpyro.distributions as dist
import pandas as pd
import xarray as xr

from astropy.cosmology import Cosmology, Planck18
from astropy.units import Unit
from chainconsumer import Chain, ChainConsumer, PlotConfig
from jax.experimental.sparse import BCOO
from jax.typing import ArrayLike
from scipy.special import gammaln

from ..fit._parameter import TiedParameter
from ..fit._prior_resolution import _per_obs_site_name, parse_prior_key
from ._plot import (
    BACKGROUND_COLOR,
    BACKGROUND_DATA_COLOR,
    COLOR_CYCLE,
    SPECTRUM_COLOR,
    SPECTRUM_DATA_COLOR,
    _compute_bin_ids,
    _compute_effective_area,
    _error_bars_for_observed_data,
    _plot_binned_samples_with_error,
    _plot_poisson_data_with_error,
    _rebin_xbins,
    rebin_counts,
)

if TYPE_CHECKING:
    from ..fit import BayesianModel


_Y_UNITS_FOR_TYPE: dict[str, Any] = {
    "counts": u.ct,
    "countrate": u.ct / u.s,
    "photon_flux": u.ct / u.cm**2 / u.s,
    # "photon_flux_density" is xunit-dependent; handled inline.
}

_XLABEL_FOR_PHYSICAL_TYPE: dict[str, str] = {
    "length": "Wavelength",
    "energy": "Energy",
    "frequency": "Frequency",
}

_SCALE_TO_AXES: dict[str, tuple[str, str]] = {
    "linear": ("linear", "linear"),
    "semilogx": ("log", "linear"),
    "semilogy": ("linear", "log"),
    "loglog": ("log", "log"),
}


class FitResult:
    """
    Container for the result of a fit using any ModelFitter class.
    """

    # TODO : Add type hints
    def __init__(
        self,
        bayesian_fitter: BayesianModel,
        inference_data: az.InferenceData,
    ):
        self.model = bayesian_fitter.spectral_model
        self.bayesian_fitter = bayesian_fitter
        self.inference_data = inference_data
        self.obsconfs = bayesian_fitter.forward_model.observations

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
        The input parameters of the model, keyed by the user's prior dict
        entry paths (e.g. ``"spectrum.powerlaw_1.alpha"``,
        ``"instrument.gain.factor"``, ``"background.countrate"``).

        Shared parameters are broadcast along a trailing observation axis and
        per-observation samples are stacked on that same axis when every
        observation leaf has the same shape. Ragged per-observation entries are
        kept as ``{observation_name: array}``.

        Returns an empty dict when the user passed a callable prior — there's
        no static key set to enumerate.
        """
        fm = self.bayesian_fitter.forward_model
        effective_prior = self.bayesian_fitter._effective_prior

        posterior = az.extract(self.inference_data, combined=False)
        chain_draw = (posterior.sizes["chain"], posterior.sizes["draw"])
        data_vars = {name: jnp.asarray(da.data) for name, da in posterior.data_vars.items()}

        # Group entries by their base path so [obs] / [*] / shared get assembled together.
        by_base: dict[str, dict[str | None, Any]] = {}
        for raw_key, value in effective_prior.items():
            base, scope = parse_prior_key(raw_key)
            by_base.setdefault(base, {})[scope] = value

        prefix_to_obs = {
            "spectrum": list(fm.observations.keys()),
            "instrument": list(fm.instrument.keys()),
            "background": list(fm.background.keys()),
        }

        out: dict[str, ArrayLike] = {}
        deferred_ties: list[tuple[str, TiedParameter, list[str]]] = []

        for base, scopes in by_base.items():
            prefix = base.split(".", 1)[0]
            rest = base[len(prefix) + 1 :]
            obs_axis = prefix_to_obs.get(prefix, [])

            if None in scopes:
                value = _resolve_shared_entry(
                    scopes[None], base, obs_axis, data_vars, chain_draw, deferred_ties
                )
                if value is not None:
                    out[base] = value
            else:
                value = _resolve_per_obs_entry(
                    scopes,
                    base,
                    prefix,
                    rest,
                    obs_axis,
                    data_vars,
                    chain_draw,
                    deferred_ties,
                )
                if value is not None:
                    out[base] = value

        _apply_tied_resolutions(out, deferred_ties)
        return out

    @cached_property
    def spectrum_parameters(self) -> dict[str, ArrayLike]:
        """Subset of :attr:`input_parameters` belonging to the spectral model."""
        return {k: v for k, v in self.input_parameters.items() if k.startswith("spectrum.")}

    def _register_derived_parameter(
        self, name: str, value: ArrayLike, prefix: str | None = None
    ) -> None:
        posterior = self.inference_data.posterior
        value = np.asarray(value)

        if prefix is None:
            prefix = "spectrum."

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

    def _tied_destination_site_names(self) -> set[str]:
        """Site names of every ``TiedParameter`` destination in the effective prior.

        These show up in the posterior as ``numpyro.deterministic`` sites and
        carry no independent posterior info, so they're excluded from the
        corner plot. Mirrors the site-naming convention used by
        :func:`~jaxspec.fit._bayesian_model._resolve_tied_entry`.
        """
        names: set[str] = set()
        effective_prior = self.bayesian_fitter._effective_prior
        if not isinstance(effective_prior, dict):
            return names
        applicable = {
            "spectrum": set(self.bayesian_fitter.forward_model.spectrum),
            "instrument": set(self.bayesian_fitter.forward_model.instrument),
            "background": set(self.bayesian_fitter.forward_model.background),
        }
        for raw_key, value in effective_prior.items():
            if not isinstance(value, TiedParameter):
                continue
            path, scope = parse_prior_key(raw_key)
            if scope is None:
                names.add(path)
                continue
            prefix = path.split(".", 1)[0]
            obs_set = applicable.get(prefix, set())
            if scope == "*":
                for obs in obs_set:
                    names.add(_per_obs_site_name(path, obs))
            elif scope in obs_set:
                names.add(_per_obs_site_name(path, scope))
        return names

    def _var_to_dataframes(self, var, array, obs_ids) -> list[pd.DataFrame]:
        """Convert a single posterior data_var into one or more named DataFrames.

        Per-obs sites display as ``"<rest>\\n[<obs>]"`` (matching the
        shared-broadcast label format); shared sites with an obs-axis broadcast
        get one column per observation; plain shared sites pass through.
        Multi-dim per-obs sites (e.g. per-bin background countrate vectors) are
        skipped since they're not useful in a corner plot.
        """
        varname = str(var)
        extra_dims = [dim for dim in array.dims if dim != "sample"]

        if varname.startswith("forward."):
            if extra_dims:
                return []
            _, _prefix_seg, obs_seg, *rest = varname.split(".")
            df = array.to_pandas()
            df.name = f"{'.'.join(rest)}\n[{obs_seg}]"
            return [df]

        if extra_dims:
            # We only support the case where the extra dimension comes from the observations
            dim = extra_dims[0]
            dfs = []
            for coord, obs_id in zip(array.coords[dim], obs_ids):
                df = array.loc[{dim: coord}].to_pandas()
                df.name += f"\n[{obs_id}]"
                dfs.append(df)
            return dfs

        return [array.to_pandas()]

    def to_chain(self, name: str) -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters_type.

        Parameters:
            name: The name of the chain.
        """
        tied_site_names = self._tied_destination_site_names()

        # Keep shared / derived sites (no "forward." prefix) and per-obs
        # parameter sites (under "forward.spectrum." / "forward.instrument." /
        # "forward.background."). Drop tied-destination sites and observed-data
        # / likelihood sites.
        keep_prefixes = (
            "spectrum.",
            "derived.",
            "forward.spectrum.",
            "forward.instrument.",
            "forward.background.",
        )
        keys_to_drop = [
            key
            for key in self.inference_data.posterior.keys()
            if str(key) in tied_site_names or not any(str(key).startswith(p) for p in keep_prefixes)
        ]

        reduced_id = az.extract(
            self.inference_data,
            var_names=[f"~{key}" for key in keys_to_drop] if keys_to_drop else None,
            group="posterior",
        )

        obs_ids = list(self.obsconfs.keys())
        df_list = [
            df
            for var, array in reduced_id.data_vars.items()
            for df in self._var_to_dataframes(var, array, obs_ids)
        ]

        df = pd.concat(df_list, axis=1)

        # Strip the structural prefix from shared / derived columns. Per-obs
        # columns already display "<rest>\n[<obs>]" with no prefix to strip.
        df = df.rename(
            columns=lambda col: (
                col.split(".", maxsplit=1)[1] if "." in col and "\n[" not in col else col
            )
        )

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

        x_unit = u.Unit(x_unit)
        y_units = _resolve_y_units(y_type, x_unit)
        figure_list = []

        with plt.style.context(style):
            for obs_id, obsconf in self.obsconfs.items():
                fig, ax = plt.subplots(
                    2, 1, figsize=figsize, sharex="col", height_ratios=[0.7, 0.3]
                )

                count = az.extract(
                    self.inference_data,
                    var_names=f"observed.{obs_id}",
                    group="posterior_predictive",
                ).values.T
                xbins, exposure, integrated_arf = _compute_effective_area(obsconf, x_unit)
                observed_counts = obsconf.folded_counts.data
                bin_ids = _compute_bin_ids(observed_counts, min_counts, grouping)

                count, observed_counts, xbins, integrated_arf = _apply_binning(
                    bin_ids, count, observed_counts, xbins, integrated_arf
                )

                denominator = _compute_denominator(y_type, exposure, integrated_arf, xbins)
                y_samples = (count * u.ct / denominator).to(y_units)
                y_observed, y_observed_low, y_observed_high = _error_bars_for_observed_data(
                    observed_counts, denominator, y_units
                )

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

                legend_plots = [(true_data_plot,), *model_plot]
                legend_labels = ["Observed", "Model"]

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
                    extra_plots, extra_labels = self._plot_components_overlay(
                        ax[0],
                        obs_id,
                        denominator,
                        y_units,
                        bin_ids,
                        xbins,
                        n_sigmas,
                        alpha_envelope,
                    )
                    legend_plots += extra_plots
                    legend_labels += extra_labels

                if (
                    self.bayesian_fitter.forward_model.background.get(obs_id) is not None
                    and plot_background
                ):
                    extra_plots, extra_labels = self._plot_background_overlay(
                        ax[0],
                        obsconf,
                        obs_id,
                        denominator,
                        y_units,
                        bin_ids,
                        xbins,
                        rescale_background,
                        n_sigmas,
                        alpha_envelope,
                    )
                    legend_plots += extra_plots
                    legend_labels += extra_labels

                _style_axes(
                    ax,
                    x_unit,
                    scale,
                    x_lims,
                    residual_samples,
                    y_units,
                    xbins,
                    np.nanmin(y_observed),
                    np.nanmax(y_observed),
                    legend_plots,
                    legend_labels,
                )

                fig.align_ylabels()
                plt.subplots_adjust(hspace=0.0)
                fig.suptitle(f"Posterior predictive - {obs_id}" if title is None else title)
                fig.tight_layout()
                figure_list.append(fig)

        plt.tight_layout()
        plt.show()

        return figure_list

    def _plot_components_overlay(
        self,
        ax,
        obs_id,
        denominator,
        y_units,
        bin_ids,
        xbins,
        n_sigmas,
        alpha_envelope,
    ) -> tuple[list, list]:
        """Overlay per-component posterior bands; return legend entries to append."""
        extra_plots: list = []
        extra_labels: list = []
        for (component_name, comp_count), color in zip(
            self._ppc_folded_branches(obs_id).items(), COLOR_CYCLE
        ):
            # _ppc_folded_branches returns (n_chains, n_draws, n_bins) — flatten chains/draws.
            comp_flat = comp_count.reshape((comp_count.shape[0] * comp_count.shape[1], -1))
            if bin_ids is not None:
                comp_flat = rebin_counts(comp_flat, bin_ids)
            y_samples = (comp_flat * u.ct / denominator).to(y_units)
            component_plot = _plot_binned_samples_with_error(
                ax,
                xbins.value,
                y_samples.value,
                color=color,
                linestyle="dashdot",
                n_sigmas=n_sigmas,
                alpha_envelope=alpha_envelope,
            )
            extra_plots += component_plot
            extra_labels.append(component_name.split("*")[-1])
        return extra_plots, extra_labels

    def _plot_background_overlay(
        self,
        ax,
        obsconf,
        obs_id,
        denominator,
        y_units,
        bin_ids,
        xbins,
        rescale_background,
        n_sigmas,
        alpha_envelope,
    ) -> tuple[list, list]:
        """Overlay the background model/data; return legend entries to append."""
        bkg_count = az.extract(
            self.inference_data,
            var_names=f"observed_background.{obs_id}",
            group="posterior_predictive",
        ).values.T
        bkg_observed = obsconf.folded_background.data

        if bin_ids is not None:
            bkg_count = rebin_counts(bkg_count, bin_ids)
            bkg_observed = rebin_counts(bkg_observed, bin_ids)
            rescale_background_factor = (
                rebin_counts(obsconf.folded_backratio.data, bin_ids) / np.bincount(bin_ids)
                if rescale_background
                else 1.0
            )
        else:
            rescale_background_factor = obsconf.folded_backratio.data if rescale_background else 1.0

        y_samples_bkg = (bkg_count * u.ct / denominator).to(y_units)
        y_observed_bkg, y_observed_bkg_low, y_observed_bkg_high = _error_bars_for_observed_data(
            bkg_observed, denominator, y_units
        )
        model_bkg_plot = _plot_binned_samples_with_error(
            ax,
            xbins.value,
            y_samples_bkg.value * rescale_background_factor,
            color=BACKGROUND_COLOR,
            alpha_envelope=alpha_envelope,
            n_sigmas=n_sigmas,
        )
        true_bkg_plot = _plot_poisson_data_with_error(
            ax,
            xbins.value,
            y_observed_bkg.value * rescale_background_factor,
            y_observed_bkg_low.value * rescale_background_factor,
            y_observed_bkg_high.value * rescale_background_factor,
            color=BACKGROUND_DATA_COLOR,
            alpha=0.7,
        )
        return [(true_bkg_plot,), *model_bkg_plot], ["Observed (bkg)", "Model (bkg)"]

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


# ---- Module-level helpers for plot_ppc ----------------------------------------


def _resolve_y_units(y_type, x_unit):
    if y_type == "photon_flux_density":
        return u.ct / u.cm**2 / u.s / x_unit
    units = _Y_UNITS_FOR_TYPE.get(y_type)
    if units is None:
        raise ValueError(
            f"Unknown y_type: {y_type}. Must be 'counts', 'countrate', 'photon_flux' "
            f"or 'photon_flux_density'"
        )
    return units


def _apply_binning(bin_ids, count, observed_counts, xbins, integrated_arf):
    if bin_ids is None:
        return count, observed_counts, xbins, integrated_arf
    count = rebin_counts(count, bin_ids)
    observed_counts = rebin_counts(observed_counts, bin_ids)
    xbins = _rebin_xbins(xbins, bin_ids)
    integrated_arf = rebin_counts(integrated_arf.value, bin_ids) * integrated_arf.unit
    return count, observed_counts, xbins, integrated_arf


def _compute_denominator(y_type, exposure, integrated_arf, xbins):
    if y_type == "counts":
        return 1
    if y_type == "countrate":
        return exposure
    if y_type == "photon_flux":
        return integrated_arf * exposure
    if y_type == "photon_flux_density":
        return (xbins[1] - xbins[0]) * integrated_arf * exposure
    raise ValueError(f"Unknown y_type: {y_type}")


def _style_axes(
    ax,
    x_unit,
    scale,
    x_lims,
    residual_samples,
    y_units,
    xbins,
    lowest_y,
    highest_y,
    legend_plots,
    legend_labels,
):
    max_residuals = min(3.5, np.nanmax(np.abs(residual_samples)))
    ax[0].loglog()
    ax[1].set_ylim(-np.nanmax([3.5, max_residuals]), +np.nanmax([3.5, max_residuals]))
    ax[0].set_ylabel(f"Folded spectrum\n [{y_units:latex_inline}]")
    ax[1].set_ylabel("Residuals \n" + r"[$\sigma$]")

    physical_type = getattr(x_unit, "physical_type")
    # astropy's PhysicalType supports __eq__ with bare strings but its str()
    # form is "energy/torque/work" for compound types, so iterate explicitly.
    label = next(
        (lbl for pt, lbl in _XLABEL_FOR_PHYSICAL_TYPE.items() if physical_type == pt), None
    )
    if label is None:
        raise RuntimeError(
            f"Unknown physical type for x_units: {x_unit}. "
            f"Must be 'length', 'energy' or 'frequency'"
        )
    ax[1].set_xlabel(f"{label} \n[{x_unit:latex_inline}]")

    ax[1].axhline(0, color=SPECTRUM_DATA_COLOR, ls="--")
    ax[1].axhline(-3, color=SPECTRUM_DATA_COLOR, ls=":")
    ax[1].axhline(3, color=SPECTRUM_DATA_COLOR, ls=":")
    ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
    ax[1].set_yticks(range(-3, 4), minor=True)

    ax[0].set_xlim(xbins.value.min(), xbins.value.max())
    ax[0].set_ylim(lowest_y.value * 0.8, highest_y.value * 1.2)
    ax[0].legend(legend_plots, legend_labels)

    xscale, yscale = _SCALE_TO_AXES[scale]
    ax[0].set_xscale(xscale)
    ax[0].set_yscale(yscale)

    if x_lims is not None:
        ax[0].set_xlim(*x_lims)


# ---- Module-level helpers for input_parameters --------------------------------


def _resolve_shared_entry(
    value, base, obs_axis, data_vars, chain_draw, deferred_ties
) -> ArrayLike | None:
    """Materialise a shared (unscoped) prior entry as an obs-broadcast array.

    Returns ``None`` and appends to ``deferred_ties`` when the entry is a
    ``TiedParameter`` — the caller should skip this base for now.
    """
    if isinstance(value, TiedParameter):
        deferred_ties.append((base, value, obs_axis))
        return None
    if isinstance(value, dist.Distribution):
        arr = data_vars[base]
    else:
        fixed = jnp.asarray(value)
        arr = jnp.broadcast_to(fixed, (*chain_draw, *fixed.shape))
    return jnp.broadcast_to(arr[..., None], (*arr.shape, len(obs_axis)))


def _resolve_per_obs_entry(
    scopes, base, prefix, rest, obs_axis, data_vars, chain_draw, deferred_ties
) -> ArrayLike | dict | None:
    """Materialise a per-obs prior entry; stack across obs when shapes agree.

    Returns ``None`` when there's nothing to assemble (empty obs_axis, or a
    ``TiedParameter`` that was deferred).
    """
    per_obs: dict[str, ArrayLike] = {}
    for obs in obs_axis:
        value = scopes.get(obs, scopes.get("*"))
        if value is None:
            continue
        if isinstance(value, TiedParameter):
            deferred_ties.append((base, value, obs_axis))
            return None
        site_name = f"forward.{prefix}.{obs}.{rest}"
        if isinstance(value, dist.Distribution):
            per_obs[obs] = data_vars[site_name]
        else:
            fixed = jnp.asarray(value)
            per_obs[obs] = jnp.broadcast_to(fixed, (*chain_draw, *fixed.shape))

    if not per_obs:
        return None
    shapes = {arr.shape for arr in per_obs.values()}
    if len(shapes) == 1:
        return jnp.stack(list(per_obs.values()), axis=-1)
    return per_obs


def _apply_tied_resolutions(out, deferred_ties) -> None:
    """Resolve every deferred TiedParameter once all direct entries are in ``out``."""
    for dest_base, tied, _obs_axis in deferred_ties:
        if tied.tied_to not in out:
            raise ValueError(
                f"TiedParameter {dest_base!r} references unknown source {tied.tied_to!r}"
            )
        source = out[tied.tied_to]
        if isinstance(source, dict):
            out[dest_base] = {obs: tied.func(v) for obs, v in source.items()}
        else:
            out[dest_base] = tied.func(source)
