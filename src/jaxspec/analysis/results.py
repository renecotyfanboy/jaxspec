from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import astropy.cosmology.units as cu
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from astropy.cosmology import Cosmology, Planck18
from astropy.units import Unit
from chainconsumer import Chain, ChainConsumer, PlotConfig
from jax.typing import ArrayLike
from scipy.special import gammaln

from ..fit._prior_resolution import (
    _DERIVED_PREFIX,
    _KNOWN_PREFIXES,
    _SITE_PREFIX,
    _parse_per_obs_site_name,
)
from . import _ppc
from ._posterior_params import build_input_parameters

if TYPE_CHECKING:
    from ..fit import BayesianModel


_CHAIN_KEEP_PREFIXES: tuple[str, ...] = (
    _DERIVED_PREFIX,
    *(f"{prefix}." for prefix in _KNOWN_PREFIXES),
    *(f"{_SITE_PREFIX}{prefix}." for prefix in _KNOWN_PREFIXES),
)


class FitResult:
    """
    Container for the result of a fit using any ModelFitter class.
    """

    def __init__(
        self,
        bayesian_fitter: BayesianModel,
        inference_data: az.InferenceData,
    ):
        self.model = bayesian_fitter.spectral_model
        self.bayesian_fitter = bayesian_fitter
        self.inference_data = inference_data
        self.obsconfs = bayesian_fitter.forward_model.observations

    @property
    def converged(self) -> bool:
        r"""
        Convergence of the chain as computed by the $\hat{R}$ statistic.
        """
        rhat = az.rhat(self.inference_data)

        return bool((rhat.to_array() < 1.01).all())

    @cached_property
    def input_parameters(self) -> dict[str, ArrayLike]:
        """
        The input parameters of the model, keyed by the user's prior dict
        entry paths (e.g. ``"spectrum.powerlaw_1.alpha"``,
        ``"instrument.gain.factor"``, ``"background.countrate"``).

        Shared parameters are broadcast along a trailing observation axis and
        per-observation samples are stacked on that same axis when every
        applicable observation is present with the same shape. Ragged
        per-observation entries, and entries covering only a subset of the
        applicable observations, are kept as ``{observation_name: array}``.

        Returns an empty dict when the user passed a callable prior — there's
        no static key set to enumerate.
        """
        return build_input_parameters(self.bayesian_fitter, self.inference_data)

    @cached_property
    def spectrum_parameters(self) -> dict[str, ArrayLike]:
        """Subset of ``input_parameters`` belonging to the spectral model."""
        return {k: v for k, v in self.input_parameters.items() if k.startswith("spectrum.")}

    def _register_derived_parameter(self, name: str, value: ArrayLike) -> None:
        """Store a derived quantity (flux, luminosity) alongside the posterior.

        ``value`` must start with ``(chain, draw)`` axes. Each remaining axis is stored
        under a generated ``derived_dim_<index>`` dimension name.
        """
        value = np.asarray(value)
        extra_dims = tuple(f"derived_dim_{i}" for i in range(value.ndim - 2))

        self.inference_data.posterior[name] = (("chain", "draw", *extra_dims), value)

    def _band_flux(
        self,
        e_min: float,
        e_max: float,
        *,
        energy: bool,
        base_unit: Unit,
        unit: Unit,
        kind: str,
        register: bool,
        n_points: int,
        n_grid: int,
    ) -> ArrayLike:
        """Integrate the unfolded model over ``[e_min, e_max]`` and convert to ``unit``.

        Shared by ``photon_flux``, ``energy_flux`` and ``luminosity``, which
        differ only in ``energy``, the unit the integral comes out in, and the name the
        result is registered under.
        """
        flux = self.model.integrated_flux(
            e_min,
            e_max,
            params=self.spectrum_parameters,
            energy=energy,
            n_points=n_points,
            n_grid=n_grid,
        )
        value = np.asarray(flux * float(base_unit.to(unit)))

        if register:
            self._register_derived_parameter(f"derived.{kind}_{e_min:.1f}_{e_max:.1f}", value)

        return value

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
        return self._band_flux(
            e_min,
            e_max,
            energy=False,
            base_unit=u.photon / u.cm**2 / u.s,
            unit=unit,
            kind="photon_flux",
            register=register,
            n_points=n_points,
            n_grid=n_grid,
        )

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
        return self._band_flux(
            e_min,
            e_max,
            energy=True,
            base_unit=u.keV / u.cm**2 / u.s,
            unit=unit,
            kind="energy_flux",
            register=register,
            n_points=n_points,
            n_grid=n_grid,
        )

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

        # ``derived.`` prefixes the ordinary grammar; strip it before parsing.
        per_obs = _parse_per_obs_site_name(varname.removeprefix(_DERIVED_PREFIX))
        if per_obs is not None:
            if extra_dims:
                return []
            user_path, obs_seg = per_obs
            df = array.to_pandas()
            df.name = f"{user_path.split('.', 1)[1]}\n[{obs_seg}]"
            return [df]

        if extra_dims:
            dim = extra_dims[0]
            dfs = []
            for coord, obs_id in zip(array.coords[dim], obs_ids):
                df = array.loc[{dim: coord}].to_pandas()
                df.name += f"\n[{obs_id}]"
                dfs.append(df)
            return dfs

        df = array.to_pandas()
        # Drop ``derived.`` from a component's quantity so the caller's one-segment
        # strip lands on the component. Post-fit ``derived.photon_flux_*`` names have
        # no model prefix and keep theirs.
        remainder = varname.removeprefix(_DERIVED_PREFIX)
        if varname != remainder and remainder.startswith(tuple(f"{p}." for p in _KNOWN_PREFIXES)):
            df.name = remainder
        return [df]

    def to_chain(self, name: str) -> Chain:
        """
        Return a ChainConsumer Chain object from the posterior distribution of the parameters_type.

        Parameters:
            name: The name of the chain.
        """
        deterministic_sites = self.bayesian_fitter._deterministic_site_names

        def is_dropped(key: str) -> bool:
            if not key.startswith(_CHAIN_KEEP_PREFIXES):
                return True
            # Deterministic sites are functions of the free parameters; keep only
            # the ones a model publishes on purpose under ``derived.``.
            return key in deterministic_sites and not key.startswith(_DERIVED_PREFIX)

        keys_to_drop = [key for key in self.inference_data.posterior.keys() if is_dropped(str(key))]

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

        return _ppc.plot_ppc(
            self,
            n_sigmas=n_sigmas,
            x_unit=x_unit,
            y_type=y_type,
            plot_background=plot_background,
            plot_components=plot_components,
            scale=scale,
            alpha_envelope=alpha_envelope,
            style=style,
            title=title,
            figsize=figsize,
            x_lims=x_lims,
            rescale_background=rescale_background,
            min_counts=min_counts,
            grouping=grouping,
        )

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

        with plt.style.context("default"):
            return consumer.plotter.plot(**kwargs)
