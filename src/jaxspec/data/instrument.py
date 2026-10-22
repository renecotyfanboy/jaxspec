import os

import numpy as np
import xarray as xr

from matplotlib import colors

from ._validation import (
    detector_channels,
    energy_bins,
    nonnegative_values,
    quantity_values,
    response_matrix,
)
from .ogip import DataARF, DataRMF


class Instrument(xr.Dataset):
    """Instrument calibration with distinct detector-channel and incident-energy axes."""

    redistribution: xr.DataArray
    """Dimensionless response weights on (instrument_channel, unfolded_channel)."""
    area: xr.DataArray
    """Effective area in cm² for each incident photon-energy bin (unfolded_channel)."""

    __slots__ = (
        "area",
        "e_max_in",
        "e_max_out",
        "e_min_in",
        "e_min_out",
        "redistribution",
    )

    @classmethod
    def from_matrix(
        cls,
        redistribution_matrix,
        spectral_response,
        e_min_unfolded,
        e_max_unfolded,
        e_min_channel,
        e_max_channel,
        *,
        channel=None,
    ):
        """Build a response, optionally retaining detector channel identifiers.

        Supply ``channel`` when matching a PHA spectrum whose detector channels
        start at one or contain gaps; channel labels then control row alignment.
        Labels must be increasing and unique. Without them, matching is positional
        and the response rows must already follow the observation's channel order.
        ``redistribution_matrix`` has shape ``(n_detector_channels, n_incident_bins)``;
        ``spectral_response`` has one effective area per incident bin. Response
        weights are dimensionless and need not sum to one in each column.
        Plain energy and effective-area arrays use keV and cm² respectively.
        Astropy Quantity inputs are converted to those units explicitly.
        Dense arrays, SciPy sparse arrays/matrices and zero-filled PyData
        sparse arrays are stored as one sparse COO representation, so all
        accepted inputs can subsequently be used for observation folding.
        """
        e_min_unfolded, e_max_unfolded = energy_bins(
            quantity_values(e_min_unfolded, "keV", name="Photon lower energies"),
            quantity_values(e_max_unfolded, "keV", name="Photon upper energies"),
            name="Photon energies",
            roundoff_dtype=np.result_type(np.asarray(e_min_unfolded), np.asarray(e_max_unfolded)),
        )
        e_min_channel, e_max_channel = energy_bins(
            quantity_values(e_min_channel, "keV", name="Detector lower energies"),
            quantity_values(e_max_channel, "keV", name="Detector upper energies"),
            name="Detector energies",
            roundoff_dtype=np.result_type(np.asarray(e_min_channel), np.asarray(e_max_channel)),
            ordered=False,
        )
        spectral_response = nonnegative_values(
            quantity_values(spectral_response, "cm2", name="Effective area"),
            name="Effective area",
            size=len(e_min_unfolded),
        )
        redistribution_matrix = response_matrix(
            redistribution_matrix, shape=(len(e_min_channel), len(e_min_unfolded))
        )
        return cls._from_validated_matrix(
            redistribution_matrix,
            spectral_response,
            e_min_unfolded,
            e_max_unfolded,
            e_min_channel,
            e_max_channel,
            channel=channel,
        )

    @classmethod
    def _from_validated_matrix(
        cls,
        redistribution_matrix,
        spectral_response,
        e_min_unfolded,
        e_max_unfolded,
        e_min_channel,
        e_max_channel,
        *,
        channel=None,
    ):
        """Store canonical response values after validating their native input precision.

        Both array and OGIP entry points validate before reaching this builder.
        Rechecking canonical float64 edges would lose the original float32 FITS
        precision used to identify harmless calibration boundary roundoff.
        """
        result = cls(
            {
                "redistribution": (
                    ["instrument_channel", "unfolded_channel"],
                    redistribution_matrix,
                    {"description": "Redistribution matrix"},
                ),
                "area": (
                    ["unfolded_channel"],
                    spectral_response,
                    {"description": "Effective area", "units": "cm^2"},
                ),
            },
            coords={
                "e_min_unfolded": (
                    ["unfolded_channel"],
                    e_min_unfolded,
                    {"description": "Low bin energy for ingoing channels", "units": "keV"},
                ),
                "e_max_unfolded": (
                    ["unfolded_channel"],
                    e_max_unfolded,
                    {"description": "High bin energy for ingoing channels", "units": "keV"},
                ),
                "e_min_channel": (
                    ["instrument_channel"],
                    e_min_channel,
                    {"description": "Low bin energy for outgoing channels", "units": "keV"},
                ),
                "e_max_channel": (
                    ["instrument_channel"],
                    e_max_channel,
                    {"description": "High bin energy for outgoing channels", "units": "keV"},
                ),
            },
            attrs={"description": "X-ray instrument response dataset"},
        )
        if channel is not None:
            labels = detector_channels(
                channel,
                size=result.sizes["instrument_channel"],
                name="Response channel identifiers",
            )
            result = result.assign_coords(channel=("instrument_channel", labels))
        return result

    @classmethod
    def from_ogip_file(cls, rmf_path: str | os.PathLike, arf_path: str | os.PathLike | None = None):
        """
        Load the data from OGIP files.

        Parameters:
            rmf_path: The RMF file path.
            arf_path: The ARF file path.

        Supply the matched ARF for a redistribution-only RMF. Without an ARF,
        the response is interpreted as a combined RSP including effective area.
        Explicit MATRIX area units are converted to cm²; area units or
        HDUCLAS3=FULL prohibit applying a second ARF. Unclassified unitless
        legacy files retain this caller-selected RMF/RSP convention.
        """

        rmf = DataRMF.from_file(rmf_path)

        if arf_path is not None:
            if getattr(rmf, "includes_effective_area", False):
                raise ValueError(
                    "This response MATRIX already includes effective area, as declared by its "
                    "area units or HDUCLAS3=FULL. Supply it without an additional ARF."
                )
            arf = DataARF.from_file(arf_path)
            if (
                np.shape(arf.energ_lo) != np.shape(rmf.energ_lo)
                or not np.allclose(arf.energ_lo, rmf.energ_lo, rtol=5e-7, atol=0)
                or not np.allclose(arf.energ_hi, rmf.energ_hi, rtol=5e-7, atol=0)
            ):
                raise ValueError(
                    "ARF and RMF photon-energy grids differ. Supply a matched response pair "
                    "or rebin the calibration files consistently before fitting."
                )
            specresp = arf.specresp

        else:
            # Combined RSP files already include effective area. Factor them
            # without allocating a dense calorimeter redistribution matrix.
            specresp = np.asarray(rmf.sparse_matrix.sum(axis=0).todense())
            rmf.sparse_matrix = rmf.sparse_matrix / np.where(specresp > 0, specresp, 1.0)

        nonnegative_values(specresp, name="Effective area", size=len(rmf.energ_lo))
        result = cls._from_validated_matrix(
            rmf.sparse_matrix,
            specresp,
            rmf.energ_lo,
            rmf.energ_hi,
            rmf.e_min,
            rmf.e_max,
            channel=rmf.channel,
        )
        result.attrs.update(
            response_matrix_file=str(rmf_path),
            ancillary_response_file=None if arf_path is None else str(arf_path),
            response_matrix_unit_original=getattr(rmf, "matrix_unit_original", None),
            response_matrix_compatibility_notes=getattr(rmf, "compatibility_notes", ()),
        )
        return result

    def plot_redistribution(
        self,
        xscale: str = "log",
        yscale: str = "log",
        cmap=None,
        vmin: float = 1e-6,
        vmax: float = 1e0,
        add_labels: bool = True,
        **kwargs,
    ):
        """
        Plot the redistribution probability matrix

        Parameters:
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            cmap (str | Colormap | None): The colormap to use.
            vmin: The minimum value for the colormap.
            vmax: The maximum value for the colormap.
            add_labels: Whether to add labels to the plot.
            **kwargs (dict): `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.plot.pcolormesh.html#xarray.plot.pcolormesh
        """

        import cmasher as cmr

        return xr.plot.pcolormesh(
            self.redistribution,
            x="e_max_unfolded",
            y="e_max_channel",
            xscale=xscale,
            yscale=yscale,
            cmap=cmr.ember_r if cmap is None else cmap,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            add_labels=add_labels,
            **kwargs,
        )

    def plot_area(self, xscale: str = "log", yscale: str = "log", where: str = "post", **kwargs):
        """
        Plot the effective area

        Parameters:
            xscale: The scale of the x-axis.
            yscale: The scale of the y-axis.
            where: The position of the steps.
            **kwargs (dict): `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.DataArray.plot.line.html#xarray.DataArray.plot.line
        """

        return self.area.plot.step(
            x="e_min_unfolded", xscale=xscale, yscale=yscale, where=where, **kwargs
        )
