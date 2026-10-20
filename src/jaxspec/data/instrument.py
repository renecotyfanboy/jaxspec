import os

import numpy as np
import sparse
import xarray as xr

from matplotlib import colors

from ._validation import (
    detector_channels,
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
    ):
        return cls(
            {
                "redistribution": (
                    ["instrument_channel", "unfolded_channel"],
                    redistribution_matrix,
                    {"description": "Redistribution matrix"},
                ),
                "area": (
                    ["unfolded_channel"],
                    np.array(spectral_response, dtype=np.float64),
                    {"description": "Effective area", "units": "cm^2"},
                ),
            },
            coords={
                "e_min_unfolded": (
                    ["unfolded_channel"],
                    np.array(e_min_unfolded, dtype=np.float64),
                    {"description": "Low bin energy for ingoing channels", "units": "keV"},
                ),
                "e_max_unfolded": (
                    ["unfolded_channel"],
                    np.array(e_max_unfolded, dtype=np.float64),
                    {"description": "High bin energy for ingoing channels", "units": "keV"},
                ),
                "e_min_channel": (
                    ["instrument_channel"],
                    np.array(e_min_channel, dtype=np.float64),
                    {"description": "Low bin energy for outgoing channels", "units": "keV"},
                ),
                "e_max_channel": (
                    ["instrument_channel"],
                    np.array(e_max_channel, dtype=np.float64),
                    {"description": "High bin energy for outgoing channels", "units": "keV"},
                ),
            },
            attrs={"description": "X-ray instrument response dataset"},
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
        """

        rmf = DataRMF.from_file(rmf_path)

        if arf_path is not None:
            specresp = DataARF.from_file(arf_path).specresp

        else:
            specresp = rmf.matrix.sum(axis=0)
            rmf.sparse_matrix = sparse.COO(rmf.matrix / specresp)

        return cls.from_matrix(
            rmf.sparse_matrix, specresp, rmf.energ_lo, rmf.energ_hi, rmf.e_min, rmf.e_max
        )

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
            cmap: The colormap to use.
            vmin: The minimum value for the colormap.
            vmax: The maximum value for the colormap.
            add_labels: Whether to add labels to the plot.
            **kwargs: `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.plot.pcolormesh.html#xarray.plot.pcolormesh
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
            **kwargs: `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.DataArray.plot.line.html#xarray.DataArray.plot.line
        """

        return self.area.plot.step(
            x="e_min_unfolded", xscale=xscale, yscale=yscale, where=where, **kwargs
        )
