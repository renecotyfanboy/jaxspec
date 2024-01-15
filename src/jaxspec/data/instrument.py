import os
import numpy as np
import xarray as xr
from .ogip import DataARF, DataRMF


class Instrument(xr.Dataset):
    __slots__ = (
        "redistribution",
        "area",
        "e_min_out",
        "e_max_out",
        "e_min_in",
        "e_max_in",
    )

    @classmethod
    def from_matrix(cls, redistribution_matrix, spectral_response, e_min_unfolded, e_max_unfolded, e_min_channel, e_max_channel):
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
    def from_ogip_file(cls, arf_file: str | os.PathLike, rmf_file: str | os.PathLike, **kwargs):
        """
        Load the data from OGIP files.

        Parameters:
            arf_file: The ARF file path.
            rmf_file: The RMF file path.
            exposure: The exposure time in second.
            grouping: The grouping matrix.
        """

        arf = DataARF.from_file(arf_file)
        rmf = DataRMF.from_file(rmf_file)

        return cls.from_matrix(rmf.matrix, arf.specresp, rmf.energ_lo, rmf.energ_hi, rmf.e_min, rmf.e_max)

    def plot_redistribution(self, **kwargs):
        """
        Plot the redistribution probability matrix

        Parameters:
            **kwargs : `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.plot.pcolormesh.html#xarray.plot.pcolormesh
        """
        import cmasher as cmr

        return xr.plot.pcolormesh(
            self.redistribution,
            x="e_max_unfolded",
            y="e_max_channel",
            xscale="log",
            yscale="log",
            cmap=cmr.cosmic,
            vmin=0,
            vmax=0.075,
            add_labels=True,
            **kwargs,
        )

    def plot_area(self, **kwargs):
        """
        Plot the effective area

        Parameters:
            **kwargs : `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.DataArray.plot.line.html#xarray.DataArray.plot.line
        """

        return self.area.plot.step(x="e_min_unfolded", xscale="log", yscale="log", where="post", **kwargs)
