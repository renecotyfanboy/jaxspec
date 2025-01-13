import numpy as np
import xarray as xr

from .ogip import DataPHA


class Observation(xr.Dataset):
    """
    Class to store the data of an observation
    """

    counts: xr.DataArray
    """The observed counts"""
    folded_counts: xr.DataArray
    """The observed counts, after grouping"""
    grouping: xr.DataArray
    """The grouping matrix"""
    quality: xr.DataArray
    """The quality flag"""
    exposure: xr.DataArray
    """The total exposure"""
    background: xr.DataArray
    """The background counts if provided, otherwise 0"""
    folded_background: xr.DataArray
    """The background counts, after grouping"""

    __slots__ = (
        "grouping",
        "channel",
        "quality",
        "exposure",
        "background",
        "folded_background",
        "counts",
        "folded_counts",
    )

    _default_attributes = {"description": "X-ray observation dataset"}

    @classmethod
    def from_matrix(
        cls,
        counts,
        grouping,
        channel,
        quality,
        exposure,
        background=None,
        background_unscaled=None,
        backratio=1.0,
        attributes: dict | None = None,
    ):
        if attributes is None:
            attributes = {}

        if background is None or background_unscaled is None:
            background = np.zeros_like(counts, dtype=np.int64)
            background_unscaled = np.zeros_like(counts, dtype=np.int64)

        data_dict = {
            "counts": (
                ["instrument_channel"],
                np.asarray(counts, dtype=np.int64),
                {"description": "Counts", "unit": "photons"},
            ),
            "folded_counts": (
                ["folded_channel"],
                np.asarray(np.ma.filled(grouping @ counts), dtype=np.int64),
                {"description": "Folded counts, after grouping", "unit": "photons"},
            ),
            "grouping": (
                ["folded_channel", "instrument_channel"],
                grouping,
                {"description": "Grouping matrix."},
            ),
            "quality": (
                ["instrument_channel"],
                np.asarray(quality, dtype=np.int64),
                {"description": "Quality flag."},
            ),
            "exposure": ([], float(exposure), {"description": "Total exposure", "unit": "s"}),
            "backratio": (
                ["instrument_channel"],
                np.asarray(backratio, dtype=float),
                {"description": "Background scaling (SRC_BACKSCAL/BKG_BACKSCAL)"},
            ),
            "folded_backratio": (
                ["folded_channel"],
                np.asarray(np.ma.filled(grouping @ backratio), dtype=float),
                {"description": "Background scaling after grouping"},
            ),
            "background": (
                ["instrument_channel"],
                np.asarray(background, dtype=np.int64),
                {"description": "Background counts", "unit": "photons"},
            ),
            "folded_background_unscaled": (
                ["folded_channel"],
                np.asarray(np.ma.filled(grouping @ background_unscaled), dtype=np.int64),
                {"description": "Background counts", "unit": "photons"},
            ),
            "folded_background": (
                ["folded_channel"],
                np.asarray(np.ma.filled(grouping @ background), dtype=np.float64),
                {"description": "Background counts", "unit": "photons"},
            ),
        }

        return cls(
            data_dict,
            coords={
                "channel": (
                    ["instrument_channel"],
                    np.asarray(channel, dtype=np.int64),
                    {"description": "Channel number"},
                ),
                "grouped_channel": (
                    ["folded_channel"],
                    np.arange(len(grouping @ counts), dtype=np.int64),
                    {"description": "Channel number"},
                ),
            },
            attrs=cls._default_attributes
            if attributes is None
            else attributes | cls._default_attributes,
        )

    @classmethod
    def from_ogip_container(cls, pha: DataPHA, bkg: DataPHA | None = None, **metadata):
        if bkg is not None:
            backratio = np.nan_to_num(
                (pha.backscal * pha.exposure * pha.areascal)
                / (bkg.backscal * bkg.exposure * bkg.areascal)
            )
        else:
            backratio = np.ones_like(pha.counts)

        if (bkg is not None) and ("NET" in pha.flags):
            counts = pha.counts + bkg.counts * backratio
        else:
            counts = pha.counts

        return cls.from_matrix(
            counts,
            pha.grouping,
            pha.channel,
            pha.quality,
            pha.exposure,
            backratio=backratio,
            background=bkg.counts * backratio if bkg is not None else None,
            background_unscaled=bkg.counts if bkg is not None else None,
            attributes=metadata,
        )

    @classmethod
    def from_pha_file(cls, pha_path: str, bkg_path: str | None = None, **metadata):
        """
        Build an observation from a PHA file

        Parameters:
            pha_path : Path to the PHA file
            bkg_path : Path to the background file
            metadata : Additional metadata to add to the observation
        """
        from .util import data_path_finder

        arf_path, rmf_path, bkg_path_default = data_path_finder(pha_path)
        bkg_path = bkg_path_default if bkg_path is None else bkg_path

        pha = DataPHA.from_file(pha_path)
        bkg = DataPHA.from_file(bkg_path) if bkg_path is not None else None

        if metadata is None:
            metadata = {}

        metadata.update(
            observation_file=pha_path,
            background_file=bkg_path,
            response_matrix_file=rmf_path,
            ancillary_response_file=arf_path,
        )

        return cls.from_ogip_container(pha, bkg=bkg, **metadata)

    def plot_counts(self, **kwargs):
        """
        Plot the counts

        Parameters:
            **kwargs : `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.DataArray.plot.step.html#xarray.DataArray.plot.line
        """

        return self.counts.plot.step(x="instrument_channel", yscale="log", where="post", **kwargs)

    def plot_grouping(self):
        """
        Plot the grouping matrix and compare the grouped counts to the true counts
        in the original channels.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        sns.heatmap(self.grouping.data.todense().T, ax=ax, cbar=False)
        ax_histx.step(np.arange(len(self.folded_counts)), self.folded_counts, where="post")
        ax_histy.step(self.counts, np.arange(len(self.counts)), where="post")

        ax.set_xlabel("Grouped channels")
        ax.set_ylabel("Channels")
        ax_histx.set_ylabel("Grouped counts")
        ax_histy.set_xlabel("Counts")

        ax_histx.semilogy()
        ax_histy.semilogx()

        _ = [label.set_visible(False) for label in ax_histx.get_xticklabels()]
        _ = [label.set_visible(False) for label in ax_histy.get_yticklabels()]
