import numpy as np
import xarray as xr

from ._validation import (
    channel_vector,
    detector_channels,
    exposure_seconds,
    poisson_counts,
    poisson_grouping,
    split_background_groups,
)
from .ogip import DataPHA


class Observation(xr.Dataset):
    """Raw event spectra and their detector-channel grouping.

    Grouped counts here retain all channels assigned to each stored group.
    ``ObsConfiguration.from_instrument`` applies quality and energy selection
    when matching the observation to its response.
    """

    counts: xr.DataArray
    """Source-aperture event counts on instrument_channel."""
    folded_counts: xr.DataArray
    """Source event counts on folded_channel, grouped before quality selection."""
    grouping: xr.DataArray
    """Zero-or-one event-sum weights on (folded_channel, instrument_channel)."""
    quality: xr.DataArray
    """Flags per raw detector channel; zero marks usable measurements."""
    exposure: xr.DataArray
    """Source exposure in seconds."""
    background: xr.DataArray
    """Background-aperture event counts on instrument_channel; zeros if absent."""
    folded_background: xr.DataArray
    """Background event counts on folded_channel, grouped before quality selection."""

    __slots__ = (
        "background",
        "channel",
        "counts",
        "exposure",
        "folded_background",
        "folded_counts",
        "grouping",
        "quality",
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
        backratio=1.0,
        attributes: dict | None = None,
        *,
        areascal=1.0,
    ):
        """Build raw and grouped event spectra with explicit extraction scaling.

        ``counts`` and ``background`` contain event counts per raw detector
        channel. ``grouping`` has shape ``(n_groups, n_channels)`` and sums
        disjoint channel sets with zero-or-one weights.
        ``backratio`` converts expected background-region counts to expected
        source-region background counts. ``areascal`` multiplies the folded
        source response per detector channel. Groups crossing unequal background
        ratios are refined so each likelihood bin has one exact scaling factor.
        Exposure is in seconds unless an explicit Astropy time Quantity is supplied.
        """
        if attributes is None:
            attributes = {}

        counts = poisson_counts(counts)
        exposure = exposure_seconds(exposure, name="Observation exposure")
        channel = detector_channels(channel, size=len(counts), name="PHA channel identifiers")
        if background is None:
            background = np.zeros_like(counts, dtype=np.int64)
        else:
            background = poisson_counts(background, name="background counts")
        if background.shape != counts.shape:
            raise ValueError("Source and background counts must have matching channel shapes.")
        quality = channel_vector(quality, len(counts), name="QUALITY", dtype=int)
        backratio = channel_vector(backratio, len(counts), name="background scaling")
        areascal = channel_vector(areascal, len(counts), name="AREASCAL")
        if np.any(backratio[quality == 0] <= 0) or np.any(areascal[quality == 0] <= 0):
            raise ValueError("Background scaling and AREASCAL must be positive in usable channels.")
        grouping = poisson_grouping(grouping, len(counts))
        grouping, split_count = split_background_groups(grouping, backratio)
        attributes = dict(attributes)
        attributes["background_scale_group_splits"] = split_count

        data_dict = {
            "counts": (
                ["instrument_channel"],
                counts,
                {"description": "Counts", "unit": "photons"},
            ),
            "folded_counts": (
                ["folded_channel"],
                poisson_counts(np.ma.filled(grouping @ counts), name="grouped counts"),
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
            "exposure": ([], exposure, {"description": "Total exposure", "unit": "s"}),
            "areascal": (
                ["instrument_channel"],
                areascal,
                {"description": "OGIP source response area scaling", "unit": "1"},
            ),
            "backratio": (
                ["instrument_channel"],
                np.asarray(backratio, dtype=float),
                {
                    "description": "Scale from expected background-aperture counts to source-aperture "
                    "background counts; PHA inputs include exposure, BACKSCAL and AREASCAL ratios."
                },
            ),
            "folded_backratio": (
                ["folded_channel"],
                np.asarray(
                    np.ma.filled(grouping @ backratio) / grouping.sum(axis=1).todense(), dtype=float
                ),
                {"description": "Background scaling after grouping"},
            ),
            "background": (
                ["instrument_channel"],
                background,
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
                    channel,
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
        """Match source/background channels and preserve their Poisson counting data.

        Background-subtracted or explicitly non-Poisson spectra cannot define
        this likelihood. Supply the original TOTAL source and BKG spectra.
        """
        for label, spectrum in (("Source", pha), ("Background", bkg)):
            if spectrum is not None and set(spectrum.flags) & {"NET", "NONPOISSON"}:
                raise ValueError(
                    f"{label} spectrum is background-subtracted or non-Poisson. "
                    "Supply original TOTAL source and Poisson BKG event spectra."
                )
        quality = pha.quality.copy()
        if bkg is not None:
            indices = np.searchsorted(bkg.channel, pha.channel)
            if np.any(indices >= len(bkg.channel)) or not np.array_equal(
                bkg.channel[np.minimum(indices, len(bkg.channel) - 1)], pha.channel
            ):
                raise ValueError(
                    "Background spectrum is missing source detector channel identifiers."
                )
            quality = np.where(quality != 0, quality, bkg.quality[indices])
            valid = quality == 0
            for label, values in (
                ("source BACKSCAL", pha.backscal),
                ("source AREASCAL", pha.areascal),
                ("background BACKSCAL", bkg.backscal[indices]),
                ("background AREASCAL", bkg.areascal[indices]),
            ):
                if np.any(values[valid] <= 0):
                    raise ValueError(f"{label} must be positive in usable channels.")
            backratio = np.divide(
                pha.backscal * pha.exposure * pha.areascal,
                bkg.backscal[indices] * bkg.exposure * bkg.areascal[indices],
                out=np.ones_like(pha.areascal),
                where=valid,
            )
            background = bkg.counts[indices]
            metadata["background_exposure"] = bkg.exposure
        else:
            backratio = np.ones_like(pha.counts)
            background = None

        return cls.from_matrix(
            pha.counts,
            pha.grouping,
            pha.channel,
            quality,
            pha.exposure,
            backratio=backratio,
            background=background,
            attributes=metadata,
            areascal=pha.areascal,
        )

    @classmethod
    def from_pha_file(cls, pha_path: str, bkg_path: str | None = None, **metadata):
        """
        Build an observation from a PHA file

        Parameters:
            pha_path: Path to the PHA file
            bkg_path: Path to the background file
            **metadata (dict): Additional metadata to add to the observation
        """
        from .util import data_path_finder

        arf_path, rmf_path, bkg_path_default = data_path_finder(
            pha_path, require_arf=False, require_rmf=False, require_bkg=False
        )
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
            **kwargs (dict): `kwargs` passed to https://docs.xarray.dev/en/latest/generated/xarray.DataArray.plot.step.html#xarray.DataArray.plot.line
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
