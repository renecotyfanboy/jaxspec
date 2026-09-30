import jax.numpy as jnp
import numpy as np
import scipy
import sparse
import xarray as xr

from jax.experimental.sparse import BCOO

from .instrument import Instrument
from .observation import Observation


def to_jax_matrix(scoo, *, sparse: bool):
    """Convert a `sparse.COO` matrix (as stored on [`ObsConfiguration`][jaxspec.data.obsconf.ObsConfiguration])
    to a JAX-typed dense array or a JAX BCOO sparse array."""
    if sparse:
        return BCOO.from_scipy_sparse(scoo.to_scipy_sparse().tocsr())
    return jnp.asarray(scoo.todense())


class ObsConfiguration(xr.Dataset):
    """Selected event counts and response arrays for predicting the fitted spectrum.

    ``folded_channel`` indexes selected, nonempty groups of usable channels.
    ``instrument_channel`` retains the matched raw detector channels, including
    channels excluded from those groups. ``unfolded_channel`` indexes retained
    incident photon-energy bins, whose coverage can extend beyond the fitted
    detector band.
    """

    transfer_matrix: xr.DataArray
    """Response on (folded_channel, unfolded_channel), in cm² s.

    Includes grouping, redistribution, effective area, exposure and source
    AREASCAL. Multiplying by integrated photon flux in photons/cm²/s predicts
    source counts in the selected groups.
    """
    redistribution: xr.DataArray
    """Dimensionless weights on (instrument_channel, unfolded_channel).

    Incident bins match the transfer matrix's columns; detector rows retain
    the matched raw channels before quality and group selection.
    """
    grouping: xr.DataArray
    """Event-sum weights on (folded_channel, instrument_channel).

    Rows match the transfer matrix's selected groups. Columns retain matched
    raw channels, with zero weight for rejected channels.
    """
    area: xr.DataArray
    """Effective area in cm² on the retained unfolded_channel bins."""
    exposure: xr.DataArray
    """Source exposure in seconds."""
    folded_counts: xr.DataArray
    """Source-aperture event counts on the selected folded_channel groups."""
    folded_background: xr.DataArray
    """Background-aperture event counts on the same selected folded_channel groups."""

    __slots__ = (
        "area",
        "exposure",
        "folded_background",
        "folded_counts",
        "grouping",
        "redistribution",
        "transfer_matrix",
    )

    def _energy_bounds(self, suffix: str) -> np.ndarray:
        """Read concrete keV bounds for response preparation and plotting."""
        return np.stack(
            (
                np.asarray(self.coords[f"e_min_{suffix}"], dtype=np.float64),
                np.asarray(self.coords[f"e_max_{suffix}"], dtype=np.float64),
            )
        )

    @property
    def in_energies(self):
        """Lower and upper incident-energy bounds in keV, shape (2, n_unfolded_bins)."""
        return self._energy_bounds("unfolded")

    @property
    def out_energies(self):
        """Lower and upper grouped detector bounds in keV, shape (2, n_folded_bins)."""

        return self._energy_bounds("folded")

    @classmethod
    def from_pha_file(
        cls,
        pha_path,
        rmf_path: str | None = None,
        arf_path: str | None = None,
        bkg_path: str | None = None,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
    ):
        r"""
        Build the observation configuration from a PHA file.

        Parameters:
            pha_path (str | os.PathLike): The path to the PHA file.
            rmf_path: The path to the RMF or combined RSP file. When omitted,
                use the PHA's RESPFILE link. Supply this argument when that link
                is absent or names an unavailable response.
            arf_path: The path to the ARF file.
            bkg_path: The path to the background file.
            low_energy: Inclusive lower bound in keV. Only groups wholly inside
                the selected band, after rejecting bad-quality channels, are used.
            high_energy: Inclusive upper bound in keV.

        Raises:
            ValueError: No response is specified by RESPFILE or ``rmf_path``.
            FileNotFoundError: A required named calibration file cannot be found.
        """

        from .util import data_path_finder

        arf_path_default, rmf_path_default, bkg_path_default = data_path_finder(
            pha_path,
            require_arf=arf_path is None,
            require_rmf=rmf_path is None,
            require_bkg=bkg_path is None,
        )

        arf_path = arf_path_default if arf_path is None else arf_path
        rmf_path = rmf_path_default if rmf_path is None else rmf_path
        bkg_path = bkg_path_default if bkg_path is None else bkg_path

        if rmf_path is None or (isinstance(rmf_path, str) and not rmf_path.strip()):
            raise ValueError(
                f"No response file was supplied for PHA {str(pha_path)!r}. "
                "Set its RESPFILE header to an RMF or combined RSP file, "
                "or pass rmf_path='path/to/response.rmf'."
            )

        instrument = Instrument.from_ogip_file(
            rmf_path, arf_path=arf_path if arf_path != "" else None
        )
        observation = Observation.from_pha_file(pha_path, bkg_path=bkg_path)

        return cls.from_instrument(
            instrument, observation, low_energy=low_energy, high_energy=high_energy
        )

    @classmethod
    def from_instrument(
        cls,
        instrument: Instrument,
        observation: Observation,
        low_energy: float = 1e-20,
        high_energy: float = 1e20,
    ):
        r"""
        Build the observation configuration from an [`Instrument`][jaxspec.data.instrument.Instrument] and an [`Observation`][jaxspec.data.observation.Observation] object.

        Parameters:
            instrument: The instrument object.
            observation: The observation object.
            low_energy: Inclusive lower bound in keV. Only groups wholly inside
                the selected band, after rejecting bad-quality channels, are used.
            high_energy: Inclusive upper bound in keV.

        """
        if not np.isfinite(low_energy) or low_energy < 0 or not high_energy > low_energy:
            raise ValueError("Energy bounds must satisfy 0 <= low_energy < high_energy.")
        if "channel" in instrument.coords:
            detector_channels = instrument.channel.data
            indices = np.searchsorted(detector_channels, observation.channel.data)
            if np.any(indices >= len(detector_channels)) or not np.array_equal(
                detector_channels[np.minimum(indices, len(detector_channels) - 1)],
                observation.channel.data,
            ):
                raise ValueError(
                    "PHA channel identifiers are absent from the response EBOUNDS. "
                    "Supply the matching RMF instead of aligning channels by position."
                )
            instrument = instrument.isel(instrument_channel=indices)
        # Apply exactly the same raw-channel mask to the response and both
        # observed spectra, including partially rejected groups.
        quality_filter = observation.quality.data == 0
        grouping = (
            scipy.sparse.csr_array(observation.grouping.data.to_scipy_sparse())
            .multiply(quality_filter)
            .tocsr()
        )
        grouping.eliminate_zeros()
        group_size = np.asarray(grouping.sum(axis=1)).ravel()
        if grouping.shape[1] != instrument.sizes["instrument_channel"]:
            raise ValueError(
                "PHA and response detector-channel counts differ; supply the matching RMF."
            )
        grouped_counts = np.asarray(grouping @ observation.counts.data).ravel()
        grouped_background = np.asarray(grouping @ observation.background.data).ravel()
        # Observation.from_matrix splits groups at every change in backratio.
        # This mean therefore recovers the common ratio within each nonempty
        # group; it does not approximate unequal source/background scales.
        grouped_backratio = np.divide(
            np.asarray(grouping @ observation.backratio.data).ravel(),
            group_size,
            out=np.zeros_like(group_size, dtype=float),
            where=group_size > 0,
        )
        e_min_channel = instrument.coords["e_min_channel"].data
        e_max_channel = instrument.coords["e_max_channel"].data
        e_min_unfolded = instrument.coords["e_min_unfolded"].data
        e_max_unfolded = instrument.coords["e_max_unfolded"].data
        redistribution = scipy.sparse.csr_array(instrument.redistribution.data.to_scipy_sparse())
        area = instrument.area.data
        exposure = observation.exposure.data
        areascal = np.where(quality_filter, observation.areascal.data, 0.0)

        # Empty groups retain sentinel bounds and are removed below. Explicit
        # sparse coordinates avoid treating rejected channels as zero energy.
        rows, columns = grouping.nonzero()
        e_min = np.full(grouping.shape[0], np.inf)
        e_max = np.full(grouping.shape[0], -np.inf)
        np.minimum.at(e_min, rows, e_min_channel[columns])
        np.maximum.at(e_max, rows, e_max_channel[columns])

        # Compute the transfer matrix
        transfer_matrix = grouping @ (redistribution.multiply(areascal[:, None]) * area * exposure)

        # These are boolean masks: rows select grouped detector channels in the
        # fitted band; columns retain positive-energy bins with a response.
        # Do not clip incident energies to the detector band: redistribution can
        # bring photons from outside that band into the selected channels.
        row_idx = (e_min >= low_energy) & (e_max <= high_energy) & (group_size > 0)
        col_idx = (e_min_unfolded > 0) & (redistribution.sum(axis=0) > 0)
        if not np.any(row_idx):
            raise ValueError(
                "No usable grouped channels lie wholly inside the selected energy band."
            )
        if not np.any(col_idx):
            raise ValueError(
                "The response has no energy bin contributing to any channel: every "
                "column of the redistribution matrix is empty or starts at zero energy. "
                "Check the RMF."
            )

        # Apply the same detector-row and photon-column masks to the transfer
        # matrix and its factors, keeping raw detector columns in the grouping.
        transfer_matrix = sparse.COO.from_scipy_sparse(transfer_matrix[row_idx][:, col_idx])
        redistribution_trimmed = sparse.COO.from_scipy_sparse(
            scipy.sparse.csr_array(redistribution)[:, col_idx]
        )
        grouping_trimmed = sparse.COO.from_scipy_sparse(
            scipy.sparse.csr_array(grouping)[row_idx, :]
        )
        folded_counts = grouped_counts[row_idx]
        folded_backratio = grouped_backratio[row_idx]
        area = instrument.area.data[col_idx]
        e_min_folded = e_min[row_idx]
        e_max_folded = e_max[row_idx]
        e_min_unfolded = e_min_unfolded[col_idx]
        e_max_unfolded = e_max_unfolded[col_idx]

        folded_background = grouped_background[row_idx]

        data_dict = {
            "transfer_matrix": (
                ["folded_channel", "unfolded_channel"],
                transfer_matrix,
                {
                    "description": "Transfer matrix to use to fold the incoming spectrum. It is built and restricted using the grouping, redistribution matrix, effective area, quality flags and energy bands defined by the user."
                },
            ),
            "redistribution": (
                ["instrument_channel", "unfolded_channel"],
                redistribution_trimmed,
                {
                    "description": "Redistribution matrix (RMF), trimmed to the same unfolded energy range as the transfer matrix. The transfer is grouping @ (areascal[:, None] * redistribution * area * exposure)."
                },
            ),
            "areascal": (
                ["instrument_channel"],
                areascal,
                {"description": "OGIP detector-channel response area scaling", "units": "1"},
            ),
            "grouping": (
                ["folded_channel", "instrument_channel"],
                grouping_trimmed,
                {
                    "description": "Grouping matrix, trimmed to the same folded channel range as the transfer matrix. Aggregates raw instrument channels into folded channels."
                },
            ),
            "area": (
                ["unfolded_channel"],
                area,
                {
                    "description": "Effective area with the same restrictions as the transfer matrix.",
                    "units": "cm^2",
                },
            ),
            "exposure": ([], exposure, {"description": "Total exposure", "unit": "s"}),
            "folded_counts": (
                ["folded_channel"],
                folded_counts,
                {
                    "description": "Folded counts after grouping, with the same restrictions as the transfer matrix.",
                    "unit": "photons",
                },
            ),
            "folded_backratio": (
                ["folded_channel"],
                folded_backratio,
                {
                    "description": "Background scaling after grouping, with the same restrictions as the transfer matrix."
                },
            ),
            "folded_background": (
                ["folded_channel"],
                folded_background,
                {
                    "description": "Folded background counts after grouping, with the same restrictions as the transfer matrix.",
                    "unit": "photons",
                },
            ),
        }

        return cls(
            data_dict,
            coords={
                "channel": (
                    ["instrument_channel"],
                    observation.channel.data,
                    {"description": "Original detector channel identifier"},
                ),
                "e_min_channel": (
                    ["instrument_channel"],
                    e_min_channel,
                    {
                        "description": "Raw detector lower energy aligned to grouping columns",
                        "units": "keV",
                    },
                ),
                "e_max_channel": (
                    ["instrument_channel"],
                    e_max_channel,
                    {
                        "description": "Raw detector upper energy aligned to grouping columns",
                        "units": "keV",
                    },
                ),
                "e_min_folded": (
                    ["folded_channel"],
                    e_min_folded,
                    {"description": "Low energy of folded channel"},
                ),
                "e_max_folded": (
                    ["folded_channel"],
                    e_max_folded,
                    {"description": "High energy of folded channel"},
                ),
                "e_min_unfolded": (
                    ["unfolded_channel"],
                    e_min_unfolded,
                    {"description": "Low energy of unfolded channel"},
                ),
                "e_max_unfolded": (
                    ["unfolded_channel"],
                    e_max_unfolded,
                    {"description": "High energy of unfolded channel"},
                ),
            },
            attrs=observation.attrs | instrument.attrs,
        )

    @classmethod
    def mock_from_instrument(
        cls,
        instrument: Instrument,
        exposure: float,
        low_energy: float = 1e-300,
        high_energy: float = 1e300,
    ):
        """
        Create a mock observation configuration from an instrument object. The fake observation will have zero counts.

        Parameters:
            instrument: The instrument object.
            exposure: Exposure in seconds; Astropy time quantities are converted.
            low_energy: Inclusive lower detector-energy bound in keV.
            high_energy: Inclusive upper detector-energy bound in keV.
        """

        n_channels = instrument.sizes["instrument_channel"]
        channels = (
            instrument.channel.data if "channel" in instrument.coords else np.arange(n_channels)
        )

        observation = Observation.from_matrix(
            np.zeros(n_channels),
            sparse.eye(n_channels),
            channels,
            np.zeros(n_channels, dtype=bool),
            exposure,
            backratio=np.ones(n_channels),
            attributes={"description": "Mock observation"} | instrument.attrs,
        )

        return cls.from_instrument(
            instrument, observation, low_energy=low_energy, high_energy=high_energy
        )

    def plot_counts(self, **kwargs):
        return self.folded_counts.plot.step(x="e_min_folded", where="post", **kwargs)
