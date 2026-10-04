"""Example registry entries keep observations and responses paired across loaders."""

from pathlib import Path

import numpy as np
import pytest

from astropy.io import fits

from jaxspec.data import Instrument, ObsConfiguration, Observation, util


@pytest.mark.parametrize(
    ("source", "detectors"),
    [("NGC7793_ULX4_PN", ("PN",)), ("NGC7793_ULX4_ALL", ("PN", "MOS1", "MOS2"))],
)
def test_existing_examples_preserve_return_types_and_file_pairing(source, detectors):
    """The single PN and joint EPIC examples retain their public names and order."""
    observations = util.load_example_pha(source)
    instruments = util.load_example_instruments(source)
    configurations = util.load_example_obsconf(source)
    if len(detectors) == 1:
        assert isinstance(observations, Observation)
        assert isinstance(instruments, Instrument)
        assert isinstance(configurations, ObsConfiguration)
        observations, instruments, configurations = (
            {"PN": value} for value in (observations, instruments, configurations)
        )
    assert tuple(observations) == tuple(instruments) == tuple(configurations) == detectors
    for detector in detectors:
        observation = observations[detector]
        assert Path(observation.attrs["observation_file"]).name.startswith(f"{detector}_")
        assert (
            Path(observation.attrs["background_file"]).name == f"{detector}background_spectrum.fits"
        )
        background = fits.getdata(observation.attrs["background_file"], "SPECTRUM")
        np.testing.assert_array_equal(observation.background, background["COUNTS"])
        area_path = Path(observation.attrs["observation_file"]).with_name(f"{detector}.arf")
        np.testing.assert_allclose(instruments[detector].area, fits.getdata(area_path)["SPECRESP"])
        expected = ObsConfiguration.from_instrument(
            instruments[detector], observation, low_energy=0.5, high_energy=8.0
        )
        np.testing.assert_array_equal(
            configurations[detector].folded_counts, expected.folded_counts
        )
        np.testing.assert_allclose(
            configurations[detector].transfer_matrix.data.todense(),
            expected.transfer_matrix.data.todense(),
        )


def write_example_files(root, index):
    """Write a small independent observation with deliberately unrelated filenames."""
    directory = root / f"other_target/epoch{index}"
    directory.mkdir(parents=True)

    def table(filename, name, columns, **headers):
        hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name=key, format=fmt, array=values, unit=unit)
                for key, fmt, values, unit in columns
            ],
            name=name,
        )
        hdu.header.update(headers)
        hdu.writeto(directory / filename)
        return hdu

    channels = np.arange(3)
    for filename, counts in (("target.pha", [3, 5, 7]), ("sky.pha", [1, 2, 3])):
        table(
            filename,
            "SPECTRUM",
            [("CHANNEL", "J", channels, None), ("COUNTS", "J", np.array(counts) * index, None)],
            EXPOSURE=10.0,
            GROUPING=0,
            QUALITY=0,
            BACKSCAL=1.0,
            AREASCAL=1.0,
        )
    energies = [
        ("ENERG_LO", "E", [0.1, 1.0, 9.0], "keV"),
        ("ENERG_HI", "E", [0.5, 2.0, 10.0], "keV"),
    ]
    matrix = table(
        "redistribution.fits",
        "MATRIX",
        [
            *energies,
            ("N_GRP", "J", [1, 1, 1], None),
            ("F_CHAN", "J", [0, 0, 0], None),
            ("N_CHAN", "J", [3, 3, 3], None),
            ("MATRIX", "3E", np.eye(3), None),
        ],
        TLMIN4=0,
    )
    bounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=channels),
            fits.Column(name="E_MIN", format="E", unit="keV", array=[0.1, 1.0, 9.0]),
            fits.Column(name="E_MAX", format="E", unit="keV", array=[0.5, 2.0, 10.0]),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), matrix, bounds]).writeto(
        directory / "redistribution.fits", overwrite=True
    )
    table(
        "collecting_area.fits",
        "SPECRESP",
        [*energies, ("SPECRESP", "E", np.array([10, 20, 30]) * index, "cm2")],
    )


@pytest.mark.parametrize("number_of_observations", [1, 2])
def test_registering_another_dataset_needs_no_loader_changes(
    tmp_path, monkeypatch, number_of_observations
):
    """Different source names, labels and filenames work through the same loaders."""
    records = []
    for index in range(1, number_of_observations + 1):
        write_example_files(tmp_path / "example_data", index)
        prefix = f"other_target/epoch{index}"
        records.append(
            util._ExampleObservation(
                f"visit-{3 - index}",
                f"{prefix}/target.pha",
                f"{prefix}/sky.pha",
                f"{prefix}/redistribution.fits",
                f"{prefix}/collecting_area.fits",
            )
        )
    monkeypatch.setitem(util._EXAMPLE_OBSERVATIONS, "OTHER_TARGET", tuple(records))
    fetched = []

    def fetch(path):
        fetched.append(path)
        assert (tmp_path / path).is_file()
        return str(tmp_path / path)

    monkeypatch.setattr(util.table_manager, "fetch", fetch)
    observations = util.load_example_pha("OTHER_TARGET")
    instruments = util.load_example_instruments("OTHER_TARGET")
    configurations = util.load_example_obsconf("OTHER_TARGET")
    if number_of_observations == 1:
        assert isinstance(observations, Observation)
        assert isinstance(instruments, Instrument)
        assert isinstance(configurations, ObsConfiguration)
        observations, instruments, configurations = (
            {records[0].label: value} for value in (observations, instruments, configurations)
        )
    expected_labels = tuple(record.label for record in records)
    assert tuple(observations) == tuple(instruments) == tuple(configurations) == expected_labels
    for index, label in enumerate(expected_labels, start=1):
        np.testing.assert_array_equal(observations[label].counts, np.array([3, 5, 7]) * index)
        np.testing.assert_array_equal(observations[label].background, np.array([1, 2, 3]) * index)
        np.testing.assert_array_equal(instruments[label].area, np.array([10, 20, 30]) * index)
        np.testing.assert_array_equal(configurations[label].folded_counts, [5 * index])
    assert set(fetched) == {
        f"example_data/other_target/epoch{index}/{name}"
        for index in range(1, number_of_observations + 1)
        for name in ("target.pha", "sky.pha", "redistribution.fits", "collecting_area.fits")
    }


def test_example_without_explicit_background_or_arf(tmp_path, monkeypatch):
    """A combined response and local BACKFILE work without optional fetches."""
    write_example_files(tmp_path / "example_data", 1)
    prefix = "other_target/epoch1"
    directory = tmp_path / "example_data" / prefix
    fits.setval(directory / "target.pha", "BACKFILE", extname="SPECTRUM", value="sky.pha")
    with fits.open(directory / "redistribution.fits", mode="update") as hdus:
        hdus["MATRIX"].data["MATRIX"] *= np.array([10, 20, 30])[:, None]
        hdus["MATRIX"].header["HDUCLAS3"] = "FULL"
    entry = util._ExampleObservation(
        "combined", f"{prefix}/target.pha", None, f"{prefix}/redistribution.fits", None
    )
    monkeypatch.setitem(util._EXAMPLE_OBSERVATIONS, "COMBINED", (entry,))

    def fetch(path):
        assert path in {f"example_data/{entry.pha}", f"example_data/{entry.rmf}"}
        return str(tmp_path / path)

    monkeypatch.setattr(util.table_manager, "fetch", fetch)
    configuration = util.load_example_obsconf("COMBINED")
    np.testing.assert_array_equal(configuration.folded_counts, [5])
    np.testing.assert_array_equal(configuration.folded_background, [2])
    np.testing.assert_allclose(configuration.transfer_matrix.data.todense(), [[0, 200, 0]])


@pytest.mark.parametrize(
    "loader", [util.load_example_pha, util.load_example_instruments, util.load_example_obsconf]
)
def test_unknown_example_fails_before_fetching(loader, monkeypatch):
    """Misspelled registry keys produce a useful error without downloading files."""
    monkeypatch.setattr(
        util.table_manager, "fetch", lambda path: pytest.fail(f"Unexpected fetch: {path}")
    )
    with pytest.raises(ValueError, match="UNKNOWN_TARGET"):
        loader("UNKNOWN_TARGET")
