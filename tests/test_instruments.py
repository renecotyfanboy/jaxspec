import shutil

import matplotlib.pyplot as plt
import pytest

from astropy.io import fits
from conftest import data_collection, data_directory
from jaxspec.data import Instrument, ObsConfiguration, Observation


def test_loading_non_existent_files():
    path = "this/path/does/not/exist"

    with pytest.raises(FileNotFoundError):
        Instrument.from_ogip_file(path)

    with pytest.raises(FileNotFoundError):
        Observation.from_pha_file(path)

    with pytest.raises(FileNotFoundError):
        ObsConfiguration.from_pha_file(path)


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_loading_curated_data_files_from_pha(observation):
    # Not working either because the header are wrong or the file is a pha2
    not_working = ["XMM-Newton/RGS", "XRISM/Resolve", "Chandra/LETGS"]

    if observation["name"] not in not_working:
        ObsConfiguration.from_pha_file(data_directory / observation["pha_path"])


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_plot_curated_data_files_grouping(observation):
    # Not working because the file is a pha2
    not_working = ["Chandra/LETGS", "IXPE/GPD"]

    if observation["name"] not in not_working:
        obs = Observation.from_pha_file(data_directory / observation["pha_path"])

        if observation["name"] in ["XRISM/Resolve", "Hitomi/SXS"]:
            obs.plot_counts()
            plt.suptitle(observation["name"])
            plt.show()

        else:
            obs.plot_grouping()
            plt.suptitle(observation["name"])
            plt.show()


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_loading_curated_data_files_from_pha_with_explicit_files(observation):
    # Not working because the file is a pha2
    not_working = ["Chandra/LETGS"]

    if observation["name"] not in not_working:
        ObsConfiguration.from_pha_file(
            data_directory / observation["pha_path"],
            rmf_path=data_directory / observation["rmf_path"],
            arf_path=data_directory / observation.get("arf_path", None)
            if observation.get("arf_path", None) is not None
            else None,
        )


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_plot_instruments_from_curated_data_files(observation):
    title = observation["name"]

    if observation["name"] not in ["Chandra/LETGS"]:
        if "arf_path" in observation.keys():
            instrument = Instrument.from_ogip_file(
                data_directory / observation["rmf_path"],
                arf_path=data_directory / observation["arf_path"],
            )

            title += " (ARF included)"

        else:
            instrument = Instrument.from_ogip_file(
                data_directory / observation["rmf_path"], arf_path=None
            )

            title += " (no ARF included)"

        instrument.plot_area()
        plt.suptitle(title)
        plt.show()

        if observation["name"] not in ["XRISM/Resolve", "Hitomi/SXS"]:
            instrument.plot_redistribution()
            plt.suptitle(observation["name"])
            plt.show()


def test_plot_exemple_instruments(instruments: list[Instrument]):
    for instrument in instruments:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        instrument.plot_redistribution()
        plt.subplot(122)
        instrument.plot_area()
        plt.show()


def test_plot_exemple_observations(observations: list[Observation]):
    for observation in observations:
        observation.plot_grouping()
        plt.show()


@pytest.fixture
def file_with_no_grouping(tmp_path):
    file_path = tmp_path / "no_grouping.fits"
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN_spectrum_grp20.fits", file_path)
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.arf", tmp_path / "PN.arf")
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.rmf", tmp_path / "PN.rmf")
    shutil.copyfile(
        data_directory / "XMM-Newton/EPIC-PN/PNbackground_spectrum.fits",
        tmp_path / "PNbackground_spectrum.fits",
    )

    with fits.open(file_path) as hdul:
        hdul[1].columns.del_col("GROUPING")
        hdul.writeto(file_path, overwrite=True)

    return file_path


def test_loading_file_with_no_grouping(file_with_no_grouping):
    with pytest.raises(ValueError):
        Observation.from_pha_file(file_with_no_grouping)


@pytest.fixture
def file_with_no_backscal(tmp_path):
    file_path = tmp_path / "no_backscal.fits"
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN_spectrum_grp20.fits", file_path)
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.arf", tmp_path / "PN.arf")
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.rmf", tmp_path / "PN.rmf")
    shutil.copyfile(
        data_directory / "XMM-Newton/EPIC-PN/PNbackground_spectrum.fits",
        tmp_path / "PNbackground_spectrum.fits",
    )

    with fits.open(file_path) as hdul:
        del hdul[1].header["BACKSCAL"]
        hdul.writeto(file_path, overwrite=True)

    return file_path


def test_loading_file_with_no_backscal(file_with_no_backscal):
    with pytest.raises(ValueError):
        Observation.from_pha_file(file_with_no_backscal)


@pytest.fixture
def file_with_no_areascal(tmp_path):
    file_path = tmp_path / "no_areascal.fits"
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN_spectrum_grp20.fits", file_path)
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.arf", tmp_path / "PN.arf")
    shutil.copyfile(data_directory / "XMM-Newton/EPIC-PN/PN.rmf", tmp_path / "PN.rmf")
    shutil.copyfile(
        data_directory / "XMM-Newton/EPIC-PN/PNbackground_spectrum.fits",
        tmp_path / "PNbackground_spectrum.fits",
    )

    with fits.open(file_path) as hdul:
        del hdul[1].header["AREASCAL"]
        hdul.writeto(file_path, overwrite=True)

    return file_path


def test_loading_file_with_no_areascal(file_with_no_areascal):
    with pytest.raises(ValueError):
        Observation.from_pha_file(file_with_no_areascal)
