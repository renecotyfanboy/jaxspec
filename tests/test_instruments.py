import matplotlib.pyplot as plt
import pytest
import yaml
from pathlib import Path
from jaxspec.data import Instrument, Observation, ObsConfiguration
from typing import List


# Dir containing 8 files
data_directory = Path(__file__).parent.resolve() / "data"

with open(data_directory / "observations.yml", "r") as file:
    data_collection = yaml.safe_load(file)


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

    if observation["name"] in not_working:
        with pytest.raises(Exception):
            ObsConfiguration.from_pha_file(data_directory / observation["pha_path"])

    else:
        ObsConfiguration.from_pha_file(data_directory / observation["pha_path"])


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_plot_curated_data_files_grouping(observation):
    # Not working because the file is a pha2
    not_working = ["Chandra/LETGS"]

    if observation["name"] in not_working:
        with pytest.raises(Exception):
            Observation.from_pha_file(data_directory / observation["pha_path"])

    else:
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

    if observation["name"] in not_working:
        with pytest.raises(Exception):
            ObsConfiguration.from_pha_file(
                data_directory / observation["pha_path"],
                rmf_path=data_directory / observation["rmf_path"],
                arf_path=data_directory / observation.get("arf_path", None)
                if observation.get("arf_path", None) is not None
                else None,
            )

    else:
        ObsConfiguration.from_pha_file(
            data_directory / observation["pha_path"],
            rmf_path=data_directory / observation["rmf_path"],
            arf_path=data_directory / observation.get("arf_path", None)
            if observation.get("arf_path", None) is not None
            else None,
        )


@pytest.mark.parametrize("observation", data_collection, ids=lambda m: m["name"])
def test_plot_instruments_from_curated_data_files(observation):
    if observation["name"] not in ["XRISM/Resolve", "Chandra/LETGS"]:
        if "arf_path" in observation.keys():
            instrument = Instrument.from_ogip_file(
                data_directory / observation["rmf_path"], arf_path=data_directory / observation["arf_path"]
            )

        else:
            instrument = Instrument.from_ogip_file(data_directory / observation["rmf_path"], arf_path=None)

        instrument.plot_area()
        plt.suptitle(observation["name"])
        plt.show()

        instrument.plot_redistribution()
        plt.suptitle(observation["name"])
        plt.show()


def test_plot_instruments(instruments: List[Instrument]):
    for instrument in instruments:
        instrument


def test_plot_observations(observations: List[Observation]):
    for observation in observations:
        observation
