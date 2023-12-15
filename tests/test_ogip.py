import os
import sys
from unittest import TestCase
from jaxspec.data.ogip import DataARF, DataRMF, DataPHA
from jaxspec.data.util import load_example_observations

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestExampleData(TestCase):
    def test_load_example_data(self):
        """
        Test loading example data
        """

        load_example_observations()


class TestRSP(TestCase):
    arf_files = [
        os.path.join(current_dir, file)
        for file in [
            "data/ogip/PN.arf",
            "data/ogip/MOS1.arf",
            "data/ogip/MOS2.arf",
            "data/ogip/nustar.arf",
        ]
    ]
    rmf_files = [
        os.path.join(current_dir, file)
        for file in [
            "data/ogip/PN.rmf",
            "data/ogip/MOS1.rmf",
            "data/ogip/MOS2.rmf",
            "data/ogip/nustar.rmf",
            "data/ogip/XIFU.rmf",
        ]
    ]
    pha_files = [
        os.path.join(current_dir, file)
        for file in [
            "data/ogip/nustar_pha.pha",
            "data/ogip/MOS1_spectrum_grp.fits",
            "data/ogip/MOS2_spectrum_grp.fits",
            "data/ogip/PN_spectrum_grp20.fits",
        ]
    ]

    def test_arf(self):
        """
        Test reading an AMF file using ref files
        """

        for arf_file in self.arf_files:
            DataARF.from_file(arf_file)

    def test_rmf(self):
        """
        Test reading an RMF file using ref files
        """

        for rmf_file in self.rmf_files:
            DataRMF.from_file(rmf_file)

    def test_pha(self):
        """
        Test opening various PHA files from fits and pha
        """

        for pha_file in self.pha_files:
            DataPHA.from_file(pha_file)
