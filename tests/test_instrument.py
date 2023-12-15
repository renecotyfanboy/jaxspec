import os
import sys
from unittest import TestCase
from jaxspec.data.instrument import Instrument
from jaxspec.data.observation import Observation

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


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
        for file in ["data/ogip/MOS1_spectrum_grp.fits", "data/ogip/MOS2_spectrum_grp.fits", "data/ogip/PN_spectrum_grp20.fits"]
    ]

    def test_instrument_constructor(self):
        """
        Test constructing an Instrument using the ogip files
        """

        for arf_file, rmf_file in zip(self.arf_files, self.rmf_files):
            instrument = Instrument.from_ogip_file(arf_file, rmf_file, 1000)
            assert instrument is not None

    def test_observation_constructor(self):
        """
        Test constructing an Instrument using the ogip files
        """

        for pha_file in self.pha_files:
            observation = Observation.from_pha_file(pha_file)
            assert observation is not None
