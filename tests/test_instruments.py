import matplotlib.pyplot as plt
from jaxspec.data import Instrument, Observation
from typing import List


def test_plot_instruments(instruments: List[Instrument]):
    for instrument in instruments:
        instrument.plot_area()
        plt.show()

        instrument.plot_redistribution()
        plt.show()


def test_plot_observations(observations: List[Observation]):
    for observation in observations:
        observation.plot_counts()
        plt.show()

        observation.plot_grouping()
        plt.show()
