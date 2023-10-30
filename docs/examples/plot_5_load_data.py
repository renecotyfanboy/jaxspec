"""
# Load your data in JAXspec

Most of the data you will use when manipulating X-ray spectra will be in the form of a PHA file. This file contains
the measured spectra, the background spectra, and also links to the response matrix file and the ancillary response
file. JAXspec provides a simple way to load this data using the `Observation.from_pha_file` function.
"""

from jaxspec.data.observation import Observation

#observation = Observation.from_pha_file("docs/examples/data/obs.pha")

# %% New cell
# If you only want to load the instrument related data (i.e. the response matrix and the ancillary response file),
# you can use the `Instrument.fr` function.

from jaxspec.data.instrument import Instrument

#instrument = Instrument.from_ogip_file("docs/examples/data/PN.arf", "docs/examples/data/PN.rmf")

#%% New cell
# This is basically the way you load data using JAXspec