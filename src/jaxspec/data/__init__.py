import astropy.units as u

from .instrument import Instrument
from .obsconf import ObsConfiguration
from .observation import Observation

u.add_enabled_aliases({"counts": u.count})
u.add_enabled_aliases({"channel": u.dimensionless_unscaled})
# Arbitrary units are found in .rsp files , let's hope it is compatible with what we would expect as the rmf x arf
# u.add_enabled_aliases({"au": u.dimensionless_unscaled})
