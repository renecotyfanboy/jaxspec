# precommit is suppressing these imports
from .obsconf import ObsConfiguration  # noqa: F401
from .instrument import Instrument  # noqa: F401
from .observation import Observation  # noqa: F401
import astropy.units as u

u.add_enabled_aliases({"counts": u.count})
u.add_enabled_aliases({"channel": u.dimensionless_unscaled})
