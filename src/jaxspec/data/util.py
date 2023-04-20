import importlib.resources
from .observation import Observation

example_observations = {
    'PN': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/PN.pha', low_energy=0.3, high_energy=12),
    'MOS1': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/MOS1.pha', low_energy=0.3, high_energy=12),
    'MOS2': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/MOS2.pha', low_energy=0.3, high_energy=12)
}
