import importlib.resources
from .observation import Observation

example_observations = {
    'PN': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/PN.pha'),
    'MOS1': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/MOS1.pha'),
    'MOS2': Observation.from_pha_file(importlib.resources.files('jaxspec') / 'data/example_data/MOS2.pha')
}
