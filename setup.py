from setuptools import setup
from jaxspec import __version__

setup(
    name="jaxspec",
    version=__version__,
    description="Fitting X-Ray spectra with jax and numpyro",
    author="Simon Dupourqué",
    author_email="sdupourque@irap.omp.eu",
    packages=["jaxspec"],
    install_requires=[
        "chex",
        "jax",
        "simpleeval",
        "dm-haiku"
    ],
    include_package_data=True,
    package_data={'': ['tables/*.fits']},
)