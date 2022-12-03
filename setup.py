from setuptools import setup

__version__ = '0.0.1'

setup(name='jaxspec',
      version=__version__,
      description='A first build of jaxspec, a future library to fit X-ray spectra with JAX',
      url='https://github.com/renecotyfanboy/jaxspec',
      author='Simon Dupourqué',
      author_email='sdupourque@irap.omp.eu',
      license='MIT',
      packages=['jaxspec'],
      zip_safe=False)