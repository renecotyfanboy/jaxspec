import importlib.resources
from astropy.io import ascii


abundance_table = ascii.read(importlib.resources.files("jaxspec") / "tables/abundances.dat")
"""
Table containing various abundances that can be used in `jaxspec`. It is adapted from
[XSPEC's abundance table](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node116.html). The following table are available:

| Name   | Reference                                                                                      | Note                         |
| :----: | :--------------------------------------------------------------------------------------------- | :--------------------------- |
| `angr` | [Anders & Gevresse (1989)](https://ui.adsabs.harvard.edu/abs/1989GeCoA..53..197A/abstract)     | Photospheric, using Table 2. |
| `aspl` | [Asplund et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract)      | Photospheric, using Table 1. |
| `feld` | [Feldman (1992)](https://ui.adsabs.harvard.edu/abs/1992PhyS...46..202F/abstract)               |
| `aneb` | [Anders & Ebihara (1982)](https://ui.adsabs.harvard.edu/abs/1982GeCoA..46.2363A/abstract)      |
| `grsa` | [Grevesse & Sauval (1998)](https://ui.adsabs.harvard.edu/abs/1998SSRv...85..161G/abstract)     |
| `wilm` | [Wilms et al. (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...542..914W/abstract)          |
| `lodd` | [Lodders (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...591.1220L/abstract)               | Photospheric, using Table 1. |
| `lgpp` | [Lodders, Palme & Gail (2009)](https://ui.adsabs.harvard.edu/abs/2009LanB...4B..712L/abstract) | Photospheric, using Table 4. |
| `lgps` | [Lodders, Palme & Gail (2009)](https://ui.adsabs.harvard.edu/abs/2009LanB...4B..712L/abstract) | Proto-solar, using Table 10. |


The table is a `astropy.table.Table` object, and can be accessed as such. For example, to get the abundance of iron in the `aspl` table, one can do:

```python
from jaxspec.util.abundance import abundance_table
assert abundance_table['Element'][ 26 - 1] == 'Fe'
print(abundance_table['aspl'][ 26 - 1]) # 3.16e-05
```

The full table is displayed below:

| Element | angr | aspl | feld | aneb | grsa | wilm | lodd | lgpp | lgps |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| H | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| He | 0.0977 | 0.0851 | 0.0977 | 0.0801 | 0.0851 | 0.0977 | 0.0792 | 0.0841 | 0.0969 |
| Li | 1.45e-11 | 1.12e-11 | 1.26e-11 | 2.19e-09 | 1.26e-11 | 0.0 | 1.9e-09 | 1.26e-11 | 2.15e-09 |
| Be | 1.41e-11 | 2.4e-11 | 2.51e-11 | 2.87e-11 | 2.51e-11 | 0.0 | 2.57e-11 | 2.4e-11 | 2.36e-11 |
| B | 3.98e-10 | 5.01e-10 | 3.55e-10 | 8.82e-10 | 3.55e-10 | 0.0 | 6.03e-10 | 5.01e-10 | 7.26e-10 |
| C | 0.000363 | 0.000269 | 0.000398 | 0.000445 | 0.000331 | 0.00024 | 0.000245 | 0.000245 | 0.000278 |
| N | 0.000112 | 6.76e-05 | 0.0001 | 9.12e-05 | 8.32e-05 | 7.59e-05 | 6.76e-05 | 7.24e-05 | 8.19e-05 |
| O | 0.000851 | 0.00049 | 0.000851 | 0.000739 | 0.000676 | 0.00049 | 0.00049 | 0.000537 | 0.000606 |
| F | 3.63e-08 | 3.63e-08 | 3.63e-08 | 3.1e-08 | 3.63e-08 | 0.0 | 2.88e-08 | 3.63e-08 | 3.1e-08 |
| Ne | 0.000123 | 8.51e-05 | 0.000129 | 0.000138 | 0.00012 | 8.71e-05 | 7.41e-05 | 0.000112 | 0.000127 |
| Na | 2.14e-06 | 1.74e-06 | 2.14e-06 | 2.1e-06 | 2.14e-06 | 1.45e-06 | 1.99e-06 | 2e-06 | 2.23e-06 |
| Mg | 3.8e-05 | 3.98e-05 | 3.8e-05 | 3.95e-05 | 3.8e-05 | 2.51e-05 | 3.55e-05 | 3.47e-05 | 3.98e-05 |
| Al | 2.95e-06 | 2.82e-06 | 2.95e-06 | 3.12e-06 | 2.95e-06 | 2.14e-06 | 2.88e-06 | 2.95e-06 | 3.27e-06 |
| Si | 3.55e-05 | 3.24e-05 | 3.55e-05 | 3.68e-05 | 3.55e-05 | 1.86e-05 | 3.47e-05 | 3.31e-05 | 3.86e-05 |
| P | 2.82e-07 | 2.57e-07 | 2.82e-07 | 3.82e-07 | 2.82e-07 | 2.63e-07 | 2.88e-07 | 2.88e-07 | 3.2e-07 |
| S | 1.62e-05 | 1.32e-05 | 1.62e-05 | 1.89e-05 | 2.14e-05 | 1.23e-05 | 1.55e-05 | 1.38e-05 | 1.63e-05 |
| Cl | 3.16e-07 | 3.16e-07 | 3.16e-07 | 1.93e-07 | 3.16e-07 | 1.32e-07 | 1.82e-07 | 3.16e-07 | 2e-07 |
| Ar | 3.63e-06 | 2.51e-06 | 4.47e-06 | 3.82e-06 | 2.51e-06 | 2.57e-06 | 3.55e-06 | 3.16e-06 | 3.58e-06 |
| K | 1.32e-07 | 1.07e-07 | 1.32e-07 | 1.39e-07 | 1.32e-07 | 0.0 | 1.29e-07 | 1.32e-07 | 1.45e-07 |
| Ca | 2.29e-06 | 2.19e-06 | 2.29e-06 | 2.25e-06 | 2.29e-06 | 1.58e-06 | 2.19e-06 | 2.14e-06 | 2.33e-06 |
| Sc | 1.26e-09 | 1.41e-09 | 1.48e-09 | 1.24e-09 | 1.48e-09 | 0.0 | 1.17e-09 | 1.26e-09 | 1.33e-09 |
| Ti | 9.77e-08 | 8.91e-08 | 1.05e-07 | 8.82e-08 | 1.05e-07 | 6.46e-08 | 8.32e-08 | 7.94e-08 | 9.54e-08 |
| V | 1e-08 | 8.51e-09 | 1e-08 | 1.08e-08 | 1e-08 | 0.0 | 1e-08 | 1e-08 | 1.11e-08 |
| Cr | 4.68e-07 | 4.37e-07 | 4.68e-07 | 4.93e-07 | 4.68e-07 | 3.24e-07 | 4.47e-07 | 4.37e-07 | 5.06e-07 |
| Mn | 2.45e-07 | 2.69e-07 | 2.45e-07 | 3.5e-07 | 2.45e-07 | 2.19e-07 | 3.16e-07 | 2.34e-07 | 3.56e-07 |
| Fe | 4.68e-05 | 3.16e-05 | 3.24e-05 | 3.31e-05 | 3.16e-05 | 2.69e-05 | 2.95e-05 | 2.82e-05 | 3.27e-05 |
| Co | 8.32e-08 | 9.77e-08 | 8.32e-08 | 8.27e-08 | 8.32e-08 | 8.32e-08 | 8.13e-08 | 8.32e-08 | 9.07e-08 |
| Ni | 1.78e-06 | 1.66e-06 | 1.78e-06 | 1.81e-06 | 1.78e-06 | 1.12e-06 | 1.66e-06 | 1.7e-06 | 1.89e-06 |
| Cu | 1.62e-08 | 1.55e-08 | 1.62e-08 | 1.89e-08 | 1.62e-08 | 0.0 | 1.82e-08 | 1.62e-08 | 2.09e-08 |
| Zn | 3.98e-08 | 3.63e-08 | 3.98e-08 | 4.63e-08 | 3.98e-08 | 0.0 | 4.27e-08 | 4.17e-08 | 5.02e-08 |


"""
