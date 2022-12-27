from astropy.table import QTable


class DataARF:
    """Class to handle ARF data defined with OGIP standards [1]_ [2]_.

    References
    ----------

    .. [1] "The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats)", https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html

    .. [2] "The Calibration Requirements for Spectral Analysis Addendum: Changes log", https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html

    """

    def __init__(self, energ_lo, energ_hi, specresp):

        self.specresp = specresp
        self.energ_lo = energ_lo
        self.energ_hi = energ_hi

    @classmethod
    def from_file(cls, arf_file):

        arf_table = QTable.read(arf_file)

        return cls(arf_table['ENERG_LO'],
                   arf_table['ENERG_HI'],
                   arf_table['SPECRESP'])

    def plot(self):

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot((self.energ_lo + self.energ_hi)/2, self.specresp)
        plt.xlabel(f'Energy [{self.energ_lo.unit.to_string("latex")}]')
        plt.ylabel(f'Spectral Response [{self.specresp.unit.to_string("latex")}]')
        plt.semilogx()
        plt.show()
