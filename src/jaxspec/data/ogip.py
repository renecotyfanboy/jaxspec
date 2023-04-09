import numpy as np
import jax.numpy as jnp
from astropy.table import QTable
from astropy.io import fits
from jax.experimental import sparse


class DataPHA:
    r"""
    Class to handle PHA data defined with OGIP standards.

    References
    ----------

    * `THE OGIP STANDARD PHA FILE FORMAT <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html>`_

    """

    def __init__(self, channel, counts, exposure, grouping=None, quality=None, backfile=None, respfile=None, ancrfile=None):

        self.channel = channel
        self.counts = counts
        self.exposure = exposure

        self.quality = quality
        self.backfile = backfile
        self.respfile = respfile
        self.ancrfile = ancrfile
        
        if grouping is not None:
            # Indices array of beginning of each group
            b_grp = np.where(grouping == 1)[0]
            # Indices array of ending of each group
            e_grp = np.hstack((b_grp[1:], len(channel)))
            # Matrix to multiply with counts/channel to have counts/group
            grp_matrix = np.zeros((len(b_grp), len(channel)), dtype=bool)

            for i in range(len(b_grp)):

                grp_matrix[i, b_grp[i]:e_grp[i]] = 1

        else:

            grp_matrix = np.eye(len(channel))

        self.grouping = grp_matrix

    @classmethod
    def from_file(cls, pha_file):

        data = QTable.read(pha_file,'SPECTRUM')
        header = fits.getheader(pha_file,'SPECTRUM')

        kwargs = {}

        # Grouping and quality parameters are in binned PHA dataset
        kwargs['grouping'] = data['GROUPING'] if 'GROUPING' in data.colnames else None
        kwargs['quality'] = data['QUALITY'] if 'QUALITY' in data.colnames else None

        # Backfile, respfile and ancrfile are in primary header
        kwargs['backfile'] = header['BACKFILE'] if len(header['BACKFILE']) > 0 else None
        kwargs['respfile'] = header['RESPFILE'] if len(header['RESPFILE']) > 0 else None
        kwargs['ancrfile'] = header['ANCRFILE'] if len(header['ANCRFILE']) > 0 else None

        return cls(data['CHANNEL'], data['COUNTS'], header['EXPOSURE'], **kwargs)


    #def plot(self):
    #   import matplotlib.pyplot as plt
    #   plt.figure()
    #   plt.plot(self.channel, self.counts/self.exposure)
    #   plt.xlabel(f'Channel')
    #   plt.ylabel(f'Countrate [cts/s]')
    #   plt.show()


class DataARF:
    r"""
    Class to handle ARF data defined with OGIP standards.

    References
    ----------

    * `The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`_
    * `The Calibration Requirements for Spectral Analysis Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`_

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

    # def plot(self):
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure()
    #     plt.plot((self.energ_lo + self.energ_hi)/2, self.specresp)
    #     plt.xlabel(f'Energy [{self.energ_lo.unit.to_string("latex")}]')
    #     plt.ylabel(f'Spectral Response [{self.specresp.unit.to_string("latex")}]')
    #     plt.semilogx()
    #     plt.show()


class DataRMF:
    r"""
    Class to handle RMF data defined with OGIP standards.

    References
    ----------

    * `The Calibration Requirements for Spectral Analysis (Definition of RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`_
    * `The Calibration Requirements for Spectral Analysis Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`_

    """

    def __init__(self, energ_lo, energ_hi, n_grp, f_chan, n_chan, matrix, channel, e_min, e_max):

        # RMF stuff
        self.energ_lo = energ_lo # "Entry" energies
        self.energ_hi = energ_hi # "Entry" energies
        self.n_grp = n_grp # "Entry" energies
        self.f_chan = f_chan
        self.n_chan = n_chan
        self.matrix_entry = matrix

        # Detector channels
        self.channel = channel
        self.e_min = e_min
        self.e_max = e_max

        self.full_matrix = np.zeros(self.n_grp.shape + self.channel.shape)

        for i, n_grp in enumerate(self.n_grp):

            base = 0

            if np.size(self.f_chan[i]) == 1:

                low = int(self.f_chan[i])
                high = min(int(self.f_chan[i] + self.n_chan[i]), self.full_matrix.shape[1])
                self.full_matrix[i, low:high] = self.matrix_entry[i][0:high - low]

            else:

                for j in range(n_grp):
                    low = self.f_chan[i][j]
                    high = min(self.f_chan[i][j] + self.n_chan[i][j], self.full_matrix.shape[1])

                    self.full_matrix[i, low:high] = self.matrix_entry[i][base:base + self.n_chan[i][j]]

                    base += self.n_chan[i][j]

        # Transposed matrix so that we just have to multiply by the spectrum
        self.full_matrix = self.full_matrix.T
        #self.sparse_matrix = sparse.BCOO.fromdense(jnp.copy(self.full_matrix))

    @classmethod
    def from_file(cls, rmf_file):

        matrix_table = QTable.read(rmf_file, 'MATRIX')
        ebounds_table = QTable.read(rmf_file, 'EBOUNDS')

        return cls(matrix_table['ENERG_LO'],
                   matrix_table['ENERG_HI'],
                   matrix_table['N_GRP'],
                   matrix_table['F_CHAN'],
                   matrix_table['N_CHAN'],
                   matrix_table['MATRIX'],
                   ebounds_table['CHANNEL'],
                   ebounds_table['E_MIN'],
                   ebounds_table['E_MAX'])


    def plot(self):

        import cmasher as cmr
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        energy_in = np.array(self.energ_lo+self.energ_hi)/2
        energy_out = np.array(self.e_min+self.e_max)/2
        mappable = ax.pcolormesh(energy_out, energy_in, self.full_matrix.T, shading='flat', cmap=cmr.cosmic)
        plt.xlabel(r'$E_{spectrum}$')
        plt.ylabel(r'$E_{instrument}$')
        plt.colorbar(mappable=mappable)
        plt.loglog()
        e = np.linspace(-6, 2, 1000)
        plt.plot(e, e)
        plt.xlim(left=min(energy_out), right=max(energy_out))
        plt.ylim(bottom=min(energy_in), top=max(energy_in))
        plt.show()

        return fig
