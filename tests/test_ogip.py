import os
import sys
import numpy as np
from unittest import TestCase
from jaxspec.instru.ogip import DataARF, DataRMF, DataPHA
from ref_clarsach_rsp import ARF as RefARF, RMF as RefRMF

#Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestRSP(TestCase):

    arf_files = [os.path.join(current_dir, file)
                 for file in ['data/ogip/PN.arf', 'data/ogip/M1.arf', 'data/ogip/M2.arf']]
    rmf_files = [os.path.join(current_dir, file)
                 for file in ['data/ogip/PN.rmf', 'data/ogip/M1.rmf', 'data/ogip/M2.rmf']]
    pha_files = [os.path.join(current_dir, file)
                 for file in ['data/ogip/nustar_pha.pha', 'data/ogip/xmm_pha.fits']]

    def test_arf(self):
        """
        Test reading an ARF file using XMM's PN, M1 and M2 files
        """

        for arf_file in self.arf_files:

            test_arf = DataARF.from_file(arf_file)
            ref_arf = RefARF(arf_file)
            print(current_dir)
            assert np.isclose(test_arf.specresp.value, ref_arf.specresp).all()

    def test_rmf(self):
        """
        Test reading an ARF file using XMM's PN, M1 and M2 files
        """

        # We should modify clarsach_rsp if we want to test with PN's RMF
        for rmf_file in self.rmf_files[1:2]:

            test_rmf = DataRMF.from_file(rmf_file)
            ref_rmf = RefRMF(rmf_file)

            dummy_spec = np.ones(test_rmf.energ_lo.shape)

            assert np.isclose(test_rmf.full_matrix@dummy_spec, ref_rmf.apply_rmf(dummy_spec)).all()

    # def test_sparse(self):
    #     """
    #     Test consistency between sparse and dense matrix
    #     """
    #
    #     for rmf_file in self.rmf_files:
    #
    #         test_rmf = DataRMF.from_file(rmf_file)
    #         dummy_spec = np.ones(test_rmf.energ_lo.shape)
    #
    #         assert np.isclose(test_rmf.full_matrix @ dummy_spec, test_rmf.sparse_matrix @ dummy_spec).all()

    def test_pha(self):
        """
        Test opening various PHA files from fits and pha
        """

        for pha_file in self.pha_files:

            test_rmf = DataPHA.from_file(pha_file)
