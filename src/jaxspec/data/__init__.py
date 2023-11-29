import os
from .ogip import DataPHA, DataARF, DataRMF


def data_loader(pha_path, arf_path=None, rmf_path=None, bkg_path=None):
    """
    This function is a convenience function that allows to load PHA, ARF and RMF data
    from a given PHA file, using either the ARF/RMF/BKG filenames in the header or the
    specified filenames overwritten by the user.

    Parameters:
        pha_path: The PHA file path.
        arf_path: The ARF file path.
        rmf_path: The RMF file path.
        bkg_path: The BKG file path.
    """

    pha = DataPHA.from_file(pha_path)

    if arf_path is None:
        arf_path = os.path.join(os.path.dirname(pha_path), pha.ancrfile)
    if rmf_path is None:
        rmf_path = os.path.join(os.path.dirname(pha_path), pha.respfile)
    if bkg_path is None:
        bkg_path = os.path.join(os.path.dirname(pha_path), pha.backfile)

    arf = DataARF.from_file(arf_path)
    rmf = DataRMF.from_file(rmf_path)
    bkg = DataPHA.from_file(bkg_path) if bkg_path is not None else None

    return pha, arf, rmf, bkg
