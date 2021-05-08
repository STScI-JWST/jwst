"""Test set_telescope_pointing using original Northrup-Grumman data

Based on the notebook `notebooks/pointing_full_test.ipynb`

Data is what was originally passed to the Telescopes group, Wayne Kinzel,
to check accuracy of STScI-derived pointing vs. NGAS.

The `original` method should pass this test. However, later methods
and revisions to the algorithms will fail this test. The use of this
test is to ensure that the original baseline did meet documented specs.
"""
from pathlib import Path

from astropy.table import Table
import numpy as np

from jwst.lib import set_telescope_pointing as stp

NGAS_DATA_PATH = Path(__file__).parent / 'data/acs_tlm_data_4_stsci_dms_jitter_file_mod_RA_DEC_PA.csv'


def row_to_pointing(r, siaf=None):
    """Convert a row of the spreadsheet to a Pointing

    Parameters
    ----------
    r: dict
        A row of pointing information

    siaf: SIAF
        SIAF info to use. If None, a non-transformative
        SIAF will be used.

    Returns
    -------
    Pointing, vinfo_expected
        A `Pointing` instance filled out.
        A `WCS` instance with the expected V1 information.
    """
    if siaf is None:
        siaf = stp.SIAF(v2_ref=0., v3_ref=0., v3yangle=0., vparity=1.)

    q = np.asarray([
        r['SA_ZATTEST1'],
        r['SA_ZATTEST2'],
        r['SA_ZATTEST3'],
        r['SA_ZATTEST4'],
    ])

    j2fgs_matrix = np.asarray([
        r['SA_ZRFGS2J11'], r['SA_ZRFGS2J21'], r['SA_ZRFGS2J31'],
        r['SA_ZRFGS2J12'], r['SA_ZRFGS2J22'], r['SA_ZRFGS2J32'],
        r['SA_ZRFGS2J13'], r['SA_ZRFGS2J23'], r['SA_ZRFGS2J33'],
    ])

    fsmcorr = np.asarray([
        r['SA_ZADUCMDX'], r['SA_ZADUCMDY']
    ])

    p = stp.Pointing(q=q, j2fgs_matrix=j2fgs_matrix, fsmcorr=fsmcorr)

    v = stp.WCSRef(ra=r['Vra'], dec=r['Vdec'], pa=r['V3PAatV1'])

    return p, v


def wcs_equality(left, right):
    """Assert equality of to WCSRef instances"""
    assert np.isclose(left.ra, right.ra), 'RAs are different {} != {}'.format(left.ra, right.ra)
    assert np.isclose(left.dec, right.dec), 'DECs are different {} != {}'.format(left.dec, right.dec)
    assert np.isclose(left.pa, right.pa), 'PAs are different {} != {}'.format(left.pa, right.pa)


def test_stp_ngas():
    """Test set_telescope_pointing using original Northrup-Grumman data

    Based on the notebook `notebooks/pointing_full_test.ipynb`

    Data is what was originally passed to the Telescopes group, Wayne Kinzel,
    to check accuracy of STScI-derived pointing vs. NGAS.

    The `original` method should pass this test. However, later methods
    and revisions to the algorithms will fail this test. The use of this
    test is to ensure that the original baseline did meet documented specs.
    """

    data = Table.read(NGAS_DATA_PATH)

    pointings = [
        row_to_pointing(r)
        for r in data
    ]

    # Setup transformation parameters
    t_pars = stp.TransformParameters()
    t_pars.method = stp.Methods.ORIGINAL
    t_pars.siaf = stp.SIAF(v2_ref=0., v3_ref=0., v3yangle=0., vparity=1.)
    any_wrong = False
    for idx, pv in enumerate(pointings):
        t_pars.pointing, v = pv
        wcsinfo, vinfo = stp.calc_wcs(t_pars)
        try:
            wcs_equality(vinfo, v)
        except AssertionError as e:
            any_wrong = True
            print('{}'.format(idx))
            print(e)

    assert not any_wrong
