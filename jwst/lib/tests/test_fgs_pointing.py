"""Test suite for ensuring correct FGS pointing"""
import os.path
import logging
from numpy import array

from jwst.datamodels import Level1bModel
from jwst.lib import set_telescope_pointing as stp


# Set logging for the module to be tested.
logger = logging.getLogger(stp.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

siaf_db = os.path.join(os.path.dirname(__file__), 'data', 'siaf.db')

# Define minimal model meta structure
WCS_META = {
    'meta': {
        'exposure': {
            'type': 'FGS_ACQ1',
        },
        'aperture': {
            'name': 'FGS1_FULL',
        },
        'observation': {
            'date': '2017-01-01',
        }
    }
}


def test_fgs_pointing():
    model = make_level1b()
    stp.update_wcs(model, siaf_path=siaf_db)

    assert model.meta.wcsinfo.pc1_1 == -0.9997617158628777
    assert model.meta.wcsinfo.pc1_2 == -0.021829143247382235
    assert model.meta.wcsinfo.pc2_1 == -0.021829143247382235
    assert model.meta.wcsinfo.pc2_2 == 0.9997617158628777


# ---------
# Utilities
# ---------
def make_level1b():
    data = array([1.])
    data.shape = (1, 1, 1, 1)
    model = Level1bModel(data)
    model.update(WCS_META)
    return model
