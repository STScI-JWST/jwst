"""Regression tests for NIRSpec IFU"""
import pytest

from . import regtestdata as rt

from astropy.io.fits.diff import FITSDiff

import jwst.datamodels as dm
from jwst.flatfield.flat_field import nirspec_ifu

# Define artifactory source and truth
INPUT_PATH = 'nirspec/ifu'
TRUTH_PATH = 'truth/test_nirspec_ifu'


@pytest.fixture(scope='module')
def run_spec2(jail, rtdata_module):
    """Run the Spec2Pipeline on a single exposure"""
    rtdata = rtdata_module

    # Setup the inputs
    asn_name = 'single_nrs1_ifu_spec2_asn.json'
    asn_path = INPUT_PATH + '/' + asn_name

    # Run the pipeline
    step_params = {
        'input_path': asn_path,
        'step': 'calwebb_spec2.cfg',
        'args': [
            '--steps.bkg_subtract.save_results=true',
            '--steps.assign_wcs.save_results=true',
            '--steps.imprint_subtract.save_results=true',
            '--steps.msa_flagging.save_results=true',
            '--steps.extract_2d.save_results=true',
            '--steps.flat_field.save_results=true',
            '--steps.srctype.save_results=true',
            '--steps.straylight.save_results=true',
            '--steps.fringe.save_results=true',
            '--steps.pathloss.save_results=true',
            '--steps.barshadow.save_results=true',
            '--steps.photom.save_results=true',
            '--steps.resample_spec.save_results=true',
            '--steps.cube_build.save_results=true',
            '--steps.extract_1d.save_results=true',
        ]
    }

    rtdata = rt.run_step_from_dict(rtdata, **step_params)
    return rtdata


@pytest.mark.slow
@pytest.mark.bigdata
@pytest.mark.parametrize(
    'suffix',
    ['assign_wcs', 'cal', 'flat_field', 'imprint_subtract', 'msa_flagging',
     'pathloss', 'photom', 's3d', 'srctype', 'x1d']
)
def test_spec2(run_spec2, fitsdiff_default_kwargs, suffix):
    """Regression test matching output files"""
    rt.is_like_truth(run_spec2, fitsdiff_default_kwargs, suffix,
                     truth_path=TRUTH_PATH)


def test_nirspec_ifu_predefined_flat(jail, rtdata_module, fitsdiff_default_kwargs):
    """Test using predefined interpolated flat"""
    rtdata = rtdata_module
    data = dm.open(rtdata.get_data('nirspec/ifu/nrs_ifu_nrs1_assign_wcs.fits'))
    interpolated_flat = dm.open(rtdata.get_data('nirspec/ifu/nrs_ifu_nrs1_interpolated_flat.fits'))

    nirspec_ifu(data, None, None, None, None, interpolated_flat=interpolated_flat)
    rtdata.output = 'ff_using_interpolated.fits'
    data.write(rtdata.output)

    rtdata.get_truth(TRUTH_PATH + '/' + 'ff_using_interpolated.fits')
    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()

