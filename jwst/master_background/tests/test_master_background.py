"""
Unit tests for master background subtraction
"""
import numpy as np
import pytest

from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep
from jwst.master_background import MasterBackgroundStep
from jwst.master_background.create_master_bkg import create_background
from jwst.master_background.master_background_step import (
    copy_background_to_surf_bright,
    split_container,
)
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs


@pytest.fixture(scope='module')
def user_background(tmpdir_factory):
    """Generate a user background spectrum"""

    filename = tmpdir_factory.mktemp('master_background_user_input')
    filename = str(filename.join('user_background.fits'))
    wavelength = np.linspace(0.5, 25, num=100)
    flux = np.linspace(2.0, 2.2, num=100)
    data = create_background(wavelength, flux)
    data.save(filename)
    return filename


@pytest.fixture(scope='function')
def science_image():
    """Generate science image """

    image = datamodels.ImageModel((10, 10))
    image.meta.instrument.name = 'MIRI'
    image.meta.instrument.detector = 'MIRIMAGE'
    image.meta.exposure.type = 'MIR_LRS-FIXEDSLIT'
    image.meta.observation.date = '2018-01-01'
    image.meta.observation.time = '00:00:00'
    image.meta.subarray.xstart = 1
    image.meta.subarray.ystart = 1
    image.meta.wcsinfo.v2_ref = 0
    image.meta.wcsinfo.v3_ref = 0
    image.meta.wcsinfo.roll_ref = 0
    image.meta.wcsinfo.ra_ref = 0
    image.meta.wcsinfo.dec_ref = 0
    image = AssignWcsStep.call(image)
    return image


def test_master_background_userbg(_jail, user_background, science_image):
    """Verify data can run through the step with a user-supplied background"""

    # Run with a user-supplied background and verify this is recorded in header
    collect_pipeline_cfgs('./config')
    result = MasterBackgroundStep.call(
        science_image,
        config_file='config/master_background.cfg',
        user_background=user_background,
    )

    assert type(science_image) is type(result)
    assert result is not science_image
    assert result.meta.cal_step.master_background == 'COMPLETE'
    assert result.meta.background.master_background_file == 'user_background.fits'


def test_master_background_logic(_jail, user_background, science_image):
    """Verify if calspec 2 background step was run the master background step will be skipped"""

    # the background step in calspec2 was done
    science_image.meta.cal_step.back_sub = 'COMPLETE'

    # Run with a user-supplied background
    collect_pipeline_cfgs('./config')
    result = MasterBackgroundStep.call(
        science_image,
        config_file='config/master_background.cfg',
        user_background=user_background,
    )

    assert result.meta.cal_step.master_background == 'SKIPPED'
    assert type(science_image) is type(result)

    # Now force it
    result = MasterBackgroundStep.call(
        science_image,
        config_file='config/master_background.cfg',
        user_background=user_background,
        force_subtract=True
    )

    assert result.meta.cal_step.master_background == 'COMPLETE'
    assert type(science_image) is type(result)


def test_copy_background_to_surf_bright():
    """Test the copy_background_to_surf_bright function"""

    wavelength = np.linspace(0.5, 25, num=100)
    surf_bright = np.linspace(2.0, 2.2, num=100) + (np.random.random(100) - 0.5) * 0.001
    sb_error = np.random.random(100) * 0.01 + 17.       # different from berror
    background = np.random.random(100) * 0.01 + 1
    berror = np.random.random(100) * 0.01
    data = create_background(wavelength, surf_bright)
    data.spec[0].spec_table['sb_error'] = sb_error
    data.spec[0].spec_table['background'] = background
    data.spec[0].spec_table['bkgd_error'] = berror

    newdata = data.copy()
    copy_background_to_surf_bright(newdata)

    assert (newdata.spec[0].spec_table['surf_bright'] == background).all()
    assert (newdata.spec[0].spec_table['sb_error'] == berror).all()
    assert (newdata.spec[0].spec_table['background'] == 0).all()


def test_split_container(tmp_path):
    path1 = tmp_path / "foo.fits"
    path2 = tmp_path / "bar.fits"
    im1 = datamodels.ImageModel()
    im1.save(path1)
    im2 = datamodels.ImageModel()
    im2.save(path2)
    container = datamodels.ModelContainer([im1, im2])
    container.meta.asn_table = {
        "products": [
            {
                "members": [
                    {
                        "expname": f"{path1}",
                        "exptype": "science"
                    },
                    {
                        "expname": f"{path2}",
                        "exptype": "background"
                    },
                ]
            }
        ]
    }

    sci, bkg = split_container(container)

    assert sci[0].meta.filename == "foo.fits"
    assert bkg[0].meta.filename == "bar.fits"
    assert len(sci) == 1
    assert len(bkg) == 1
