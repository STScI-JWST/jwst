"""
Test NIRCAM grism extract_2d functionality.

Notes:
No images are needed here to check the location
and size of bounding boxes.

NIRCAM and NIRISS WFSS use the same code, testing with NIRCAM
settings here for functionality. In the future, full data
regression tests will provide the truth between instruments.

For the testing catalog:
objects 9 and 19 should have order 1 extracted
object 25 should have partial boxes for both orders
object 26 should be excluded
"""
import os
import pytest
import numpy as np

from astropy.io import fits
from gwcs import wcs

from jwst.datamodels.image import ImageModel
from jwst.datamodels import CubeModel, SlitModel, MultiSlitModel
from jwst.assign_wcs.util import create_grism_bbox
from jwst.assign_wcs import AssignWcsStep, nircam

from jwst.extract_2d.extract_2d_step import Extract2dStep
from jwst.extract_2d.grisms import extract_tso_object, extract_grism_objects
from jwst.extract_2d.tests import data


# Allowed settings for nircam
tsgrism_filters = ['F277W', 'F444W', 'F322W2', 'F356W']

data_path = os.path.split(os.path.abspath(data.__file__))[0]


# Default wcs information
# This is set for a standard nircam image just as an example
# It does not test the validity of the absolute results
# for create_tso_wcsimage, set the width of the output image to this value:
NIRCAM_TSO_WIDTH = 10
wcs_image_kw = {'wcsaxes': 2, 'ra_ref': 53.1490299775, 'dec_ref': -27.8168745624,
                'v2_ref': 86.103458, 'v3_ref': -493.227512, 'roll_ref': 45.04234459270135,
                'crpix1': 1024.5, 'crpix2': 1024.5,
                'crval1': 53.1490299775, 'crval2': -27.8168745624,
                'cdelt1': 1.81661111111111e-05, 'cdelt2': 1.8303611111111e-05,
                'ctype1': 'RA---TAN', 'ctype2': 'DEC--TAN',
                'pc1_1': -0.707688557183348, 'pc1_2': 0.7065245261360363,
                'pc2_1': 0.7065245261360363, 'pc2_2': 1.75306861111111e-05,
                'cunit1': 'deg', 'cunit2': 'deg',
                }

wcs_wfss_kw = {'wcsaxes': 2, 'ra_ref': 53.1423683802, 'dec_ref': -27.8171119969,
               'v2_ref': 86.103458, 'v3_ref': -493.227512, 'roll_ref': 45.04234459270135,
               'crpix1': 1024.5, 'crpix2': 1024.5,
               'crval1': 53.1423683802, 'crval2': -27.8171119969,
               'cdelt1': 1.74460027777777e-05, 'cdelt2': 1.75306861111111e-05,
               'ctype1': 'RA---TAN', 'ctype2': 'DEC--TAN',
               'pc1_1': -0.7076885519484576, 'pc1_2': 0.7065245313795517,
               'pc2_1': 0.7065245313795517, 'pc2_2': 0.7076885519484576,
               'cunit1': 'deg', 'cunit2': 'deg',
               }

wcs_tso_kw = {'wcsaxes': 2, 'ra_ref': 86.9875, 'dec_ref': 23.423,
              'v2_ref': 95.043034, 'v3_ref': -556.150466, 'roll_ref': 359.9521,
              }


def get_file_path(filename):
    """
    Construct an absolute path.
    """
    return os.path.join(data_path, filename)


def create_hdul(detector='NRCALONG', channel='LONG', module='A',
                filtername='F335M', exptype='NRC_IMAGE', pupil='CLEAR',
                subarray='FULL', wcskeys=wcs_image_kw):
    hdul = fits.HDUList()
    phdu = fits.PrimaryHDU()
    phdu.header['telescop'] = "JWST"
    phdu.header['filename'] = "test+" + filtername
    phdu.header['instrume'] = 'NIRCAM'
    phdu.header['channel'] = channel
    phdu.header['detector'] = detector
    phdu.header['FILTER'] = filtername
    phdu.header['PUPIL'] = pupil
    phdu.header['MODULE'] = module
    phdu.header['time-obs'] = '8:59:37'
    phdu.header['date-obs'] = '2017-09-05'
    phdu.header['exp_type'] = exptype
    phdu.header['SUBARRAY'] = subarray
    phdu.header['SUBSIZE1'] = 2048
    phdu.header['SUBSIZE2'] = 2048
    phdu.header['SUBSTRT1'] = 1
    phdu.header['SUBSTRT2'] = 1
    scihdu = fits.ImageHDU()
    scihdu.header['EXTNAME'] = "SCI"
    scihdu.header.update(wcskeys)
    hdul.append(phdu)
    hdul.append(scihdu)
    return hdul


def create_wfss_wcs(pupil, filtername='F335M'):
    """Help create WFSS GWCS object."""
    hdul = create_hdul(exptype='NRC_WFSS', filtername=filtername,
                       pupil=pupil, wcskeys=wcs_wfss_kw)
    im = ImageModel(hdul)
    ref = get_reference_files(im)
    pipeline = nircam.create_pipeline(im, ref)
    wcsobj = wcs.WCS(pipeline)
    return wcsobj


def create_wfss_image(pupil, filtername='F444W'):
    hdul = create_hdul(exptype='NRC_WFSS', filtername=filtername,
                       pupil=pupil, wcskeys=wcs_wfss_kw)
    hdul['sci'].data = np.ones((hdul[0].header['SUBSIZE1'], hdul[0].header['SUBSIZE2']))
    im = ImageModel(hdul)
    return AssignWcsStep.call(im)


def create_tso_wcsimage(filtername="F277W", subarray=False):
    """Help create tsgrism GWCS object."""
    if subarray:
        subarray = "SUBGRISM256"
    else:
        subarray = "FULL"
    hdul = create_hdul(exptype='NRC_TSGRISM', pupil='GRISMR',
                       filtername=filtername, detector='NRCALONG',
                       subarray=subarray, wcskeys=wcs_tso_kw)
    hdul['sci'].header['SUBSIZE1'] = NIRCAM_TSO_WIDTH

    if subarray:
        hdul['sci'].header['SUBSIZE2'] = 256
        subsize = 256
    else:
        hdul['sci'].header['SUBSIZE2'] = 2048
        subsize = 2048

    hdul['sci'].data = np.ones((2, subsize, NIRCAM_TSO_WIDTH))
    im = CubeModel(hdul)
    im.meta.wcsinfo.siaf_xref_sci = 887.0
    im.meta.wcsinfo.siaf_yref_sci = 35.0
    aswcs = AssignWcsStep()
    return aswcs.process(im)


def get_reference_files(datamodel):
    """Get the reference files associated with extract 2d."""
    refs = {}
    step = Extract2dStep()
    for reftype in Extract2dStep.reference_file_types:
        refs[reftype] = step.get_reference_file(datamodel, reftype)
    return refs


@pytest.fixture(params=tsgrism_filters)
def tsgrism_inputs(request):
    def _add_missing_key(missing_key=None):
        tso_kw = wcs_tso_kw.copy()
        tso_kw.update({'xref_sci': 887.0, 'yref_sci': 35.0})

        if missing_key is not None:
            tso_kw[missing_key] = None

        hdu = create_hdul(
            exptype='NRC_TSGRISM',
            pupil='GRISMR',
            filtername=request.param,
            detector='NRCALONG',
            subarray='SUBGRISM256',
            wcskeys=tso_kw,
            channel='LONG',
            module='A',
        )

        image_model = CubeModel(hdu)

        return image_model, get_reference_files(image_model)

    return _add_missing_key


@pytest.mark.parametrize('key', ['xref_sci', 'yref_sci'])
def test_extract_tso_object_fails_without_xref_yref(tsgrism_inputs, key):
    with pytest.raises(ValueError):
        image_model, refs = tsgrism_inputs(missing_key=key)
        extract_tso_object(image_model, reference_files=refs)


@pytest.mark.filterwarnings("ignore: Card is too long")
def test_create_box_fits():
    """Make sure that a box is created around a source catalog object.
    This version allows use of the FITS WCS to translate the source location

    The objects selected here should be contained on the image
    """
    source_catalog = get_file_path('step_SourceCatalogStep_cat.ecsv')
    hdul = create_hdul(exptype='NRC_WFSS', pupil='GRISMR', wcskeys=wcs_wfss_kw)
    im = ImageModel(hdul)
    aswcs = AssignWcsStep()
    imwcs = aswcs(im)
    imwcs.meta.source_catalog = source_catalog
    refs = get_reference_files(im)
    test_boxes = create_grism_bbox(imwcs, refs,
                                   mmag_extract=99.)

    assert len(test_boxes) >= 2  # the catalog has 4 objects
    for sid in [9, 19]:
        ids = [source for source in test_boxes if source.sid == sid]
        assert len(ids) == 1
        assert ids[0].xcentroid > 0
        assert ids[0].ycentroid > 0
        if sid == 19:
            assert [1, 2] == list(ids[0].order_bounding.keys())
        if sid == 9:
            assert [1] == list(ids[0].order_bounding.keys())


def test_create_box_gwcs():
    """Make sure that a box is created around a source catalog object.
    This version allows use of the GWCS to translate the source location.

    This is currently expected to fail because of the distortion
    reference file. The settings and catalog used should produce
    first order trace boxes for the objects.
    """
    source_catalog = get_file_path('step_SourceCatalogStep_cat.ecsv')
    hdul = create_hdul(exptype='NRC_WFSS', pupil='GRISMR', wcskeys=wcs_wfss_kw)
    im = ImageModel(hdul)
    aswcs = AssignWcsStep()
    imwcs = aswcs(im)
    imwcs.meta.source_catalog = source_catalog
    refs = get_reference_files(im)
    test_boxes = create_grism_bbox(imwcs, refs,
                                   mmag_extract=99.)
    assert len(test_boxes) >= 2  # the catalog has 4 objects
    for sid in [9, 19]:
        ids = [source for source in test_boxes if source.sid == sid]
        assert len(ids) == 1
        assert ids[0].xcentroid > 0
        assert ids[0].ycentroid > 0
        if sid == 19:
            assert [1, 2] == list(ids[0].order_bounding.keys())
        if sid == 9:
            assert [1] == list(ids[0].order_bounding.keys())


def setup_image_cat():
    """basic setup for image header and references."""
    source_catalog = get_file_path('step_SourceCatalogStep_cat.ecsv')
    hdul = create_hdul(exptype='NRC_WFSS', pupil='GRISMR', wcskeys=wcs_wfss_kw)
    im = ImageModel(hdul)
    im.meta.source_catalog = source_catalog
    aswcs = AssignWcsStep()
    imwcs = aswcs(im)
    refs = get_reference_files(im)

    return imwcs, refs


@pytest.mark.filterwarnings("ignore: Card is too long")
def test_create_specific_orders():
    """Test that boxes only for the specified orders
    are created.. instead of the default in the reference
    file.

     Notes
     -----
     The filter warning is for fits card length

     objects 9 and 19 should have order 1 extracted
     object 25 should have partial boxes for both orders
     object 26 should have order 2 excluded at order 1 partial
    """
    imwcs, refs = setup_image_cat()
    extract_orders = [1]  # just extract the first order
    test_boxes = create_grism_bbox(imwcs, refs,
                                   mmag_extract=99.,
                                   extract_orders=extract_orders)

    for sid in [9, 19]:
        ids = [source for source in test_boxes if source.sid == sid]
        assert len(ids) == 1
        assert [1] == list(ids[0].order_bounding.keys())


def test_extract_tso_subarray():
    """Test extraction of a TSO object.

    NRC_TSGRISM mode doesn't use the catalog since
    objects are always in the same place on the
    detector. This does an actual test of the
    extraction with a small CubeModel
    """

    wcsimage = create_tso_wcsimage(subarray=True)
    refs = get_reference_files(wcsimage)
    outmodel = extract_tso_object(wcsimage,
                                  reference_files=refs)
    assert isinstance(outmodel, SlitModel)
    assert outmodel.source_xpos == (outmodel.meta.wcsinfo.siaf_xref_sci - 1)
    assert outmodel.source_ypos == 34
    assert outmodel.source_id == 1
    assert outmodel.xstart > 0
    assert outmodel.ystart > 0
    assert outmodel.meta.wcsinfo.spectral_order == 1

    # These are the sizes of the valid wavelength regions
    # not the size of the cutout
    assert outmodel.ysize > 0
    assert outmodel.xsize > 0
    with pytest.raises(TypeError):
        extract_tso_object(wcsimage, reference_files=refs,
                           extract_orders=1)
    with pytest.raises(TypeError):
        extract_tso_object(wcsimage, reference_files=refs,
                           extract_orders=['1'])
    with pytest.raises(NotImplementedError):
        extract_tso_object(wcsimage, reference_files=refs,
                           extract_orders=[1, 2])
    with pytest.raises(TypeError):
        extract_tso_object(wcsimage, reference_files='myspecwcs.asdf')
    with pytest.raises(KeyError):
        extract_tso_object(wcsimage, reference_files={})
    del outmodel


def test_extract_tso_height():
    """Test extraction of a TSO object with given height.

    NRC_TSGRISM mode doesn't use the catalog since
    objects are always in the same place on the
    detector. This does an actual test of the
    extraction with a small CubeModel
    """

    wcsimage = create_tso_wcsimage(subarray=False)
    refs = get_reference_files(wcsimage)
    outmodel = extract_tso_object(wcsimage,
                                  tsgrism_extract_height=50,
                                  reference_files=refs)
    assert isinstance(outmodel, SlitModel)
    assert outmodel.source_xpos == (outmodel.meta.wcsinfo.siaf_xref_sci - 1)
    assert outmodel.source_ypos == 34
    assert outmodel.source_id == 1
    assert outmodel.xstart > 0
    assert outmodel.ystart > 0
    assert outmodel.meta.wcsinfo.spectral_order == 1

    # These are the sizes of the valid wavelength regions
    # not the size of the cutout
    assert outmodel.ysize > 0
    assert outmodel.xsize > 0

    # check the size of the cutout
    num, ysize, xsize = outmodel.data.shape
    assert num == wcsimage.data.shape[0]
    assert ysize == 50
    assert xsize == NIRCAM_TSO_WIDTH
    del outmodel


@pytest.mark.filterwarnings("ignore: Card is too long")
def test_extract_wfss_object():
    """Test extraction of a WFSS object.

    Test extraction of 2 objects into a MultiSlitModel.
    The data is all ones, this just tests extraction
    on the detector of expected locations.

    """
    source_catalog = get_file_path('step_SourceCatalogStep_cat.ecsv')
    wcsimage = create_wfss_image(pupil='GRISMR')
    wcsimage.meta.source_catalog = source_catalog
    refs = get_reference_files(wcsimage)
    outmodel = extract_grism_objects(wcsimage,
                                     reference_files=refs,
                                     compute_wavelength=False)
    assert isinstance(outmodel, MultiSlitModel)
    assert len(outmodel.slits) == 3
    ids = [slit.source_id for slit in outmodel.slits]
    assert ids == [9, 19, 19]

    names = [slit.name for slit in outmodel.slits]
    assert names == ['9', '19', '19']

    with pytest.raises(TypeError):
        extract_tso_object(wcsimage, reference_files='myspecwcs.asdf')
    with pytest.raises(KeyError):
        extract_tso_object(wcsimage, reference_files={})
    with pytest.raises(ValueError):
        wcsimage.meta.exposure.type = 'NIS_IMAGE'
        extract_tso_object(wcsimage, reference_files=refs)
    with pytest.raises(ValueError):
        wcsimage.meta.instrument.name = 'NIRISS'
        extract_tso_object(wcsimage, reference_files=refs)


def test_wfss_extract_custom_height():
    """Test WFSS extraction with a user supplied half height.

     Notes
     -----
     The filter warning is for fits card length

     objects 9 and 19 should have order 1 extracted
     object 25 should have partial boxes for both orders
     object 26 should have order 2 excluded at order 1 partial
    """
    imwcs, refs = setup_image_cat()
    imwcs.meta.wcsinfo._instance['dispersion_direction'] = 1
    extract_orders = [1]  # just extract the first order
    test_boxes = create_grism_bbox(imwcs, refs,
                                   mmag_extract=99.,
                                   extract_orders=extract_orders,
                                   wfss_extract_half_height=5)

    for sid in [9, 19]:
        ids = [source for source in test_boxes if source.sid == sid]
        assert len(ids) == 1
        assert [1] == list(ids[0].order_bounding.keys())
        y_extent = ids[0].order_bounding[1][0]
        assert y_extent[1] - y_extent[0] == 10


def test_wfss_extract_custom_wavelength_range():
    """ Test WFSS extraction with a user supplied wavelength_range. """
    imwcs, refs = setup_image_cat()
    test_boxes = create_grism_bbox(imwcs, mmag_extract=99., wavelength_range={1: (3.01, 4.26)})

    for sid in [9, 19]:
        ids = [source for source in test_boxes if source.sid == sid]
        assert len(ids) == 1
        assert [1] == list(ids[0].order_bounding.keys())
