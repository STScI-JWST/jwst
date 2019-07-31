"""
Unit tests for pathloss correction
"""

from jwst.datamodels import MultiSlitModel, PathlossModel, IFUImageModel
from jwst.pathloss import PathLossStep
from jwst.pathloss.pathloss import (get_center, 
                                    get_aperture_from_model, 
                                    calculate_pathloss_vector,
                                    is_pointsource)
from jwst.pathloss.pathloss import do_correction 
import numpy as np
import pytest


# Begin get_center tests
def test_skip_step(make_imagemodel):
    """If no pathloss reference file is specified or given in the header,
    make sure the step skips.
    """

    im = make_imagemodel(10, 10, 'NRS_AUTOWAVE')
    result = PathLossStep.call(im)

    assert(result.meta.cal_step.pathloss == 'SKIPPED')


def test_get_center_ifu():
    """get_center assumes IFU targets are centered @ (0.0, 0.0)
    """

    x_pos,y_pos = get_center("NRS_IFU", None)

    assert(x_pos==y_pos==0.0)


def test_get_center_attr_err():
    """if no center provided for modes that are not IFU,
    center is assigned to (0.0, 0.0)
    """
    datmod = MultiSlitModel()
    x_pos, y_pos = get_center("NRS_MSASPEC", datmod)

    assert(x_pos==y_pos==0.0)


def test_get_center_exp_type():
    """if exp_type is not in NRS, center is returned (0.0,0.0) 
    """
    datmod = MultiSlitModel()
    x_pos, y_pos = get_center("NRC_IMAGE", datmod)

    assert(x_pos==y_pos==0.0)


def test_get_center_exptype():
    """ If exptype is "NRS_MSASPEC" | "NRS_FIXEDSLIT" | "NRS_BRIGHTOBJ" and
    source_xpos and source_ypos exist in datamod.slits, make sure it's returned
    """
    
    datmod = MultiSlitModel()
    datmod.slits.append({'source_xpos':1, 'source_ypos':2})
    
    x_pos, y_pos = get_center("NRS_FIXEDSLIT", datmod.slits[0])

    assert(x_pos==1)
    assert(y_pos==2)


# Begin get_aperture_from_model tests
def test_get_app_from_model_null():
    """If exp_type isn't the NRS or NIS specific mode,
    routine returns None
    """

    datmod = MultiSlitModel()
    datmod.meta.exposure.type = 'NRC_IMAGE'

    result = get_aperture_from_model(datmod, None)

    assert(result==None)


def test_get_aper_from_model_fixedslit():
    """For a given exposures aperture, make sure the correct
    aperture reference data is returned for fixedslit mode
    """

    datmod = PathlossModel()
    datmod.apertures.append({'name':'S200A1'})
    datmod.meta.exposure.type = 'NRS_FIXEDSLIT'

    result = get_aperture_from_model(datmod, 'S200A1')

    assert(result == datmod.apertures[0])


def test_get_aper_from_model_msa():
    """For a given exposures aperture, make sure the correct
    aperture reference data is returned for MSA mode
    """

    datmod = PathlossModel()
    datmod.apertures.append({'shutters':1})
    datmod.meta.exposure.type = 'NRS_MSASPEC'

    result = get_aperture_from_model(datmod, 1)

    assert(result == datmod.apertures[0])


# Begin calculate_pathloss_vector tests.
def test_calculate_pathloss_vector_pointsource_data():
    """Calculate pathloss vector for 3D pathloss data
    """
    datmod = PathlossModel()

    ref_data = {'pointsource_data':np.ones((10,10,10), dtype=np.float32),
                'pointsource_wcs': {'crval2': -0.5, 'crpix2': 1.0, 'cdelt2': 0.05, 
                                    'cdelt3': 1, 'crval1': -0.5, 'crpix1': 1.0, 
                                    'crpix3': 1.0, 'crval3': 1, 'cdelt1': 0.05}                  
                }

    datmod.apertures.append(ref_data)

    wavelength, pathloss, is_inside_slitlet = calculate_pathloss_vector(datmod.apertures[0].pointsource_data,
                                                                        datmod.apertures[0].pointsource_wcs,
                                                                        0.0, 0.0)

    # Wavelength array is calculated with this: crval3 +(float(i+1) - crpix3)*cdelt3
    # Where i is the iteration of np.arange(wavesize) which is the 1st dimension of the pointsource
    # data array.
    wavelength_comparison = np.array([1 + (float(i+1) - 1.0)*1 for i in np.arange(10)])
    assert(np.all(wavelength==wavelength_comparison))

    # pathloss vector gets assigned at beginning of calculate_pathloss_vector and in this
    # case, doesnt change (np.zeros(wavesize, dtype=np.float32))
    pathloss_comparison = np.zeros(10, dtype=np.float32)
    assert(np.all(pathloss==pathloss_comparison))

    # With the current wcs values, the logic should be returning False
    assert(is_inside_slitlet==False)

def test_calculate_pathloss_vector_uniform_data():
    """Calculate the pathloss vector for uniform data arrays.
    """
    datmod = PathlossModel()

    ref_data = {'uniform_data':np.ones((10,), dtype=np.float32),
                'uniform_wcs': {'crpix1': 1.0, 'cdelt1': 1, 'crval1': 1}}

    datmod.apertures.append(ref_data)

    wavelength, pathloss, _ = calculate_pathloss_vector(datmod.apertures[0].uniform_data,
                                                        datmod.apertures[0].uniform_wcs,
                                                        0.0, 0.0)
    
    # Wavelength array is calculated with this: crval1 +(float(i+1) - crpix1)*cdelt1
    # Where i is the iteration of np.arange(wavesize) which is the shape of the uniform
    # data array.
    comparison = np.array([1 +(float(i+1) - 1)*1 for i in np.arange(10)])
    assert(np.all(wavelength==comparison))
    
    # The same array is returned in this case
    assert(np.all(datmod.apertures[0].uniform_data == pathloss))


def test_calculate_pathloss_vector_interpolation():
    """Calculate the pathloss vector for when interpolation is necessary.
    """
    datmod = PathlossModel()

    ref_data = {'pointsource_data':np.ones((10,10,10), dtype=np.float32),
                'pointsource_wcs': {'crval2': -0.5, 'crpix2': 1.0, 'cdelt2': 0.5, 
                                    'cdelt3': 1.0, 'crval1': -0.5, 'crpix1': 1.0, 
                                    'crpix3': 1.0, 'crval3': 1.0, 'cdelt1': 0.5}}

    datmod.apertures.append(ref_data)

    wavelength, pathloss, is_inside_slitlet = calculate_pathloss_vector(datmod.apertures[0].pointsource_data,
                                                                        datmod.apertures[0].pointsource_wcs,
                                                                        0.0, 0.0)

    # Wavelength array is calculated with this: crval3 +(float(i+1) - crpix3)*cdelt3
    # Where i is the iteration of np.arange(wavesize) which is the 1st dimension of the pointsource
    # data array.
    wavelength_comparison = np.array([1 + (float(i+1) - 1.0)*1 for i in np.arange(10)])
    assert(np.all(wavelength==wavelength_comparison))

    # In this instance we interpolate to get the array for pathloss VS wavelength.
    # With the current values inside of the of the pointsource_wcs starting at line 143 of pathloss.py
    # dx1 = 1 - int(1) = 0.0
    # dx2 = 1 - dx1 = 1.0
    # dy1 = 1 - int(1) = 0.0
    # dy2 = 1 - dx1 = 1.0
    # This means a11 == a12 == a21 == 0 and a22 == 1
    # This simplifies that pathloss vector to:
    # pathloss_vector = (a22*pathloss_ref[:,i,j]) = (1*pathloss_ref[:1,j])
    # Thus pathloss == the input array to the function.
    pathloss_comparison = datmod.apertures[0].pointsource_data
    assert(np.all(pathloss==pathloss_comparison))

    # With the current wcs values, the logic should be returning True
    assert(is_inside_slitlet==True)


def test_is_pointsource():
    """Check to see if object it point source
    """
    point_source = None
    result = is_pointsource(point_source)
    assert(result == False)

    point_source = 'point'
    result = is_pointsource(point_source)
    assert(result == True)

    point_source = 'not a point'
    result = is_pointsource(point_source)
    assert(result == False)


def test_do_correction_msa_slit_size_eq_0():
    """If slits have size 0, quit calibration.
    """
    datmod = MultiSlitModel()
    datmod.slits.append({'data':np.empty((10,10))})
    pathlossmod = PathlossModel()
    datmod.meta.exposure.type = 'NRS_MSASPEC'

    result = do_correction(datmod, pathlossmod)
    assert(result.meta.cal_step.pathloss == 'COMPLETE')


@pytest.fixture(scope='function')
def make_imagemodel():
    '''Image model for testing'''
    def _im(ysize, xsize, exptype):
        # create the data arrays
        im = IFUImageModel((ysize, xsize))
        im.data = np.random.rand(ysize, xsize)
        im.meta.instrument.name = 'NIRSPEC'
        im.meta.exposure.type = exptype
        im.meta.observation.date = '2018-01-01'
        im.meta.observation.time = '00:00:00'
        
        return im

    return _im