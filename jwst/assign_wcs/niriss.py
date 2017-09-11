from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import logging
from copy import deepcopy

from asdf import AsdfFile
from astropy import coordinates as coord
from astropy import units as u
from astropy.modeling.models import Const1D, Mapping, Scale, Identity
import gwcs.coordinate_frames as cf
from gwcs import wcs

from .util import not_implemented_mode, subarray_transform
from . import pointing
from ..transforms.models import (NirissSOSSModel,
                                 NIRISSForwardRowGrismDispersion,
                                 NIRISSBackwardGrismDispersion,
                                 NIRISSForwardColumnGrismDispersion)
from ..datamodels import ImageModel, NIRISSGrismModel

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def create_pipeline(input_model, reference_files):
    '''
    get reference files from crds

    '''

    exp_type = input_model.meta.exposure.type.lower()
    pipeline = exp_type2transform[exp_type](input_model, reference_files)

    return pipeline


def niriss_soss_set_input(model, order_number):
    """
    Get the right model given the order number.

    Parameters
    ----------
    model - Input model
    order_number - the, well, order number desired

    Returns
    -------
    WCS - the WCS corresponding to the order_number

    """

    # Make sure the order number is correct.
    if order_number < 1 or order_number > 3:
        raise ValueError('Order must be between 1 and 3')

    # Return the correct transform based on the order_number
    obj = model.meta.wcs.forward_transform.get_model(order_number)

    # use the size of the input subarray7
    detector = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    spec = cf.SpectralFrame(name='spectral', axes_order=(2,), unit=(u.micron,),
                            axes_names=('wavelength',))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(),
                            axes_names=('ra', 'dec'),
                            axes_order=(0, 1), unit=(u.deg, u.deg), name='sky')
    world = cf.CompositeFrame([sky, spec], name='world')
    pipeline = [(detector, obj),
                (world, None)
                ]

    return wcs.WCS(pipeline)


def niriss_soss(input_model, reference_files):
    """
    The NIRISS SOSS pipeline includes 3 coordinate frames -
    detector, focal plane and sky

    reference_files={'specwcs': 'soss_wavelengths_configuration.asdf'}
    """

    # Get the target RA and DEC, they will be used for setting the WCS RA and DEC based on a conversation
    # with Kevin Volk.
    try:
        target_ra = float(input_model['meta.target.ra'])
        target_dec = float(input_model['meta.target.dec'])
    except:
        # There was an error getting the target RA and DEC, so we are not going to continue.
        raise ValueError('Problem getting the TARG_RA or TARG_DEC from input model {}'.format(input_model))

    # Define the frames
    detector = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    spec = cf.SpectralFrame(name='spectral', axes_order=(2,), unit=(u.micron,),
                            axes_names=('wavelength',))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(),
                            axes_names=('ra', 'dec'),
                            axes_order=(0, 1), unit=(u.deg, u.deg), name='sky')
    world = cf.CompositeFrame([sky, spec], name='world')
    try:
        with AsdfFile.open(reference_files['specwcs']) as wl:
            wl1 = wl.tree[1].copy()
            wl2 = wl.tree[2].copy()
            wl3 = wl.tree[3].copy()
    except Exception as e:
        raise IOError('Error reading wavelength correction from {}'.format(reference_files['specwcs']))

    subarray2full = subarray_transform(input_model)

    # Reverse the order of inputs passed to Tabular because it's in python order in modeling.
    # Consider changing it in modelng ?
    cm_order1 = subarray2full | (Mapping((0, 1, 1, 0)) | \
                                 (Const1D(target_ra) & Const1D(target_dec) & wl1)
                                 ).rename('Order1')
    cm_order2 = subarray2full | (Mapping((0, 1, 1, 0)) | \
                                 (Const1D(target_ra) & Const1D(target_dec) & wl2)
                                 ).rename('Order2')
    cm_order3 = subarray2full | (Mapping((0, 1, 1, 0)) | \
                                 (Const1D(target_ra) & Const1D(target_dec) & wl3)
                                 ).rename('Order3')

    # Define the transforms, they should accept (x,y) and return (ra, dec, lambda)
    soss_model = NirissSOSSModel([1, 2, 3],
                                 [cm_order1, cm_order2, cm_order3]
                                 ).rename('3-order SOSS Model')

    # Define the pipeline based on the frames and models above.
    pipeline = [(detector, soss_model),
                (world, None)
                ]

    return pipeline


def imaging(input_model, reference_files):
    """
    The NIRISS imaging pipeline includes 3 coordinate frames -
    detector, focal plane and sky

    reference_files={'distortion': 'jwst_niriss_distortioon_0001.asdf'}
    """
    detector = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    v2v3 = cf.Frame2D(name='v2v3', axes_order=(0, 1), unit=(u.deg, u.deg))
    world = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world')

    subarray2full = subarray_transform(input_model)
    imdistortion = imaging_distortion(input_model, reference_files)
    distortion = subarray2full | imdistortion
    distortion.bounding_box = imdistortion.bounding_box
    del imdistortion.bounding_box
    tel2sky = pointing.v23tosky(input_model)
    pipeline = [(detector, distortion),
                (v2v3, tel2sky),
                (world, None)]
    return pipeline


def imaging_distortion(input_model, reference_files):
    distortion = AsdfFile.open(reference_files['distortion']).tree['model']
    # Convert to deg.  Output of distortion model is in arcsec.
    transform = distortion | Scale(1 / 3600) & Scale(1 / 3600)

    try:
        # assign the bounding box to the entire compound model
        transform.bounding_box = distortion.bounding_box
    except NotImplementedError:
        shape = input_model.data.shape
        # Note: Since bounding_box is attached to the model here it's in reverse order.
        transform.bounding_box = ((-0.5, shape[0] - 0.5),
                                   (-0.5, shape[1] - 0.5))
    return transform


def wfss(input_model, reference_files):
    """
    Create the WCS pipeline for a NIRISS grism observation.

    Parameters
    ----------
    input_model: jwst.datamodels.ImagingModel
        The input datamodel, derived from datamodels
    reference_files: dict
        Dictionary specifying reference file names

    Notes
    -----
    reference_files = {
        "specwcs": 'GR150C_F090W.asdf'
        "distortion": 'NRCA1_FULL_distortion.asdf'
    }

    The tree in the grism reference file has a section for each order/beam as
    well as the link to the filter data file, not sure if there will be a
    separate passband reference file needed for the wavelength scaling or the
    wedge offsets. This file is currently created in
    jwreftools/niriss/niriss_reftools

    The direct image the catalog has been created from was corrected for
    distortion, but the dispersed images have not. This is OK if the trace and
    dispersion solutions are defined with respect to the distortion-corrected
    image. The catalog from the combined direct image has object locations in
    in detector space and the RA DEC of the object on sky.

    The WCS information for the grism image  plus the observed filter will be
    used to translate these to pixel locations for each of the objects.
    The grism images will then use their grism trace information to translate
    to detector space. The translation is assumed to be one-to-one for purposes
    of identifying the center of the object trace.

    The extent of the trace for each object can then be calculated based on
    the grism in use (row or column). Where the left/bottom of the trace starts
    at t = 0 and the right/top of the trace ends at t = 1, as long as they
    have been defined as such by th team.

    The extraction box is calculated to be the minimum bounding box of the
    object extent in the segmentation map associated with the direct image.
    The values of the min and max corners are saved in the photometry
    catalog in units of RA,DEC so they can be translated to pixels by
    the dispersed image's imaging wcs.

    For each spectral order, the configuration file contains a
    magnitude-cutoff value. Sources with magnitudes fainter than the
    extraction cutoff (MMAG_EXTRACT)  will not be extracted, but are
    accounted for when computing the spectral contamination and background
    estimates. The default extraction value is 99 right now.

    The sensitivity information from the original aXe style configuration
    file needs to be modified by the passband of the filter used for
    the direct image to get the min and max wavelengths
    which correspond to t=0 and t=1, this currently has been done by the team
    and the min and max wavelengths to use to calculate t are stored in the
    grism reference file as wrange, which can be selected by wrange_selector
    which contains the filter names.


    Step 1: Convert the source catalog from the reference frame of the
            uberimage to that of the dispersed image.  For the Vanilla
            Pipeline we assume that the pointing information in the file
            headers is sufficient.  This will be strictly true if all images
            were obtained in a single visit (same guide stars).
    Step 2: Record source information for each object in the catalog: position
            (RA and Dec), shape (A_IMAGE, B_IMAGE, THETA_IMAGE), and all
            available magnitudes.
    Step 3: Compute the trace and wavelength solutions for each object in the
            catalog and for each spectral order.  Record this information.
    Step 4: Compute the WIDTH of each spectral subwindow, which may be fixed or
            variable (see discussion of optimal extraction, below).  Record
            this information.
    Step 4: Record the MMAG_EXTRACT for each object and spectral order.

    Source catalog use moved to extract_2d
    """

    # The input is the grism image
    if not isinstance(input_model, ImageModel):
        raise TypeError('The input data model must be an ImageModel.')

    # make sure this is a grism image
    if "NIS_WFSS" not in input_model.meta.exposure.type:
            raise TypeError('The input exposure is not NIRISS grism')

    # Create the empty detector as a 2D coordinate frame in pixel units
    gdetector = cf.Frame2D(name='grism_detector',
                           axes_order=(0, 1),
                           unit=(u.pix, u.pix))

    # translate the x,y detector-in to x,y detector out coordinates
    # Get the disperser parameters which are defined as a model for each
    # spectral order
    with NIRISSGrismModel(reference_files['specwcs']) as f:
        dispx = f.dispx
        dispy = f.dispy
        displ = f.displ
        invdispl = f.invdispl
        orders = f.orders
        fwcpos_ref = f.fwcpos_ref

    # sep the row and column grism models
    if 'R' in input_model.instrument.pupil[-1]:
        det2det = NIRISSForwardRowGrismDispersion(orders, displ, dispx, dispy, fwcpos_ref)
    else:
        det2det = NIRISSForwardColumnGrismDispersion(orders, displ, dispx, dispy, fwcpos_ref)

    backward = NIRISSBackwardGrismDispersion(orders, invdispl, dispx, dispy, fwcpos_ref)
    det2det.inverse = backward

    # create the pipeline to construct a WCS object for the whole image
    # which can translate ra,dec to image frame reference pixels
    # it also needs to be part of the grism image wcs pipeline to
    # go from detector to world coordinates. However, the grism image
    # will be effectively translating pixel->world coordinates in a
    # manner that gives you the originating pixels ra and dec, not the
    # pure ra/dec on the sky from the pointing wcs.

    # use the imaging_distortion reference file here
    img_reference = deepcopy(reference_files)
    img_reference['distortion'] = reference_files['imaging_distortion']
    image_pipeline = imaging(input_model, img_reference)
    del img_reference

    # forward input is (x,y,lam,order) -> x, y
    # backward input needs to be the same ra, dec, lam, order -> x, y
    grism_pipeline = [(gdetector, det2det)]

    # pass through the wave and beam on the pipeline
    imagepipe = []
    world, _ = image_pipeline.pop()
    for cframe, trans in image_pipeline:
        trans = trans & (Identity(2))
        imagepipe.append((cframe, trans))
    imagepipe.append((world, None))
    grism_pipeline.extend(imagepipe)

    return grism_pipeline


exp_type2transform = {'nis_image': imaging,
                      'nis_wfss': wfss,
                      'nis_soss': niriss_soss,
                      'nis_ami': imaging,
                      'nis_tacq': imaging,
                      'nis_taconfirm': imaging,
                      'nis_focus': imaging,
                      'nis_dark': not_implemented_mode,
                      'nis_lamp': not_implemented_mode,
                      }
