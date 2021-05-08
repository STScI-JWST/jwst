#
#  Module for 2d extraction of grism spectra
#

import copy
import logging

import numpy as np

from astropy.modeling.models import Shift, Const1D, Mapping
from gwcs.wcstools import grid_from_bounding_box

from .. import datamodels
from ..assign_wcs import util
from ..datamodels import WavelengthrangeModel

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def extract_tso_object(input_model,
                       reference_files=None,
                       tsgrism_extract_height=None,
                       extract_orders=None,
                       compute_wavelength=True):
    """
    Extract the spectrum for a NIRCam TSO grism observation.

    Parameters
    ----------
    input_model : `~jwst.datamodels.CubeModel` or `~jwst.datamodels.ImageModel`
        The input TSO data is an instance of a CubeModel (3D) or ImageModel (2D)

    reference_files : dict
        Needs to include the name of the wavelengthrange reference file

    tsgrism_extract_height : int
        The extraction height, in total, for the spectrum in the
        cross-dispersion direction. If this is other than None,
        it will override the team default of 64 pixels. The team
        wants the source centered near row 34, so the extraction
        height is not the same on either size of the central row.

    extract_orders : list[ints]
        This is an optional parameter that will override the
        orders specified for extraction in the wavelengthrange
        reference file.

    compute_wavelength : bool
        Compute a wavelength array for the datamodel.

    Returns
    -------
    output_model : `~jwst.datamodels.SlitModel`


    Notes
    -----
    This method supports NRC_TSGRISM only, where only one bright object is
    considered in the field, so there's no catalog of sources and the object
    is assumed to have been observed at the aperture reference position.
    The aperture reference location is read during level-1b (uncal) product
    creation by the "set_telescope_pointing" script from the SIAF entries
    XSciRef and YSciRef (reference location in the science frame) and saved as
    "meta.wcsinfo.siaf_xref_sci" and "meta.wcsinfo.siaf_yref_sci" (FITS header
    keywords XREF_SCI and YREF_SCI).

    Because this mode has a single known source location, the utilities used
    in the WFSS modes are overkill. Instead, similar structures are created
    during the extract2d process and then directly used.

    https://jwst-docs.stsci.edu/near-infrared-camera/nircam-observing-modes/nircam-time-series-observations/nircam-grism-time-series
    """

    # Check for reference files
    if not isinstance(reference_files, dict):
        raise TypeError("Expected a dictionary for reference_files")

    # Check for wavelengthrange reference file
    if 'wavelengthrange' not in reference_files:
        raise KeyError("No wavelengthrange reference file specified")

    # If an extraction height is not supplied, default to entire
    # cross-dispersion size of the data array
    if tsgrism_extract_height is None:
        tsgrism_extract_height = input_model.meta.subarray.ysize
    log.info("Setting extraction height to {}".format(tsgrism_extract_height))

    # Get the disperser parameters that have the wave limits
    with WavelengthrangeModel(reference_files['wavelengthrange']) as f:
        if (f.meta.instrument.name != 'NIRCAM' or
                f.meta.exposure.type != 'NRC_TSGRISM'):
            raise ValueError("Wavelengthrange reference file is not for NIRCAM TSGRISM mode!")
        wavelengthrange = f.wavelengthrange
        ref_extract_orders = f.extract_orders

    # If user-supplied spectral orders are not provided,
    # default to extracting only the 1st order
    if extract_orders is None:
        log.info("Using default order extraction from reference file")
        extract_orders = ref_extract_orders
        available_orders = [x[1] for x in extract_orders if
                            x[0] == input_model.meta.instrument.filter].pop()
    else:
        if (not isinstance(extract_orders, list) or
                not all(isinstance(item, int) for item in extract_orders)):
            raise TypeError("Expected extract_orders to be a list of integers.")
        available_orders = extract_orders

    if len(available_orders) > 1:
        raise NotImplementedError("Multiple order extraction for TSO is "
                                  "not currently implemented.")

    # Check for the existence of the aperture reference location meta data
    if (input_model.meta.wcsinfo.siaf_xref_sci is None or
            input_model.meta.wcsinfo.siaf_yref_sci is None):
        raise ValueError('XREF_SCI and YREF_SCI are required for TSO mode.')

    # Create the extracted output as a SlitModel
    log.info("Extracting order: {}".format(available_orders))
    output_model = datamodels.SlitModel()
    output_model.update(input_model)
    subwcs = copy.deepcopy(input_model.meta.wcs)

    # Loop over spectral orders
    for order in available_orders:
        range_select = [(x[2], x[3]) for x in wavelengthrange if
                        (x[0] == order and x[1] == input_model.meta.instrument.filter)]

        # Use the filter that was in front of the grism for translation
        lmin, lmax = range_select.pop()

        # Create the order bounding box
        source_xpos = input_model.meta.wcsinfo.siaf_xref_sci - 1  # remove FITS 1-indexed offset
        source_ypos = input_model.meta.wcsinfo.siaf_yref_sci - 1  # remove FITS 1-indexed offset
        transform = input_model.meta.wcs.get_transform('direct_image', 'grism_detector')
        xmin, ymin, _ = transform(source_xpos,
                                  source_ypos,
                                  lmin,
                                  order)
        xmax, ymax, _ = transform(source_xpos,
                                  source_ypos,
                                  lmax,
                                  order)

        # Add the shift to the lower corner to the subarray WCS object.
        # The shift should just be the lower bounding box corner.
        # Also replace the object center location inputs to the GrismDispersion
        # model with the known object center and order information (in pixels of direct image)
        # This changes the user input to the model from (x,y,x0,y0,order) -> (x,y)
        #
        # The bounding box is limited to the size of the detector in the dispersion direction
        # and 64 pixels in the cross-dispersion direction (at request of instrument team).
        #
        # The team wants the object to fall near row 34 for all cutouts, but the default cutout
        # height is 64 pixels (32 on either side). So bump the extraction ycenter, when necessary,
        # so that the height is 30 above and 34 below (in full frame) the object center.
        bump = source_ypos - 34
        extract_y_center = source_ypos - bump

        splitheight = int(tsgrism_extract_height / 2)
        below = extract_y_center - splitheight
        if below == 34:
            extract_y_min = 0
            extract_y_max = extract_y_center + splitheight
        elif below < 0:
            extract_y_min = 0
            extract_y_max = tsgrism_extract_height - 1
        else:
            extract_y_min = extract_y_center - 34  # always return source at row 34 in cutout
            extract_y_max = extract_y_center + tsgrism_extract_height - 34 - 1

        # Check for bad results
        if extract_y_min > extract_y_max:
            raise ValueError("Something bad happened calculating extraction y-size")

        # Limit the bounding box to the detector edges
        ymin, ymax = (max(extract_y_min, 0), min(extract_y_max, input_model.meta.subarray.ysize))
        xmin, xmax = (max(xmin, 0), min(xmax, input_model.meta.subarray.xsize))

        # The order and source position are put directly into the new WCS of the subarray
        # for the forward transform.
        #
        # NOTE NOTE NOTE  2020-02-14
        # We would normally use x-axis (along dispersion) extraction limits calculated
        # above based on the min/max wavelength range and the source position to do the
        # subarray extraction and set the subarray WCS accordingly. HOWEVER, the NIRCam
        # team has asked for all data along the dispersion direction to be included in
        # subarray cutout, so here we override the xmin/xmax values calculated above and
        # instead hardwire the extraction limits for the x (dispersion) direction to
        # cover the entire range of the data and use this new minimum x value in the
        # subarray WCS transform. If the team ever decides to change the extraction limits,
        # the following two constants must be modified accordingly.
        xmin_ext = 0  # hardwire min x for extraction to zero
        xmax_ext = input_model.data.shape[-1] - 1  # hardwire max x for extraction to size of data

        order_model = Const1D(order)
        order_model.inverse = Const1D(order)
        tr = input_model.meta.wcs.get_transform('grism_detector', 'direct_image')
        tr = Mapping((0, 1, 0)) | Shift(xmin_ext) & Shift(ymin) & order_model | tr
        subwcs.set_transform('grism_detector', 'direct_image', tr)

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)

        log.info("WCS made explicit for order: {}".format(order))
        log.info("Spectral trace extents: (xmin: {}, ymin: {}), "
                 "(xmax: {}, ymax: {})".format(xmin, ymin, xmax, ymax))
        log.info("Extraction limits: (xmin: {}, ymin: {}), "
                 "(xmax: {}, ymax: {})".format(xmin_ext, ymin, xmax, ymax))

        # Cut out the subarray from the input data arrays
        ext_data = input_model.data[..., ymin: ymax + 1, xmin_ext:xmax_ext + 1].copy()
        ext_err = input_model.err[..., ymin: ymax + 1, xmin_ext:xmax_ext + 1].copy()
        ext_dq = input_model.dq[..., ymin: ymax + 1, xmin_ext:xmax_ext + 1].copy()
        if input_model.var_poisson is not None and np.size(input_model.var_poisson) > 0:
            var_poisson = input_model.var_poisson[..., ymin:ymax + 1, xmin_ext:xmax_ext + 1].copy()
        else:
            var_poisson = None
        if input_model.var_rnoise is not None and np.size(input_model.var_rnoise) > 0:
            var_rnoise = input_model.var_rnoise[..., ymin:ymax + 1, xmin_ext:xmax_ext + 1].copy()
        else:
            var_rnoise = None
        if input_model.var_flat is not None and np.size(input_model.var_flat) > 0:
            var_flat = input_model.var_flat[..., ymin:ymax + 1, xmin_ext:xmax_ext + 1].copy()
        else:
            var_flat = None

        # Finish populating the output model and meta data
        if output_model.meta.model_type == "SlitModel":
            output_model.data = ext_data
            output_model.err = ext_err
            output_model.dq = ext_dq
            output_model.var_poisson = var_poisson
            output_model.var_rnoise = var_rnoise
            output_model.var_flat = var_flat
            output_model.meta.wcs = subwcs
            output_model.meta.wcs.bounding_box = util.wcs_bbox_from_shape(ext_data.shape)
            output_model.meta.wcsinfo.siaf_yref_sci = 34  # update for the move, vals are the same
            output_model.meta.wcsinfo.siaf_xref_sci = input_model.meta.wcsinfo.siaf_xref_sci
            output_model.meta.wcsinfo.spectral_order = order
            output_model.meta.wcsinfo.dispersion_direction = \
                input_model.meta.wcsinfo.dispersion_direction
            if compute_wavelength:
                log.debug("Computing wavelengths (this takes a while ...)")
                output_model.wavelength = compute_wavelength_array(output_model)
            output_model.name = '1'
            output_model.source_type = input_model.meta.target.source_type
            output_model.source_name = input_model.meta.target.catalog_name
            output_model.source_alias = input_model.meta.target.proposer_name
            output_model.xstart = 1  # FITS pixels are 1-indexed
            output_model.xsize = ext_data.shape[-1]
            output_model.ystart = ymin + 1  # FITS pixels are 1-indexed
            output_model.ysize = ext_data.shape[-2]
            output_model.source_xpos = source_xpos
            output_model.source_ypos = 34
            output_model.source_id = 1
            output_model.bunit_data = input_model.meta.bunit_data
            output_model.bunit_err = input_model.meta.bunit_err
            if hasattr(input_model, 'int_times'):
                output_model.int_times = input_model.int_times.copy()

    del subwcs
    log.info("Finished extraction")
    return output_model


def extract_grism_objects(input_model,
                          grism_objects=None,
                          reference_files=None,
                          extract_orders=None,
                          mmag_extract=99.,
                          compute_wavelength=True,
                          wfss_extract_half_height=None):
    """
    Extract 2d boxes around each objects spectra for each order.

    Parameters
    ----------
    input_model : `~jwst.datamodels.ImageModel`
        An instance of an ImageModel, this is the grism image

    grism_objects : list(GrismObject)
        A list of GrismObjects

    reference_files : dict
        Needs to include the name of the wavelengthrange reference file

    extract_orders : int
        Spectral orders to extract

    mmag_extract : float
        Sources with magnitudes fainter than this minimum magnitude extraction
        cutoff will not be extracted

    compute_wavelength : bool
        Compute a wavelength array for the datamodel.

    wfss_extract_half_height : int, (optional)
        Cross-dispersion extraction half height in pixels, WFSS mode.
        Overwrites the computed extraction height.

    Returns
    -------
    output_model : `~jwst.datamodels.MultiSlitModel`


    Notes
    -----
    This method supports NRC_WFSS and NIS_WFSS only

    GrismObject is a named tuple which contains distilled
    information about each catalog object. It can be created
    by calling jwst.assign_wcs.util.create_grism_bbox() which
    will return a list of GrismObjects that countain the bounding
    boxes that will be used to define the 2d extraction area.

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
    grism reference file as wavelengthrange.

    Step 1: Convert the source catalog from the reference frame of the
            uberimage to that of the dispersed image.  For the Vanilla
            Pipeline we assume that the pointing information in the file
            headers is sufficient.  This will be strictly true if all images
            were obtained in a single visit (same guide stars).

    Step 2: Record source information for each object in the catalog: position
            (RA and Dec), shape (A_IMAGE, B_IMAGE, THETA_IMAGE), and all
            available magnitudes, and minimum bounding boxes

    Step 3: Compute the trace and wavelength solutions for each object in the
            catalog and for each spectral order.  Record this information.

    Step 4: Compute the WIDTH of each spectral subwindow, which may be fixed or
            variable. The cross-dispersion size is taken from the minimum
            bounding box.
    """
    if reference_files is None or not reference_files:
        raise TypeError("Expected a dictionary for reference_files")

    if grism_objects is None:
        # get the wavelengthrange reference file from the input_model
        if ('wavelengthrange' not in reference_files or reference_files['wavelengthrange'] in ['N/A', '']):
            raise ValueError("Expected name of wavelengthrange reference file")
        else:
            grism_objects = util.create_grism_bbox(input_model, reference_files,
                                                   extract_orders=extract_orders,
                                                   mmag_extract=mmag_extract,
                                                   wfss_extract_half_height=wfss_extract_half_height)
            log.info("Grism object list created from source catalog: {0:s}"
                     .format(input_model.meta.source_catalog))

    if not isinstance(grism_objects, list):
        raise TypeError("Expected input grism objects to be a list")
    if len(grism_objects) == 0:
        raise ValueError("No grism objects created from source catalog")

    log.info("Extracting grism objects into MultiSlitModel")
    output_model = datamodels.MultiSlitModel()
    output_model.update(input_model)

    # One WCS model can be used to govern all the extractions
    # and in fact the model transforms rely on the full frame
    # coordinates of the input pixel location. So the WCS
    # attached to the extraction is just a copy of the
    # input_model WCS with a shift transform to the corner
    # of the subarray. They also depend on the source object
    # center, this information will be saved to the meta of
    # the output model as source_[x/y]pos
    inwcs = input_model.meta.wcs

    # For easy reference here, GrismObjects has:
    #
    # xcenter,ycenter: in direct image pixels
    # order_bounding in grism_detector pixels
    # sky_centroid: SkyCoord of object center
    # sky_bbox_ :lower and upper bounding box in SkyCoord
    # sid: catalog ID of the object

    slits = []
    for obj in grism_objects:
        for order in obj.order_bounding.keys():

            # Add the shift to the lower corner to each subarray WCS object
            # The shift should just be the lower bounding box corner
            # also replace the object center location inputs to the GrismDispersion
            # model with the known object center and order information (in pixels of direct image)
            # This is changes the user input to the model from (x,y,x0,y0,order) -> (x,y)
            #
            # The bounding boxes here are also limited to the size of the detector
            # The check for boxes entirely off the detector is done in create_grism_bbox right now
            y, x = obj.order_bounding[order]

            # limit the boxes to the detector
            ymin = clamp(y[0], 0, input_model.meta.subarray.ysize)
            ymax = clamp(y[1], 0, input_model.meta.subarray.ysize)
            xmin = clamp(x[0], 0, input_model.meta.subarray.xsize)
            xmax = clamp(x[1], 0, input_model.meta.subarray.xsize)

            # don't extract anything that ended up with zero dimensions in one axis
            # this means that it was identified as a partial order but only on one
            # row or column of the detector
            if (((ymax - ymin) > 0) and ((xmax - xmin) > 0)):
                subwcs = copy.deepcopy(inwcs)
                log.info("Subarray extracted for obj: {} order: {}:".format(obj.sid, order))
                log.info("Subarray extents are: "
                         "(xmin:{}, xmax:{}), (ymin:{}, ymax:{})".format(xmin, xmax, ymin, ymax))

                # only the first two numbers in the Mapping are used
                # the order and source position are put directly into
                # the new wcs for the subarray for the forward transform
                xcenter_model = Const1D(obj.xcentroid)
                xcenter_model.inverse = Const1D(obj.xcentroid)

                ycenter_model = Const1D(obj.ycentroid)
                ycenter_model.inverse = Const1D(obj.ycentroid)

                order_model = Const1D(order)
                order_model.inverse = Const1D(order)

                tr = inwcs.get_transform('grism_detector', 'detector')

                tr = Mapping((0, 1, 0, 0, 0)) | (Shift(xmin) & Shift(ymin) &
                                                 xcenter_model &
                                                 ycenter_model &
                                                 order_model) | tr

                ext_data = input_model.data[ymin: ymax + 1, xmin: xmax + 1].copy()
                ext_err = input_model.err[ymin: ymax + 1, xmin: xmax + 1].copy()
                ext_dq = input_model.dq[ymin: ymax + 1, xmin: xmax + 1].copy()
                if input_model.var_poisson is not None and np.size(input_model.var_poisson) > 0:
                    var_poisson = input_model.var_poisson[ymin:ymax + 1, xmin:xmax + 1].copy()
                else:
                    var_poisson = None
                if input_model.var_rnoise is not None and np.size(input_model.var_rnoise) > 0:
                    var_rnoise = input_model.var_rnoise[ymin:ymax + 1, xmin:xmax + 1].copy()
                else:
                    var_rnoise = None
                if input_model.var_flat is not None and np.size(input_model.var_flat) > 0:
                    var_flat = input_model.var_flat[ymin:ymax + 1, xmin:xmax + 1].copy()
                else:
                    var_flat = None

                tr.bounding_box = util.transform_bbox_from_shape(ext_data.shape)
                subwcs.set_transform('grism_detector', 'detector', tr)

                new_slit = datamodels.SlitModel(data=ext_data,
                                                err=ext_err,
                                                dq=ext_dq,
                                                var_poisson=var_poisson,
                                                var_rnoise=var_rnoise,
                                                var_flat=var_flat)
                new_slit.meta.wcsinfo.spectral_order = order
                new_slit.meta.wcsinfo.dispersion_direction = \
                    input_model.meta.wcsinfo.dispersion_direction
                new_slit.meta.wcsinfo.specsys = input_model.meta.wcsinfo.specsys
                new_slit.meta.coordinates = input_model.meta.coordinates
                new_slit.meta.wcs = subwcs
                if compute_wavelength:
                    log.debug("Computing wavelengths")
                    new_slit.wavelength = compute_wavelength_array(new_slit)

                # set x/ystart values relative to the image (screen) frame.
                # The overall subarray offset is recorded in model.meta.subarray.
                # nslit = obj.sid - 1  # catalog id starts at zero
                new_slit.name = "{0}".format(obj.sid)
                new_slit.is_star = obj.is_star
                new_slit.xstart = xmin + 1  # fits pixels
                new_slit.xsize = ext_data.shape[1]
                new_slit.ystart = ymin + 1  # fits pixels
                new_slit.ysize = ext_data.shape[0]
                new_slit.source_xpos = float(obj.xcentroid)
                new_slit.source_ypos = float(obj.ycentroid)
                new_slit.source_id = obj.sid
                new_slit.bunit_data = input_model.meta.bunit_data
                new_slit.bunit_err = input_model.meta.bunit_err
                slits.append(new_slit)
    output_model.slits.extend(slits)
    # In the case that there are no spectra to extract deleting the variables
    # will fail so add the try block.
    try:
        del subwcs
    except UnboundLocalError:
        pass
    try:
        del new_slit
    except UnboundLocalError:
        pass
    # del subwcs
    # del new_slit
    log.info("Finished extractions")
    return output_model


def clamp(value, minval, maxval):
    """
    Return the value clipped between minval and maxval.

    Parameters
    ----------
    value : float
        The value to limit
    minval : float
        The minimal acceptable value
    maxval : float
        The maximum acceptable value

    Returns
    -------
    value: float
        The value that falls within the min-max range or the minimum limit
    """
    return max(minval, min(value, maxval))


def compute_dispersion(wcs):
    """
    Compute the pixel dispersion.

    Make a model for the pixel dispersion from the grismconf specs

    Parameters
    ----------
    wcs : `~gwcs.wcs.WCS`
        The WCS object for this slit.

    Returns
    -------
    dispersion : ndarray-like
        The pixel dispersion [in m].

    """
    raise NotImplementedError


def compute_wavelength_array(slit):
    """
    Compute the wavelength array for a slit with gwcs object

    Parameters
    ----------
    slit : `~jwst.datamodels.SlitModel`
        JWST slit datamodel containing a meta.wcs GWCS object

    Returns
    -------
    wavelength : numpy.array
        The wavelength array
    """
    transform = slit.meta.wcs.forward_transform
    x, y = grid_from_bounding_box(slit.meta.wcs.bounding_box)
    wavelength = transform(x, y)[2]
    return wavelength
