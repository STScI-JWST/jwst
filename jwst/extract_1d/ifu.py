import json
import logging
import math

import numpy as np
from photutils import CircularAperture, CircularAnnulus, \
                      RectangularAperture, aperture_photometry

from .. import datamodels
from .. datamodels import dqflags
from . import spec_wcs

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def ifu_extract1d(input_model, refname, source_type):
    """Extract a 1-D spectrum from an IFU cube.

    Parameters
    ----------
    input_model: IFUCubeModel
        The input model.

    refname: string
        The name of the JSON reference file.

    source_type: string
        "point" or "extended"

    Returns
    -------
    output_model: MultiSpecModel
        This will contain the extracted spectrum.
    """

    if not isinstance(input_model, datamodels.IFUCubeModel):
        log.error("Expected an IFU cube.")
        raise RuntimeError("Expected an IFU cube.")

    if source_type != "point" and source_type != "extended":
        log.warning("source_type was '%s', setting to 'point'.", source_type)
        source_type = "point"
    log.info("source_type = %s", source_type)

    output_model = datamodels.MultiSpecModel()
    output_model.update(input_model)

    slitname = input_model.meta.exposure.type
    if slitname is None:
        slitname = "ANY"

    extract_params = ifu_extract_parameters(refname, slitname, source_type)

    if extract_params:
        (ra, dec, wavelength, net, background, dq) = extract_ifu(
                        input_model, source_type, extract_params)
    else:
        log.critical('Missing extraction parameters.')
        raise ValueError('Missing extraction parameters.')

    # Check whether the data have been converted to flux density.
    fluxcorr_complete = True                    # initial values
    missing = False
    try:
        bunit = input_model.meta.bunit_data
    except AttributeError:
        bunit = None
    if bunit is None:
        fluxcorr_complete = False
        missing = True
    elif (bunit.find("Jy") < 0 and
          bunit.find("jansky") < 0 and bunit.find("Jansky") < 0):
        fluxcorr_complete = False
    if missing:
        log.warning("No BUNIT found in input science data header.")

    # If the data have already been converted to flux density, `net`
    # contains fluxes, so move that column to `flux`.  If not, it's
    # too late to do flux correction.
    if fluxcorr_complete:
        flux = net.copy()
        net[:] = 0.
        log.info("Data have been flux calibrated; setting net to 0.")
    else:
        flux = np.zeros_like(net)
        log.info("Data have NOT been flux calibrated; setting flux to 0.")

    fl_error = np.ones_like(net)
    nerror = np.ones_like(net)
    berror = np.ones_like(net)
    spec = datamodels.SpecModel()
    otab = np.array(list(zip(wavelength, flux, fl_error, dq,
                         net, nerror, background, berror)),
                    dtype=spec.spec_table.dtype)
    spec = datamodels.SpecModel(spec_table=otab)
    spec.meta.wcs = spec_wcs.create_spectral_wcs(ra, dec, wavelength)
    spec.slit_ra = ra
    spec.slit_dec = dec
    if slitname is not None and slitname != "ANY":
        spec.name = slitname
    output_model.spec.append(spec)

    # See output_model.spec[0].meta.wcs instead.
    output_model.meta.wcs = None

    return output_model


def ifu_extract_parameters(refname, slitname, source_type):
    """Read extraction parameters for an IFU."""

    extract_params = {}
    with open(refname) as f:
        ref = json.load(f)
    for aper in ref['apertures']:
        if 'id' in aper and aper['id'] != "dummy" and \
           (aper['id'] == slitname or aper['id'] == "ANY" or
            slitname == "ANY"):
            region_type = aper.get("region_type", "target")
            if region_type == "target":
                extract_params['x_center'] = aper.get('x_center')
                extract_params['y_center'] = aper.get('y_center')
                extract_params['method'] = aper.get('method', 'exact')
                extract_params['subpixels'] = aper.get('subpixels', 5)
                extract_params['radius'] = aper.get('radius')
                extract_params['subtract_background'] = \
                      aper.get('subtract_background', False)
                extract_params['inner_bkg'] = aper.get('inner_bkg')
                extract_params['outer_bkg'] = aper.get('outer_bkg')
                extract_params['width'] = aper.get('width')
                extract_params['height'] = aper.get('height')
                # theta is in degrees (converted to radians later)
                extract_params['theta'] = aper.get('theta', 0.)
            break

    return extract_params


def extract_ifu(input_model, source_type, extract_params):
    """This function does the extraction.

    Parameters
    ----------
    input_model: IFUCubeModel
        The input model.

    source_type: string
        "point" or "extended"

    extract_params: dict
        The extraction parameters for aperture photometry.

    Returns
    -------
        (ra, dec, wavelength, net, background, dq)
    """

    data = input_model.data
    shape = data.shape
    if len(shape) != 3:
        log.error("Expected a 3-D IFU cube; dimension is %d.", len(shape))
        raise RuntimeError("The IFU cube should be 3-D.")

    # We need to allocate net, background, and dq arrays no matter what.
    net = np.zeros(shape[0], dtype=np.float64)
    background = np.zeros(shape[0], dtype=np.float64)
    dq = np.zeros(shape[0], dtype=np.int32)

    x_center = extract_params['x_center']
    y_center = extract_params['y_center']
    if x_center is None:
        x_center = float(shape[2]) / 2.
    else:
        x_center = float(x_center)
    if y_center is None:
        y_center = float(shape[1]) / 2.
    else:
        y_center = float(y_center)

    method = extract_params['method']
    # subpixels is only needed if method = 'subpixel'.
    subpixels = extract_params['subpixels']

    subtract_background = extract_params['subtract_background']
    smaller_axis = float(min(shape[1], shape[2]))       # for defaults
    if source_type == 'point':
        radius = extract_params['radius']
        if radius is None:
            radius = smaller_axis / 4.
        if subtract_background:
            inner_bkg = extract_params['inner_bkg']
            if inner_bkg is None:
                inner_bkg = radius
            outer_bkg = extract_params['outer_bkg']
            if outer_bkg is None:
                outer_bkg = min(inner_bkg * math.sqrt(2.),
                                smaller_axis / 2. - 1.)
            if inner_bkg <= 0. or outer_bkg <= 0. or inner_bkg >= outer_bkg:
                log.debug("Turning background subtraction off, due to "
                          "the values of inner_bkg and outer_bkg.")
                subtract_background = False
        width = None
        height = None
        theta = None
    else:
        width = extract_params['width']
        if width is None:
            width = smaller_axis / 2.
        height = extract_params['height']
        if height is None:
            height = smaller_axis / 2.
        theta = extract_params['theta'] * math.pi / 180.
        radius = None
        subtract_background = False
        inner_bkg = None
        outer_bkg = None

    log.debug("IFU 1-D extraction parameters:")
    log.debug("  x_center = %s", str(x_center))
    log.debug("  y_center = %s", str(y_center))
    if source_type == 'point':
        log.debug("  radius = %s", str(radius))
        log.debug("  subtract_background = %s", str(subtract_background))
        log.debug("  inner_bkg = %s", str(inner_bkg))
        log.debug("  outer_bkg = %s", str(outer_bkg))
        log.debug("  method = %s", method)
        if method == "subpixel":
            log.debug("  subpixels = %s", str(subpixels))
    else:
        log.debug("  width = %s", str(width))
        log.debug("  height = %s", str(height))
        log.debug("  theta = %s degrees", str(extract_params['theta']))
        log.debug("  subtract_background = %s", str(subtract_background))
        log.debug("  method = %s", method)
        if method == "subpixel":
            log.debug("  subpixels = %s", str(subpixels))

    # Check for out of bounds.
    # The problem with having the background aperture extend beyond the
    # image is that the normalization would not account for the resulting
    # decrease in the area of the annulus, so the background subtraction
    # would be systematically low.
    outside = False
    f_nx = float(shape[2])
    f_ny = float(shape[1])
    if x_center < 0. or x_center >= f_nx - 1. or \
       y_center < 0. or y_center >= f_ny - 1.:
        outside = True
        log.error("Target location is outside the image.")
    if subtract_background and \
       (x_center - outer_bkg < -0.5 or x_center + outer_bkg > f_nx - 0.5 or
        y_center - outer_bkg < -0.5 or y_center + outer_bkg > f_ny - 0.5):
            outside = True
            log.error("Background region extends outside the image.")

    if outside:
        (ra, dec) = (0., 0.)
        wavelength = np.zeros(shape[0], dtype=np.float64)
        dq[:] = dqflags.pixel['DO_NOT_USE']
        return (ra, dec, wavelength, net, background, dq)       # all bad

    if hasattr(input_model.meta, 'wcs'):
        wcs = input_model.meta.wcs
    else:
        log.warning("WCS function not found in input.")
        wcs = None

    if wcs is not None:
        x_array = np.empty(shape[0], dtype=np.float64)
        x_array.fill(float(shape[2]) / 2.)
        y_array = np.empty(shape[0], dtype=np.float64)
        y_array.fill(float(shape[1]) / 2.)
        z_array = np.arange(shape[0], dtype=np.float64) # for wavelengths
        if input_model.meta.instrument.name == "NIRSPEC":
            # xxx temporary:  NIRSpec wcs expects one-based pixel coordinates.
            ra, dec, wavelength = wcs(x_array + 1., y_array + 1., z_array + 1.)
        else:
            ra, dec, wavelength = wcs(x_array, y_array, z_array)
        nelem = len(wavelength)
        ra = ra[nelem // 2]
        dec = dec[nelem // 2]
    else:
        (ra, dec) = (0., 0.)
        wavelength = np.arange(1, shape[0] + 1, dtype=np.float64)

    position = (x_center, y_center)
    if source_type == 'point':
        aperture = CircularAperture(position, r=radius)
        if subtract_background:
            annulus = CircularAnnulus(position,
                                      r_in=inner_bkg, r_out=outer_bkg)
            normalization = aperture.area() / annulus.area()
    else:
        aperture = RectangularAperture(position, width, height, theta)
        # No background is computed for an extended source.

    for k in range(shape[0]):
        phot_table = aperture_photometry(data[k, :, :], aperture,
                                         method=method, subpixels=subpixels)
        net[k] = float(phot_table['aperture_sum'][0])
        if subtract_background:
            bkg_table = aperture_photometry(data[k, :, :], annulus,
                                            method=method, subpixels=subpixels)
            background[k] = float(bkg_table['aperture_sum'][0])
            net[k] = net[k] - background[k] * normalization

    return (ra, dec, wavelength, net, background, dq)
