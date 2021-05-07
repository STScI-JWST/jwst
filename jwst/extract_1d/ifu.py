import logging
import numpy as np
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                RectangularAperture, aperture_photometry)

from .apply_apcorr import select_apcorr
from ..assign_wcs.util import compute_scale
from astropy import stats
from .. import datamodels
from ..datamodels import dqflags

from . import spec_wcs
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# These values are used to indicate whether the input extract1d reference file
# (if any) is ASDF (default) or IMAGE

FILE_TYPE_ASDF = "ASDF"
FILE_TYPE_IMAGE = "IMAGE"

# This is to prevent calling offset_from_offset multiple times for
# multi-integration data.
OFFSET_NOT_ASSIGNED_YET = "not assigned yet"

# This is intended to be larger than any possible distance (in pixels)
# between the target and any point in the image; used by locn_from_wcs().
HUGE_DIST = 1.e10


def ifu_extract1d(input_model, ref_dict, source_type, subtract_background,
                  bkg_sigma_clip, apcorr_ref_model=None):
    """Extract a 1-D spectrum from an IFU cube.

    Parameters
    ----------
    input_model : JWST data model for an IFU cube (IFUCubeModel)
        The input model.

    ref_dict : dict
        The contents of the extract1d reference file.

    source_type : string
        "POINT" or "EXTENDED"

    subtract_background : bool or None
        User supplied flag indicating whether the background should be subtracted.
        If None, the value in the extract_1d reference file will be used.
        If not None, this parameter overrides the value in the
        extract_1d reference file.

    bkg_sigma_clip : float
        Background sigma clipping value to use to remove noise/outliers in background

    apcorr_ref_model : apcorr datamodel or None
        Aperture correction table.

    Returns
    -------
    output_model : MultiSpecModel
        This will contain the extracted spectrum.
    """

    if not isinstance(input_model, datamodels.IFUCubeModel):
        log.error("Expected an IFU cube.")
        raise RuntimeError("Expected an IFU cube.")

    instrument = input_model.meta.instrument.name
    if instrument is not None:
        instrument = instrument.upper()
    if source_type is not None:
        source_type = source_type.upper()
    if source_type != 'POINT' and source_type != 'EXTENDED':
        default_source_type = 'EXTENDED'
        log.warning(f"Source type was '{source_type}'; setting to '{default_source_type}'.")
        source_type = default_source_type
    else:
        log.info(f"Source type = {source_type}")

    # The input units will normally be MJy / sr, but for NIRSpec point-source
    # spectra the units will be MJy.
    input_units_are_megajanskys = (source_type == 'POINT' and instrument == 'NIRSPEC')

    output_model = datamodels.MultiSpecModel()
    output_model.update(input_model, only="PRIMARY")

    slitname = input_model.meta.exposure.type
    if slitname is None:
        slitname = "ANY"

    extract_params = get_extract_parameters(ref_dict, bkg_sigma_clip, slitname)

    if subtract_background is not None:
        if subtract_background and source_type == "EXTENDED":
            subtract_background = False
            log.info("Turning off background subtraction because "
                     "the source is extended.")
        extract_params['subtract_background'] = subtract_background

    if extract_params:
        if extract_params['ref_file_type'] == FILE_TYPE_ASDF:
            (ra, dec, wavelength, temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
             background, b_var_poisson, b_var_rnoise, b_var_flat, npixels, dq, npixels_bkg,
             radius_match, x_center, y_center) = \
                extract_ifu(input_model, source_type, extract_params)
        else:                                   # FILE_TYPE_IMAGE
            (ra, dec, wavelength, temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
             background, b_var_poisson, b_var_rnoise, b_var_flat, npixels, dq, npixels_bkg,
             x_center, y_center) = \
                image_extract_ifu(input_model, source_type, extract_params)
    else:
        log.critical('Missing extraction parameters.')
        raise ValueError('Missing extraction parameters.')

    npixels_temp = np.where(npixels > 0., npixels, 1.)
    npixels_bkg_temp = np.where(npixels_bkg > 0., npixels_bkg, 1.)

    # Convert the sum to an average, for surface brightness.
    surf_bright = temp_flux / npixels_temp
    sb_var_poisson = f_var_poisson / npixels_temp
    sb_var_rnoise = f_var_rnoise / npixels_temp
    sb_var_flat = f_var_flat / npixels_temp
    background /= npixels_bkg_temp
    b_var_poisson /= npixels_bkg_temp
    b_var_rnoise /= npixels_bkg_temp
    b_var_flat /= npixels_bkg_temp

    del npixels_temp
    del npixels_bkg_temp

    pixel_solid_angle = input_model.meta.photometry.pixelarea_steradians
    if pixel_solid_angle is None:
        log.warning("Pixel area (solid angle) is not populated; "
                    "the flux will not be correct.")
        pixel_solid_angle = 1.

    if input_units_are_megajanskys:
        # Convert flux from MJy to Jy, and convert background to MJy / sr.
        flux = temp_flux * 1.e6
        f_var_poisson *= 1.e12  # MJy**2 --> Jy**2
        f_var_rnoise *= 1.e12  # MJy**2 --> Jy**2
        f_var_flat *= 1.e12  # MJy**2 --> Jy**2
        surf_bright[:] = 0.
        sb_var_poisson = 0.
        sb_var_rnoise = 0.
        sb_var_flat = 0.
        background[:] /= pixel_solid_angle  # MJy / sr
        b_var_poisson /= pixel_solid_angle
        b_var_rnoise /= pixel_solid_angle
        b_var_flat /= pixel_solid_angle

    else:
        # Convert flux from MJy / steradian to Jy.
        flux = temp_flux * pixel_solid_angle * 1.e6
        f_var_poisson *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
        f_var_rnoise *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
        f_var_flat *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
        # surf_bright and background were computed above
    del temp_flux
    error = np.sqrt(f_var_poisson + f_var_rnoise + f_var_flat)
    sb_error = np.sqrt(sb_var_poisson + sb_var_rnoise + sb_var_flat)
    berror = np.sqrt(b_var_poisson + b_var_rnoise + b_var_flat)
    spec_dtype = datamodels.SpecModel().spec_table.dtype

    otab = np.array(
        list(
            zip(wavelength, flux, error, f_var_poisson, f_var_rnoise, f_var_flat,
                surf_bright, sb_error, sb_var_poisson, sb_var_rnoise, sb_var_flat,
                dq, background, berror, b_var_poisson, b_var_rnoise, b_var_flat, npixels)
        ),
        dtype=spec_dtype
    )

    spec = datamodels.SpecModel(spec_table=otab)
    spec.meta.wcs = spec_wcs.create_spectral_wcs(ra, dec, wavelength)
    spec.spec_table.columns['wavelength'].unit = 'um'
    spec.spec_table.columns['flux'].unit = "Jy"
    spec.spec_table.columns['flux_error'].unit = "Jy"
    spec.spec_table.columns['flux_var_poisson'].unit = "Jy^2"
    spec.spec_table.columns['flux_var_rnoise'].unit = "Jy^2"
    spec.spec_table.columns['flux_var_flat'].unit = "Jy^2"
    spec.spec_table.columns['surf_bright'].unit = "MJy/sr"
    spec.spec_table.columns['sb_error'].unit = "MJy/sr"
    spec.spec_table.columns['sb_var_poisson'].unit = "(MJy/sr)^2"
    spec.spec_table.columns['sb_var_rnoise'].unit = "(MJy/sr)^2"
    spec.spec_table.columns['sb_var_flat'].unit = "(MJy/sr)^2"
    spec.spec_table.columns['background'].unit = "MJy/sr"
    spec.spec_table.columns['bkgd_error'].unit = "MJy/sr"
    spec.spec_table.columns['bkgd_var_poisson'].unit = "(MJy/sr)^2"
    spec.spec_table.columns['bkgd_var_rnoise'].unit = "(MJy/sr)^2"
    spec.spec_table.columns['bkgd_var_flat'].unit = "(MJy/sr)^2"
    spec.slit_ra = ra
    spec.slit_dec = dec
    if slitname is not None and slitname != "ANY":
        spec.name = slitname

    spec.source_type = source_type
    spec.extraction_x = x_center
    spec.extraction_y = y_center

    if source_type == 'POINT' and apcorr_ref_model is not None:
        log.info('Applying Aperture correction.')

        if instrument == 'NIRSPEC':
            wl = np.median(wavelength)
        else:
            wl = wavelength.min()

        apcorr = select_apcorr(input_model)(
            input_model,
            apcorr_ref_model,
            location=(ra, dec, wl)
        )

        # determine apcor function and apcor radius to use at each wavelength
        apcorr.match_wavelengths(wavelength)

        # at each IFU wavelength we have the extraction radius defined by radius_match (radius size in pixels)
        for i in range(wavelength.size):
            radius = radius_match[i]
            apcorr.find_apcorr_func(i, radius)

        apcorr.apply(spec.spec_table)

    output_model.spec.append(spec)

    # See output_model.spec[0].meta.wcs instead.
    output_model.meta.wcs = None

    return output_model


def get_extract_parameters(ref_dict, bkg_sigma_clip, slitname):
    """Read extraction parameters for an IFU.

    Parameters
    ----------
    ref_dict : dict
        The contents of the extract1d reference file.

    slitname : str
        The name of the slit, or "ANY".

    Returns
    -------
    dict
        The extraction parameters.
    """

    extract_params = {}
    # for consistency put the bkg_sigma_clip in dictionary: extract_params
    extract_params['bkg_sigma_clip'] = bkg_sigma_clip

    if ref_dict['ref_file_type'] == FILE_TYPE_ASDF:
        extract_params['ref_file_type'] = FILE_TYPE_ASDF
        refmodel = ref_dict['ref_model']
        subtract_background = refmodel.meta.subtract_background
        method = refmodel.meta.method
        subpixels = refmodel.meta.subpixels

        data = refmodel.data
        wavelength = data.wavelength
        radius = data.radius
        inner_bkg = data.inner_bkg
        outer_bkg = data.outer_bkg

        extract_params['subtract_background'] = bool(subtract_background)
        extract_params['method'] = method
        extract_params['subpixels'] = subpixels
        extract_params['wavelength'] = wavelength
        extract_params['radius'] = radius
        extract_params['inner_bkg'] = inner_bkg
        extract_params['outer_bkg'] = outer_bkg

    elif ref_dict['ref_file_type'] == FILE_TYPE_IMAGE:
        extract_params['ref_file_type'] = FILE_TYPE_IMAGE
        foundit = False
        for im in ref_dict['ref_model'].images:
            if (im.name is None or im.name == "ANY" or slitname == "ANY" or im.name == slitname):
                extract_params['ref_image'] = im
                foundit = True
                break

        if not foundit:
            log.error("No match for slit name %s in reference image", slitname)
            raise RuntimeError("Specify slit name or use 'any' in ref image.")

    else:
        log.error("Reference file type %s not recognized",
                  ref_dict['ref_file_type'])
        raise RuntimeError("extract1d reference file must be JSON or a FITS image.")

    return extract_params


def extract_ifu(input_model, source_type, extract_params):
    """This function does the extraction.

    Parameters
    ----------
    input_model : IFUCubeModel
        The input model.

    source_type : string
        "POINT" or "EXTENDED"

    extract_params : dict
        The extraction parameters for aperture photometry.

    Returns
    -------
    ra, dec : float
        ra and dec are the right ascension and declination respectively
        at the nominal center of the image.

    wavelength : ndarray, 1-D
        The wavelength in micrometers at each plane of the IFU cube.

    temp_flux : ndarray, 1-D
        The sum of the data values in the extraction aperture minus the
        sum of the data values in the background region (scaled by the
        ratio of areas), for each plane.
        The data values are in units of surface brightness, so this value
        isn't really the flux, it's an intermediate value.  Dividing by
        `npixels` (to compute the average) will give the value for the
        `surf_bright` (surface brightness) column, and multiplying by
        the solid angle of a pixel will give the flux for a point source.

    f_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        temp_flux array.

    f_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        temp_flux array.

    f_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        temp_flux array.

    background : ndarray, 1-D
        For point source data, the background array is the count rate that was subtracted
        from the total source data values to get `temp_flux`. This background is determined
        for annulus region. For extended source data, the background array is the sigma clipped
        extracted region.

    b_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        background array.

    b_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        background array.

    b_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        background array.

    npixels : ndarray, 1-D, float64
        For each slice, this is the number of pixels that were added
        together to get `temp_flux`.

    dq : ndarray, 1-D, uint32
        The data quality array.

    npixels_bkg : ndarray, 1-D, float64
        For each slice, for point source data  this is the number of pixels that were added
        together to get `temp_flux` for an annulus region or for extended source
        data it is the number of pixels used to determine the background

    radius_match : ndarray,1-D, float64
        The size of the extract radius in pixels used at each wavelength of the IFU cube

    x_center, y_center : float
        The x and y center of the extraction region
    """

    data = input_model.data
    var_poisson = input_model.var_poisson
    var_rnoise = input_model.var_rnoise
    var_flat = input_model.var_flat
    weightmap = input_model.weightmap

    shape = data.shape
    if len(shape) != 3:
        log.error("Expected a 3-D IFU cube; dimension is %d.", len(shape))
        raise RuntimeError("The IFU cube should be 3-D.")

    # We need to allocate temp_flux, background, npixels, and dq arrays
    # no matter what.  We may need to divide by npixels, so the default
    # is 1 rather than 0.
    temp_flux = np.zeros(shape[0], dtype=np.float64)
    background = np.zeros(shape[0], dtype=np.float64)
    npixels = np.ones(shape[0], dtype=np.float64)
    npixels_bkg = np.ones(shape[0], dtype=np.float64)
    f_var_poisson = np.zeros(shape[0], dtype=np.float64)
    f_var_rnoise = np.zeros(shape[0], dtype=np.float64)
    f_var_flat = np.zeros(shape[0], dtype=np.float64)
    b_var_poisson = np.zeros(shape[0], dtype=np.float64)
    b_var_rnoise = np.zeros(shape[0], dtype=np.float64)
    b_var_flat = np.zeros(shape[0], dtype=np.float64)

    dq = np.zeros(shape[0], dtype=np.uint32)

    # For an extended target, the entire aperture will be extracted, so
    # it makes no sense to shift the extraction location.
    if source_type != "EXTENDED":
        ra_targ = input_model.meta.target.ra
        dec_targ = input_model.meta.target.dec
        locn = locn_from_wcs(input_model, ra_targ, dec_targ)

        if locn is None or np.isnan(locn[0]):
            log.warning("Couldn't determine pixel location from WCS, so "
                        "source offset correction will not be applied.")

            x_center = float(shape[-1]) / 2.
            y_center = float(shape[-2]) / 2.

        else:
            (x_center, y_center) = locn
            log.info("Using x_center = %g, y_center = %g, based on "
                     "TARG_RA and TARG_DEC.", x_center, y_center)

    method = extract_params['method']
    subpixels = extract_params['subpixels']
    subtract_background = extract_params['subtract_background']

    radius = None
    inner_bkg = None
    outer_bkg = None
    width = None
    height = None
    theta = None
    # pull wavelength plane out of input data.
    # using extract 1d wavelength, interpolate the radius, inner_bkg, outer_bkg to match input wavelength

    # find the wavelength array of the IFU cube
    x0 = float(shape[2]) / 2.
    y0 = float(shape[1]) / 2.
    (ra, dec, wavelength) = get_coordinates(input_model, x0, y0)

    # interpolate the extraction parameters to the wavelength of the IFU cube
    radius_match = None
    if source_type == 'POINT':
        wave_extract = extract_params['wavelength'].flatten()
        inner_bkg = extract_params['inner_bkg'].flatten()
        outer_bkg = extract_params['outer_bkg'].flatten()
        radius = extract_params['radius'].flatten()

        frad = interp1d(wave_extract, radius, bounds_error=False, fill_value="extrapolate")
        radius_match = frad(wavelength)
        # radius_match is in arc seconds - need to convert to pixels
        # the spatial scale is the same for all wavelengths do we only need to call compute_scale once.

        if locn is None:
            locn_use = (input_model.meta.wcsinfo.crval1, input_model.meta.wcsinfo.crval2, wavelength[0])
        else:
            locn_use = (ra_targ, dec_targ, wavelength[0])

        scale_degrees = compute_scale(
            input_model.meta.wcs,
            locn_use,
            disp_axis=input_model.meta.wcsinfo.dispersion_direction)

        scale_arcsec = scale_degrees * 3600.00
        radius_match /= scale_arcsec

        finner = interp1d(wave_extract, inner_bkg, bounds_error=False, fill_value="extrapolate")
        inner_bkg_match = finner(wavelength) / scale_arcsec

        fouter = interp1d(wave_extract, outer_bkg, bounds_error=False, fill_value="extrapolate")
        outer_bkg_match = fouter(wavelength) / scale_arcsec

    elif source_type == 'EXTENDED':
        # Ignore any input parameters, and extract the whole image.
        width = float(shape[-1])
        height = float(shape[-2])
        x_center = width / 2. - 0.5
        y_center = height / 2. - 0.5
        theta = 0.
        subtract_background = False
        bkg_sigma_clip = extract_params['bkg_sigma_clip']

    log.debug("IFU 1-D extraction parameters:")
    log.debug("  x_center = %s", str(x_center))
    log.debug("  y_center = %s", str(y_center))
    if source_type == 'POINT':
        log.debug("  method = %s", method)
        if method == "subpixel":
            log.debug("  subpixels = %s", str(subpixels))
    else:
        log.debug("  width = %s", str(width))
        log.debug("  height = %s", str(height))
        log.debug("  theta = %s degrees", str(theta))
        log.debug("  subtract_background = %s", str(subtract_background))
        log.debug("  sigma clip value for background = %s", str(bkg_sigma_clip))
        log.debug("  method = %s", method)
        if method == "subpixel":
            log.debug("  subpixels = %s", str(subpixels))

    position = (x_center, y_center)

    # get aperture for extended it will not change with wavelength
    if source_type == 'EXTENDED':
        aperture = RectangularAperture(position, width, height, theta)
        annulus = None

    for k in range(shape[0]):  # looping over wavelength
        inner_bkg = None
        outer_bkg = None

        if source_type == 'POINT':
            radius = radius_match[k]  # this radius has been converted to pixels
            aperture = CircularAperture(position, r=radius)
            inner_bkg = inner_bkg_match[k]
            outer_bkg = outer_bkg_match[k]
            if inner_bkg <= 0. or outer_bkg <= 0. or inner_bkg >= outer_bkg:
                log.debug("Turning background subtraction off, due to "
                          "the values of inner_bkg and outer_bkg.")
                subtract_background = False

        if subtract_background and inner_bkg is not None and outer_bkg is not None:
            annulus = CircularAnnulus(position, r_in=inner_bkg, r_out=outer_bkg)
        else:
            annulus = None

        subtract_background_plane = subtract_background
        # Compute the area of the aperture and possibly also of the annulus.
        # for each wavelength bin (taking into account empty spaxels)
        normalization = 1.
        temp_weightmap = weightmap[k, :, :]
        temp_weightmap[temp_weightmap > 1] = 1
        aperture_area = 0
        annulus_area = 0

        # aperture_photometry - using weight map
        phot_table = aperture_photometry(temp_weightmap, aperture,
                                         method=method, subpixels=subpixels)

        aperture_area = float(phot_table['aperture_sum'][0])
        log.debug("aperture.area = %g; aperture_area = %g",
                  aperture.area, aperture_area)

        if(aperture_area == 0 and aperture.area > 0):
            aperture_area = aperture.area

        if subtract_background and annulus is not None:
            # Compute the area of the annulus.
            phot_table = aperture_photometry(temp_weightmap, annulus,
                                             method=method, subpixels=subpixels)
            annulus_area = float(phot_table['aperture_sum'][0])
            log.debug("annulus.area = %g; annulus_area = %g",
                      annulus.area, annulus_area)

            if(annulus_area == 0 and annulus.area > 0):
                annulus_area = annulus.area

            if annulus_area > 0.:
                normalization = aperture_area / annulus_area
            else:
                log.warning("Background annulus has no area, so background "
                            f"subtraction will be turned off. {k}")
                subtract_background_plane = False

        npixels[k] = aperture_area
        npixels_bkg[k] = 0.0
        if annulus is not None:
            npixels_bkg[k] = annulus_area
        # aperture_photometry - using data

        phot_table = aperture_photometry(data[k, :, :], aperture,
                                         method=method, subpixels=subpixels)
        temp_flux[k] = float(phot_table['aperture_sum'][0])

        var_poisson_table = aperture_photometry(var_poisson[k, :, :], aperture,
                                                method=method, subpixels=subpixels)
        f_var_poisson[k] = float(var_poisson_table['aperture_sum'][0])

        var_rnoise_table = aperture_photometry(var_rnoise[k, :, :], aperture,
                                               method=method, subpixels=subpixels)
        f_var_rnoise[k] = float(var_rnoise_table['aperture_sum'][0])

        var_flat_table = aperture_photometry(var_flat[k, :, :], aperture,
                                             method=method, subpixels=subpixels)
        f_var_flat[k] = float(var_flat_table['aperture_sum'][0])

        # Point source type of data with defined annulus size
        if subtract_background_plane:
            bkg_table = aperture_photometry(data[k, :, :], annulus,
                                            method=method, subpixels=subpixels)
            background[k] = float(bkg_table['aperture_sum'][0])
            temp_flux[k] = temp_flux[k] - background[k] * normalization

            var_poisson_table = aperture_photometry(var_poisson[k, :, :], annulus,
                                                    method=method, subpixels=subpixels)
            b_var_poisson[k] = float(var_poisson_table['aperture_sum'][0])

            var_rnoise_table = aperture_photometry(var_rnoise[k, :, :], annulus,
                                                   method=method, subpixels=subpixels)
            b_var_rnoise[k] = float(var_rnoise_table['aperture_sum'][0])

            var_flat_table = aperture_photometry(var_flat[k, :, :], annulus,
                                                 method=method, subpixels=subpixels)
            b_var_flat[k] = float(var_flat_table['aperture_sum'][0])

        # Extended source data - background determined from sigma clipping
        if source_type == 'EXTENDED':
            bkg_data = data[k, :, :]
            # pull out the data with coverage in IFU cube. We do not want to use
            # the edge data that is zero to define the statistics on clipping
            bkg_stat_data = bkg_data[temp_weightmap == 1]

            bkg_mean, _, bkg_stddev = stats.sigma_clipped_stats(bkg_stat_data,
                                                                sigma=bkg_sigma_clip, maxiters=5)
            low = bkg_mean - bkg_sigma_clip * bkg_stddev
            high = bkg_mean + bkg_sigma_clip * bkg_stddev

            # set up the mask to flag data that should not be used in aperture photometry
            maskclip = np.logical_or(bkg_data < low, bkg_data > high)

            bkg_table = aperture_photometry(bkg_data, aperture, mask=maskclip,
                                            method=method, subpixels=subpixels)
            background[k] = float(bkg_table['aperture_sum'][0])
            phot_table = aperture_photometry(temp_weightmap, aperture, mask=maskclip,
                                             method=method, subpixels=subpixels)
            npixels_bkg[k] = float(phot_table['aperture_sum'][0])

            var_poisson_table = aperture_photometry(var_poisson[k, :, :], aperture, mask=maskclip,
                                                    method=method, subpixels=subpixels)
            b_var_poisson[k] = float(var_poisson_table['aperture_sum'][0])

            var_rnoise_table = aperture_photometry(var_rnoise[k, :, :], aperture, mask=maskclip,
                                                   method=method, subpixels=subpixels)
            b_var_rnoise[k] = float(var_rnoise_table['aperture_sum'][0])

            var_flat_table = aperture_photometry(var_flat[k, :, :], aperture, mask=maskclip,
                                                 method=method, subpixels=subpixels)
            b_var_flat[k] = float(var_flat_table['aperture_sum'][0])

        del temp_weightmap
        # done looping over wavelength bins
    # Check for NaNs in the wavelength array, flag them in the dq array,
    # and truncate the arrays if NaNs are found at endpoints (unless the
    # entire array is NaN).

    wavelength, dq, nan_slc = nans_in_wavelength(wavelength, dq)
    temp_flux = temp_flux[nan_slc]
    background = background[nan_slc]
    npixels = npixels[nan_slc]
    npixels_bkg = npixels_bkg[nan_slc]
    f_var_poisson = f_var_poisson[nan_slc]
    f_var_rnoise = f_var_rnoise[nan_slc]
    f_var_flat = f_var_flat[nan_slc]
    b_var_poisson = b_var_poisson[nan_slc]
    b_var_rnoise = b_var_rnoise[nan_slc]
    b_var_flat = b_var_flat[nan_slc]

    return (ra, dec, wavelength, temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
            background, b_var_poisson, b_var_rnoise, b_var_flat,
            npixels, dq, npixels_bkg, radius_match, x_center, y_center)


def locn_from_wcs(input_model, ra_targ, dec_targ):
    """Get the location of the spectrum, based on the WCS.

    Parameters
    ----------
    input_model : data model
        The input science model.

    ra_targ, dec_targ : float or None
        The right ascension and declination of the target (degrees).

    Returns
    -------
    locn : tuple of two int, or None
        X and Y coordinates (in that order) of the pixel that has right
        ascension and declination coordinates closest to the target
        location.  The spectral extraction region should be centered here.
    """

    if ra_targ is None or dec_targ is None:
        log.warning("TARG_RA and/or TARG_DEC not found; can't compute "
                    "pixel location of target.")
        locn = None
    else:
        shape = input_model.data.shape
        grid = np.indices(shape[-2:])
        z = np.zeros(shape[-2:], dtype=np.float64) + shape[0] // 2
        # The arguments are the X, Y, and Z pixel coordinates, and the
        # output arrays will be 2-D.
        (ra_i, dec_i, wl) = input_model.meta.wcs(grid[1], grid[0], z)
        cart = celestial_to_cartesian(ra_i, dec_i)
        v = celestial_to_cartesian(ra_targ, dec_targ)   # a single vector
        diff = cart - v
        # We want the pixel with the minimum distance from v, but the pixel
        # with the minimum value of distance squared will be the same.
        dist2 = (diff**2).sum(axis=-1)
        nan_mask = np.isnan(wl)
        dist2[..., :] = np.where(nan_mask, HUGE_DIST, dist2[..., :])
        del nan_mask
        k = np.argmin(dist2.ravel())
        (j, i) = divmod(k, dist2.shape[1])      # y, x coordinates

        if i <= 0 or j <= 0 or i >= shape[-1] - 1 or j >= shape[-2] - 1:
            log.warning("WCS implies the target is beyond the edge of the image")
            log.warning("This location will not be used")
            locn = None
        else:
            locn = (i, j)                       # x, y coordinates

    return locn


def celestial_to_cartesian(ra, dec):
    """Convert celestial coordinates to Cartesian.

    Parameters
    ----------
    ra, dec : ndarray or float
        The right ascension and declination (degrees).  Both `ra` and `dec`
        should be arrays of the same shape, or they should both be float.

    Returns
    -------
    cart : ndarray
        If `ra` and `dec` are float, `cart` with be a 3-element array.
        If `ra` and `dec` are arrays, `cart` will be an array with shape
        ra.shape + (3,).
        For each element of `ra` (or `dec`), the last axis of `cart` will
        give the Cartesian coordinates of a unit vector in the direction
        `ra`, `dec`.  The elements of the vector in Cartesian coordinates
        are in the order x, y, z, where x is the direction toward the
        vernal equinox, y is the direction toward right ascension = 90
        degrees (6 hours) and declination = 0, and z is toward the north
        celestial pole.
    """

    if hasattr(ra, 'shape'):
        shape = ra.shape + (3,)
    else:
        shape = (3,)

    cart = np.zeros(shape, dtype=np.float64)
    cart[..., 2] = np.sin(dec * np.pi / 180.)
    r_xy = np.cos(dec * np.pi / 180.)
    cart[..., 1] = r_xy * np.sin(ra * np.pi / 180.)
    cart[..., 0] = r_xy * np.cos(ra * np.pi / 180.)

    return cart


def image_extract_ifu(input_model, source_type, extract_params):
    """Extraction using a extract1d reference image.

    Extended summary
    ----------------
    The IMAGE extract1d reference file should have pixels with avalue of 1 for the
    source extraction region, 0 for pixels not to include in source or background,
    and -1 for the background region.
    For SRCTYPE=POINT the source extraction region is defined by pixels in the ref
    image = 1 and the background region is defiend by pixels in the ref image with
    -1.
    For SRCTYPE=EXTENDED the extraction region is defined by pixels in the ref image
    =  1 (only the source region is used). The default procedure of using the extract 1d
    asdf reference files extracts the entire region for EXTENDED source data. However,
    if the user supplies the reference image it is assumed they have defined a specific
    region to be extracted instead of the entire field. At each wavelenght bin sigma
    clipping is performed on the extration region and is store in the background column of
    spec table to be used in masterbackground subtraction. In the extended source case
    pixels flagged as background (-1) in the reference image are ignored.

    Parameters
    ----------
    input_model : IFUCubeModel
        The input model.

    source_type : string
        "POINT" or "EXTENDED"

    extract_params : dict
        The extraction parameters.  One of these is an open file handle
        for an image that specifies which pixels should be included as
        source and which (if any) should be used as background.

    Returns
    -------
    ra, dec : float
        ra and dec are the right ascension and declination respectively
        at the centroid of the target region in the reference image.

    wavelength : ndarray, 1-D
        The wavelength in micrometers at each plane of the IFU cube.

    temp_flux : ndarray, 1-D
        The sum of the data values in the extraction aperture minus the
        sum of the data values in the background region (scaled by the
        ratio of areas), for each plane.
        The data values are in units of surface brightness, so this value
        isn't really the flux, it's an intermediate value.  Dividing by
        `npixels` (to compute the average) will give the value for the
        `surf_bright` (surface brightness) column, and multiplying by
        the solid angle of a pixel will give the flux for a point source.

    f_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        temp_flux array.

    f_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        temp_flux array.

    f_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        temp_flux array.

    background : ndarray, 1-D
        The background count rate that was subtracted from the total
        source data values to get `temp_flux`.

    b_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        background array.

    b_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        background array.

    b_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        background array.

    npixels : ndarray, 1-D, float64
        For each slice, this is the number of pixels that were added
        together to get `temp_flux`.

    dq : ndarray, 1-D, uint32
        The data quality array.

    n_bkg : ndarray, 1-D, float64
        For each slice, this is the number of pixels that were added
        together to get background.

    x_center, y_center : float
        The x and y center of the extraction region
    """

    data = input_model.data
    try:
        var_poisson = input_model.var_poisson
        var_rnoise = input_model.var_rnoise
        var_flat = input_model.var_flat
    except AttributeError:
        var_poisson = np.zeros_like(input_model.data)
        var_rnoise = np.zeros_like(input_model.data)
        var_flat = np.zeros_like(input_model.data)
    # The axes are (z, y, x) in the sense that (x, y) are the ordinary
    # axes of the image for one plane, i.e. at one wavelength.  The
    # wavelengths vary along the z-axis.
    shape = data.shape

    # The dispersion direction is the first axis.
    npixels = np.ones(shape[0], dtype=np.float64)
    n_bkg = np.ones(shape[0], dtype=np.float64)

    dq = np.zeros(shape[0], dtype=np.uint32)

    ref_image = extract_params['ref_image']
    ref = ref_image.data
    subtract_background = extract_params['subtract_background']

    (mask_target, mask_bkg) = separate_target_and_background(ref)

    # subtracting the background is only allowed for source_type=POINT
    # subtract_background = False for EXTENDED data (set in ifu_extract1d)

    if subtract_background:
        if mask_bkg is None:
            log.info("Skipping background subtraction because "
                     "background regions are not defined.")
        subtract_background = False

    ra_targ = input_model.meta.target.ra
    dec_targ = input_model.meta.target.dec
    locn = locn_from_wcs(input_model, ra_targ, dec_targ)
    x_center = None
    y_center = None
    if locn is not None:
        log.info("Target location is x_center = %g, y_center = %g, "
                 "based on TARG_RA and TARG_DEC.", locn[0], locn[1])

    # Use the centroid of mask_target as the point where the target
    # would be without any source position correction.
    (y0, x0) = im_centroid(data, mask_target)
    log.debug("Target location based on reference image is X = %g, Y = %g",
              x0, y0)

    # TODO - check if shifting location should be done for reference_image option
    if locn is None or np.isnan(locn[0]):
        log.warning("Couldn't determine pixel location from WCS, so "
                    "source position correction will not be applied.")
    else:
        (x_center, y_center) = locn
        # Shift the reference image so it will be centered at locn.
        # Only shift by a whole number of pixels.
        delta_x = int(round(x_center - x0))         # must be integer
        delta_y = int(round(y_center - y0))
        log.debug("Shifting reference image by %g in X and %g in Y",
                  delta_x, delta_y)
        temp = shift_ref_image(mask_target, delta_y, delta_x)
        if temp is not None:
            mask_target = temp
            del temp
            if mask_bkg is not None:
                mask_bkg = shift_ref_image(mask_bkg, delta_y, delta_x)
            # Since we have shifted mask_target and mask_bkg to
            # x_center and y_center, update x0 and y0.
            x0 = x_center
            y0 = y_center

    # Extract the data.
    # First add up the values along the x direction, then add up the
    # values along the y direction.
    gross = (data * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)
    f_var_poisson = (var_poisson * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)
    f_var_rnoise = (var_rnoise * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)
    f_var_flat = (var_flat * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)

    # Compute the number of pixels that were added together to get gross.
    normalization = 1.

    weightmap = input_model.weightmap
    temp_weightmap = weightmap
    temp_weightmap[temp_weightmap > 1] = 1
    npixels[:] = (temp_weightmap * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)
    bkg_sigma_clip = extract_params['bkg_sigma_clip']

    # Point Souce data 1. extract background and subtract 2. do not
    if source_type == 'POINT':
        if subtract_background and mask_bkg is not None:
            n_bkg[:] = (temp_weightmap * mask_bkg).sum(axis=2, dtype=np.float64).sum(axis=1)
            n_bkg[:] = np.where(n_bkg <= 0., 1., n_bkg)
            normalization = npixels / n_bkg
            background = (data * mask_bkg).sum(axis=2, dtype=np.float64).sum(axis=1)
            b_var_poisson = (var_poisson * mask_bkg).sum(axis=2, dtype=np.float64).sum(axis=1)
            b_var_rnoise = (var_rnoise * mask_bkg).sum(axis=2, dtype=np.float64).sum(axis=1)
            b_var_flat = (var_flat * mask_bkg).sum(axis=2, dtype=np.float64).sum(axis=1)
            temp_flux = gross - background * normalization
        else:
            background = np.zeros_like(gross)
            b_var_poisson = np.zeros_like(gross)
            b_var_rnoise = np.zeros_like(gross)
            b_var_flat = np.zeros_like(gross)
            temp_flux = gross.copy()
    else:
        temp_flux = (data * mask_target).sum(axis=2, dtype=np.float64).sum(axis=1)

    # Extended source data, sigma clip outliers of extraction region is performed
    # at each wavelength plane.
        (background, b_var_poisson, b_var_rnoise,
         b_var_flat, n_bkg) = sigma_clip_extended_region(data, var_poisson, var_rnoise,
                                                         var_flat, mask_target,
                                                         temp_weightmap, bkg_sigma_clip)

    del temp_weightmap

    # Compute the ra, dec, and wavelength at the pixels of a column through
    # the IFU cube at the target location.  ra and dec should be constant
    # (so they're scalars), but wavelength will vary from plane to plane.

    log.debug("IFU 1-D extraction parameters (using reference image):")
    log.debug("  x_center = %s", str(x0))
    log.debug("  y_center = %s", str(y0))
    log.debug("  subtract_background parameter = %s", str(subtract_background))
    if mask_bkg is not None:
        log.debug("    background will be subtracted")
    else:
        log.debug("    background will not be subtracted")

    (ra, dec, wavelength) = get_coordinates(input_model, x0, y0)

    # Check for NaNs in the wavelength array, flag them in the dq array,
    # and truncate the arrays if NaNs are found at endpoints (unless the
    # entire array is NaN).
    wavelength, dq, nan_slc = nans_in_wavelength(wavelength, dq)
    temp_flux = temp_flux[nan_slc]
    background = background[nan_slc]
    npixels = npixels[nan_slc]
    n_bkg = n_bkg[nan_slc]
    f_var_poisson = f_var_poisson[nan_slc]
    f_var_rnoise = f_var_rnoise[nan_slc]
    f_var_flat = f_var_flat[nan_slc]
    b_var_poisson = b_var_poisson[nan_slc]
    b_var_rnoise = b_var_rnoise[nan_slc]
    b_var_flat = b_var_flat[nan_slc]

    return (ra, dec, wavelength, temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
            background, b_var_poisson, b_var_rnoise, b_var_flat,
            npixels, dq, n_bkg, x_center, y_center)


def get_coordinates(input_model, x0, y0):
    """Get celestial coordinates and wavelengths.

    Parameters
    ----------
    input_model : IFUCubeModel
        The input model.

    x0, y0 : float
        The pixel number at which the coordinates should be determined.
        If the extract1d reference file is JSON format, this point will be the
        nominal center of the image.  For an image extract1d reference file this
        will be the centroid of the pixels that were flagged as source.

    Returns
    -------
    ra, dec : float
        ra and dec are the right ascension and declination respectively
        at pixel (0, y0, x0).

    wavelength : ndarray, 1-D
        The wavelength in micrometers at each pixel.
    """

    if hasattr(input_model.meta, 'wcs'):
        wcs = input_model.meta.wcs
    else:
        log.warning("WCS function not found in input.")
        wcs = None

    nelem = input_model.data.shape[0]

    if wcs is not None:
        x_array = np.empty(nelem, dtype=np.float64)
        x_array.fill(x0)
        y_array = np.empty(nelem, dtype=np.float64)
        y_array.fill(y0)
        z_array = np.arange(nelem, dtype=np.float64)  # for wavelengths
        ra, dec, wavelength = wcs(x_array, y_array, z_array)
        # ra and dec should be constant, so nelem // 2 is an arbitrary element.
        ra = ra[nelem // 2]
        dec = dec[nelem // 2]
    else:
        (ra, dec) = (0., 0.)
        wavelength = np.arange(1, nelem + 1, dtype=np.float64)

    return (ra, dec, wavelength)


def nans_in_wavelength(wavelength, dq):
    """Check for NaNs in the wavelength array.

    If NaNs are found in the wavelength array, flag them in the dq array,
    and truncate the arrays at either or both ends if NaNs are found at
    endpoints (unless the entire array is NaN).

    Parameters
    ----------
    wavelength : ndarray, 1-D, float64
        The wavelength in micrometers at each pixel.

    dq : ndarray, 1-D, uint32
        The data quality array.

    Returns
    -------
    wavelength : ndarray, 1-D, float64

    dq : ndarray, 1-D, uint32

    slc : slice
    """

    nelem = np.size(wavelength)
    slc = slice(nelem)
    if nelem == 0:
        log.warning("Output arrays are empty!")
        return (wavelength, dq, slice(nelem))

    nan_mask = np.isnan(wavelength)
    n_nan = nan_mask.sum(dtype=np.intp)
    if n_nan == nelem:
        log.warning("Wavelength array is all NaN!")
        dq = np.bitwise_or(dq[:], dqflags.pixel['DO_NOT_USE'])
        return (wavelength, dq, slice(0))

    if n_nan > 0:
        log.warning("%d NaNs in wavelength array.", n_nan)
        dq[nan_mask] = np.bitwise_or(dq[nan_mask], dqflags.pixel['DO_NOT_USE'])
        not_nan = np.logical_not(nan_mask)
        flag = np.where(not_nan)
        if len(flag[0]) > 0:
            n_trimmed = flag[0][0] + nelem - (flag[0][-1] + 1)
            if n_trimmed > 0:
                slc = slice(flag[0][0], flag[0][-1] + 1)
                wavelength = wavelength[slc]
                dq = dq[slc]
                log.info("Output arrays have been trimmed by %d elements",
                         n_trimmed)

    return wavelength, dq, slc


def separate_target_and_background(ref):
    """Create masks for target and background.

    Parameters
    ----------
    ref : ndarray, 2-D or 3-D
        This is the reference image data array.  This should be the same
        shape as one plane of the science data (or the same shape as the
        entire 3-D science data array).  A value of 1 in a pixel indicates
        that the pixel should be included when computing the target
        spectrum.  A value of 0 means the pixel is not part of either the
        target or background.  A value of -1 means the pixel should be
        included as part of the background region.

    Returns
    -------
    mask_target : ndarray, 2-D or 3-D
        This is an array of the same type and shape as the reference
        image, but with values of only 0 or 1.  A value of 1 indicates
        that the corresponding pixel of the science data array should be
        included when adding up values to make the 1-D spectrum, and a
        value of 0 means that it should not be included.

    mask_bkg : ndarray, 2-D or 3-D, or None.
        This is like `mask_target` (i.e. values are 0 or 1) but for
        background regions.  A value of -1 in `mask_bkg` indicates a pixel
        that should be included as part of the background.  If there is no
        pixel in the reference image with a value of -1, `mask_bkg` will
        be set to None.
    """

    mask_target = np.where(ref == 1., 1., 0.)

    if np.any(ref == -1.):
        mask_bkg = np.where(ref == -1., 1., 0.)
    else:
        mask_bkg = None

    return (mask_target, mask_bkg)


def im_centroid(data, mask_target):
    """Compute the mean location of the target.

    Parameters
    ----------
    data : ndarray, 3-D
        This is the science image data array.

    mask_target : ndarray, 2-D or 3-D
        This is an array of the same type and shape as one plane of the
        science image (or the same type and shape of the entire 3-D science
        image), but with values of 0 or 1, where 1 indicates a pixel within
        the source.

    Returns
    -------
    y0, x0 : tuple of two float
        The centroid of pixels flagged as source.
    """

    # Collapse the science data along the dispersion direction to get a
    # 2-D image of the IFU field of view.  Multiplying by mask_target
    # zeros out all pixels that are not regarded as part of the target
    # (or targets).
    if len(mask_target.shape) == 2:
        data_2d = data.sum(axis=0, dtype=np.float64) * mask_target
    else:
        data_2d = (data * mask_target).sum(axis=0, dtype=np.float64)
    if data_2d.sum() == 0.:
        log.warning("Couldn't compute image centroid.")
        shape = data_2d.shape
        y0 = shape[0] / 2.
        x0 = shape[0] / 2.
        return (y0, x0)

    x_profile = data_2d.sum(axis=0, dtype=np.float64)
    x = np.arange(data_2d.shape[1], dtype=np.float64)
    s_x = (x_profile * x).sum()
    x0 = s_x / x_profile.sum()

    y_profile = data_2d.sum(axis=1, dtype=np.float64)
    y = np.arange(data_2d.shape[0], dtype=np.float64)
    s_y = (y_profile * y).sum()
    y0 = s_y / y_profile.sum()

    return (y0, x0)


def shift_ref_image(mask, delta_y, delta_x, fill=0):
    """Apply source position offset to target or background for ref image.

    Parameters
    ----------
    mask : ndarray, 2-D or 3-D
        This is either the target mask or the background mask, which was
        created from an image extract1d reference file.
        It is assumed that all pixels in the science data are good.  If
        that is not correct, `shift_ref_image` may be called with a data
        quality array as the first argument, in order to obtain a shifted
        data quality array.  In this case, `fill` should be set to a
        positive value, e.g. 1.

    delta_y, delta_x : int
        These are the shifts to be applied to the vertical and horizontal
        axes respectively.

    fill : int or float
        The output array will be initialized to this value.  This should
        be 0 (the default) if `mask` is a target or background mask, but
        it should be set to 1 (or some other positive value) if `mask` is
        a data quality array.

    Returns
    -------
    temp : ndarray, same type and shape as `mask`
        A copy of `mask`, but shifted by `delta_y` and `delta_x`.
    """

    if delta_x == 0 and delta_y == 0:
        return mask.copy()

    shape = mask.shape
    if abs(delta_y) >= shape[-2] or abs(delta_x) >= shape[-1]:
        log.warning("Nod offset %d or %d is too large, skipping ...",
                    delta_y, delta_x)
        return None

    if delta_y > 0:
        islice_y = slice(0, -delta_y)
        oslice_y = slice(delta_y, shape[-2])
    elif delta_y < 0:
        islice_y = slice(-delta_y, shape[-2])
        oslice_y = slice(0, delta_y)
    else:
        islice_y = slice(0, shape[-2])
        oslice_y = slice(0, shape[-2])

    if delta_x > 0:
        islice_x = slice(0, -delta_x)
        oslice_x = slice(delta_x, shape[-1])
    elif delta_x < 0:
        islice_x = slice(-delta_x, shape[-1])
        oslice_x = slice(0, delta_x)
    else:
        islice_x = slice(0, shape[-1])
        oslice_x = slice(0, shape[-1])

    temp = np.zeros_like(mask) + fill
    temp[..., oslice_y, oslice_x] = mask[..., islice_y, islice_x]

    return temp


def sigma_clip_extended_region(data, var_poisson, var_rnoise, var_flat, mask_targ, wmap, sigma_clip):
    """ sigma clip the extraction region

    Parameters
    ----------
    data : ndarray, 3-D
        Input data array to perform extraction from

    var_poisson : ndarray, 2-D
        Poisson noise variance array to be extracted following data extraction method.

    var_rnoise : ndarray, 2-D
        Read noise variance array to be extracted following data extraction method.

    var_flat : ndarray, 2-D
        Flat noise variance array to be extracted following data extraction method.

    mask_targ : ndarray, 2-D or 3-D
        Mask of pixels defining the extended source region. A value of 1 indicated
        pixel is in the extraction region.
    wmap : ndarray, 3-D
       weight map for IFU
    sigma_clip : float
        outlier sigma clipping parameter

    Returns
    -------
    sigma_clip_region : ndarray, 1-D. Summed extracted region with sigma clipping for each wavelength plane
    d_var_poisson : ndarray, 1-D. Sigma-clipped var_poisson array
    d_var_rnoise : ndarray, 1-D. Sigma-clipped var_rnoise array
    d_var_flat : ndarray, 1-D. Sigma-clipped var_flat array
    n_bkg : ndarray, 1-D, sum of pixels used in sigma clipped extracted region
    """
    shape = data.shape
    shape_ref = mask_targ.shape
    n_bkg = np.ones(shape[0], dtype=np.float64)

    sigma_clip_region = np.zeros(shape[0], dtype=np.float64)
    d_var_poisson = np.zeros(shape[0], dtype=np.float64)
    d_var_rnoise = np.zeros(shape[0], dtype=np.float64)
    d_var_flat = np.zeros(shape[0], dtype=np.float64)

    # for each wavelength plane mark outliers as 0 in mask_bkg
    for k in range(shape[0]):  # looping over wavelength
        if len(shape_ref) == 2:
            extract_region = mask_targ.copy()
        else:
            extract_region = mask_targ[k, :, :].copy()
        data_plane = data[k, :, :]
        var_poisson_plane = var_poisson[k, :, :]
        var_rnoise_plane = var_rnoise[k, :, :]
        var_flat_plane = var_flat[k, :, :]
        # pull out extract source region to determined stats on for sigma clipping
        extract_data = data_plane[extract_region == 1]
        ext_mean, _, ext_stddev = stats.sigma_clipped_stats(extract_data,
                                                            sigma=sigma_clip, maxiters=5)
        low = ext_mean - sigma_clip * ext_stddev
        high = ext_mean + sigma_clip * ext_stddev

        # set up the mask to flag data that should not be used
        maskclip = np.logical_or(data_plane < low, data_plane > high)  # flag outliers
        extract_region[maskclip] = 0

        sigma_clip_region[k] = np.sum(data_plane * extract_region * wmap[k, :, :])
        d_var_poisson[k] = np.sum(var_poisson_plane * extract_region * wmap[k, :, :])
        d_var_rnoise[k] = np.sum(var_rnoise_plane * extract_region * wmap[k, :, :])
        d_var_flat[k] = np.sum(var_flat_plane * extract_region * wmap[k, :, :])
        n_bkg[k] = np.sum(wmap[k, :, :] * extract_region)

    return sigma_clip_region, d_var_poisson, d_var_rnoise, d_var_flat, n_bkg
