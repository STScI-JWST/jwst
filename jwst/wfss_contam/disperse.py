import numpy as np

from scipy.interpolate import interp1d

from ..lib.winclip import get_clipped_pixels
from .sens1d import create_1d_sens


def dispersed_pixel(x0, y0, width, height, lams, flxs, order, wmin, wmax,
                    sens_waves, sens_resp, seg_wcs, grism_wcs, ID, naxis,
                    oversample_factor=2, extrapolate_SED=False, xoffset=0,
                    yoffset=0):
    """
    This function take a list of pixels and disperses them using the information contained
    in the grism image WCS object and returns a list of dispersed pixels and fluxes.

    Parameters
    ----------
    x0 : float array
        Array of x-coordinates of the centers of the pixels. One value for each
        direct image, so if only 1 direct image is in use, this is a single value.
    y0 : float array
        Array of y-coordinates of the centers of the pixels. One value for each
        direct image, so if only 1 direct image is in use, this is a single value.
    width : float
        Width of the pixels to be dispersed.
    height : float
        Height of the pixels to be dispersed.
    lams : float array
        Array of wavelengths corresponding to the fluxes (flxs) for each pixel.
        One wavelength per direct image, so can be a single value.
    flxs : float array
        Array of fluxes (flam) for the pixels contained in x0, y0. If a single
        direct image is in use, this will be a single value.
    order : int
        The spectral order to disperse.
    wmin : float
        Min wavelength to be dispersed.
    wmax : float
        Max wavelength to be dispersed.
    sens_waves : float array
        Array of wavelengths corresponding to flux calibration (sens_resp) values.
    sens_resp : float array
        Flux calibration values as a function of wavelength.
    seg_wcs : WCS object
        The WCS object of the segmentation map.
    grism_wcs : WCS object
        The WCS object of the grism image.
    ID : int
        The ID of the object to which the pixel belongs.
    naxis : tuple
        Dimensions (shape) of grism image into which pixels are dispersed.
    oversample_factor : int
        The amount of oversampling required above that of the input spectra or natural dispersion,
        whichever is smaller. Default=2.
    extrapolate_SED : bool
        Whether to allow for the SED of the object to be extrapolated when it does not fully cover the
        needed wavelength range. Default if False.
    xoffset : int
        Pixel offset to apply when computing the dispersion (accounts for padded direct images)
    yoffset : int
        Pixel offset to apply when computing the dispersion (accounts for padded direct images)

    Returns
    -------
    xs : array
        1D array of dispersed pixel x-coordinates
    ys : array
        1D array of dispersed pixel y-coordinates
    areas : array
        1D array of the areas of the incident pixel that when dispersed falls on each dispersed pixel
    lams : array
        1D array of the wavelengths of each dispersed pixel
    counts : array
        1D array of counts for each dispersed pixel
    ID : int
        The source ID. Returned for bookkeeping convenience.
    """

    # Setup the transforms we need from the input WCS objects
    sky_to_imgxy = grism_wcs.get_transform('world', 'detector')
    imgxy_to_grismxy = grism_wcs.get_transform('detector', 'grism_detector')

    # The WCS comes from the first 2D cutout in the grism image, which has
    # the offset from the full-frame origin to the cutout origin in it.
    # We'll use these values later to get back to full-frame coords.
    offset_x = -imgxy_to_grismxy.offset_1
    offset_y = -imgxy_to_grismxy.offset_2

    # Setup function for retrieving flux values at each dispersed wavelength
    if len(lams) > 1:
        # If we have direct image flux values from more than one filter (lambda),
        # we have the option to extrapolate the fluxes outside the
        # wavelength range of the direct images
        if extrapolate_SED is False:
            flux = interp1d(lams, flxs, fill_value=0., bounds_error=False)
        else:
            flux = interp1d(lams, flxs, fill_value="extrapolate", bounds_error=False)
    else:
        # If we only have flux from one lambda, just use that
        # single flux value at all wavelengths
        def flux(x):
            return flxs[0]

    # Get x/y positions in the grism image corresponding to wmin and wmax:
    # Start with RA/Dec of the input pixel position in segmentation map,
    # then convert to x/y in the direct image frame corresponding
    # to the grism image,
    # then finally convert to x/y in the grism image frame
    x0_sky, y0_sky = seg_wcs(x0, y0)
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
    xwmin, ywmin = imgxy_to_grismxy(x0_xy + offset_x, y0_xy + offset_y, wmin, order)
    xwmax, ywmax = imgxy_to_grismxy(x0_xy + offset_x, y0_xy + offset_y, wmax, order)
    dxw = xwmax - xwmin
    dyw = ywmax - ywmin

    # Compute the delta-wave per pixel
    dw = np.abs((wmax - wmin) / (dyw - dxw))

    # Use a natural wavelength scale or the wavelength scale of the input SED/spectrum,
    # whichever is smaller, divided by oversampling requested
    input_dlam = np.median(lams[1:] - lams[:-1])
    if input_dlam < dw:
        dlam = input_dlam / oversample_factor
    else:
        # this value gets used when we only have 1 direct image wavelength
        dlam = dw / oversample_factor

    # Create list of wavelengths on which to compute dispersed pixels
    lambdas = np.arange(wmin, wmax + dlam, dlam)
    n_lam = len(lambdas)

    # Compute lists of x/y positions in the grism image for
    # the set of desired wavelengths:
    # As above, first get RA/Dec of segmentation map pixel positions,
    # then convert to x/y in image frame of grism image,
    # then convert to x/y in grism frame.
    x0_sky, y0_sky = seg_wcs([x0] * n_lam, [y0] * n_lam)
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, lambdas, [order] * n_lam)
    x0s, y0s = imgxy_to_grismxy(x0_xy + offset_x, y0_xy + offset_y, lambdas, [order] * n_lam)

    # If none of the dispersed pixel indexes are within the image frame,
    # return a null result without wasting time doing other computations
    if x0s.min() >= naxis[0] or x0s.max() < 0 or y0s.min() >= naxis[1] or y0s.max() < 0:
        return None

    # Compute arrays of dispersed pixel locations and areas
    padding = 1
    xs, ys, areas, index = get_clipped_pixels(
        x0s, y0s,
        padding,
        naxis[0], naxis[1],
        width, height
    )
    lams = np.take(lambdas, index)

    # If results give no dispersed pixels, return null result
    if xs.size <= 1:
        return None

    # compute 1D sensitivity array corresponding to list of wavelengths
    sens, no_cal = create_1d_sens(lams, sens_waves, sens_resp)

    # Compute countrates for dispersed pixels. Note that dispersed pixel
    # values are naturally in units of physical fluxes, so we divide out
    # the sensitivity (flux calibration) values to convert to units of
    # countrate (DN/s).
    counts = flux(lams) * areas / sens
    counts[no_cal] = 0.  # set to zero where no flux cal info available

    return xs, ys, areas, lams, counts, ID
