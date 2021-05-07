"""
1-D spectral extraction

:Authors: Mihai Cara (contact: help@stsci.edu)

"""

# STDLIB
import logging
import math
import copy

# THIRD PARTY
import numpy as np
from astropy.modeling import models, fitting

__all__ = ['extract1d']
__taskname__ = 'extract1d'
__author__ = 'Mihai Cara'

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def extract1d(image, var_poisson, var_rnoise, var_flat, lambdas, disp_range,
              p_src, p_bkg=None, independent_var="wavelength",
              smoothing_length=0, bkg_fit="poly", bkg_order=0, weights=None):
    """Extract the spectrum, optionally subtracting background.

    Parameters:
    -----------
    image : 2-D ndarray
        The array may have been transposed so that the dispersion direction
        is the second index.

    var_poisson : 2-D ndarray
        The array may have been transposed so that the dispersion direction
        is the second index.

    var_rnoise : 2-D ndarray
        The array may have been transposed so that the dispersion direction
        is the second index.

    var_flat : 2-D ndarray
        The array may have been transposed so that the dispersion direction
        is the second index.

    lambdas : 1-D array
        Wavelength at each pixel within `disp_range`.  For example,
        lambdas[0] is the wavelength for image[:, disp_range[0]].

    disp_range : two-element list
        Limits of a slice for extracting the spectrum from `image`.

    p_src : list of two-element lists of functions
        These are Astropy polynomial functions defining the limits in the
        cross-dispersion direction of the source extraction region(s).

    p_bkg : list of two-element lists of functions, or None
        These are Astropy polynomial functions defining the limits in the
        cross-dispersion direction of the background extraction regions.

    independent_var : string
        The value may be "wavelength" or "pixel", indicating whether the
        independent variable for the source and background polynomial
        functions is wavelength or pixel number.

    smoothing_length : int
        If this is greater than one and background regions have been
        specified, the background regions will be boxcar smoothed by this
        length (must be zero or an odd integer).
        This argument is only used if background regions have been
        specified.

    bkg_fit : string
        Type of fitting to apply to background values in each column (or
        row, if the dispersion is vertical).

    bkg_order : int
        Polynomial order for fitting to each column of background.  A value
        of 0 means that a simple average of the background regions, column
        by column, will be used.
        This argument must be positive or zero, and it is only used if
        background regions have been specified and if `bkg_fit` is `poly`.

    weights : function or None
        If not None, this computes the weights for the source extraction
        region as a function of the wavelength (a single float) for the
        current column and an array of Y pixel coordinates.

    Returns:
    --------
    countrate : ndarray, 1-D, float64
        The extracted spectrum in units of counts / s.

    f_var_poisson : ndarray, 1-D, float64
        The extracted variance due to Poisson noise, units of (counts/s)^2

    f_var_rnoise : ndarray, 1-D, float64
        The extracted variance due to read noise, units of (counts/s)^2

    f_var_flat : ndarray, 1-D, float64
        The extracted flat-field variance, units of (counts/s)^2

    background : ndarray, 1-D, float64
        The background that was subtracted from the source.

    b_var_poisson : ndarray, 1-D, float64
        The extracted background variance due to Poisson noise, units of (counts/s)^2

    b_var_rnoise : ndarray, 1-D, float64
        The extracted background variance due to read noise, units of (counts/s)^2

    b_var_flat : ndarray, 1-D, float64
        The extracted background flat-field variance, units of (counts/s)^2

    npixels : ndarray, 1-D, float64
        For each column, this is the number of pixels that were added
        together to get `countrate`.
    """
    nl = lambdas.shape[0]

    # Evaluate the functions for source and (optionally) background limits,
    # saving the resulting arrays of lower and upper limits in srclim and
    # bkglim.
    # If independent_var is pixel, the zero point is the first column in
    # the input image, rather than the first pixel in the extracted
    # spectrum.  The spectrum will be extracted from image columns
    # disp_range[0] to disp_range[1], so disp_range[0] is the value of
    # the independent variable (if not wavelength) at the first pixel of
    # the extracted spectrum.

    if not (independent_var.startswith("pixel") or
            independent_var.startswith("wavelength")):
        log.warning("independent_var was '%s'; using 'pixel' instead.",
                    independent_var)
        independent_var = "pixel"
    if independent_var.startswith("pixel"):
        # Temporary array for the independent variable.
        pixels = np.arange(disp_range[0], disp_range[1], dtype=np.float64)

    srclim = []                 # this will be a list of lists, like p_src
    n_srclim = len(p_src)
    for i in range(n_srclim):
        lower = p_src[i][0]
        upper = p_src[i][1]
        if independent_var.startswith("wavelength"):    # OK if 'wavelengths'
            srclim.append([lower(lambdas), upper(lambdas)])
        else:
            srclim.append([lower(pixels), upper(pixels)])

    if p_bkg is None:
        nbkglim = 0
    else:
        nbkglim = len(p_bkg)
        bkglim = []             # this will be a list of lists, like p_bkg
        for i in range(nbkglim):
            lower = p_bkg[i][0]
            upper = p_bkg[i][1]
            if independent_var.startswith("wavelength"):
                bkglim.append([lower(lambdas), upper(lambdas)])
            else:
                bkglim.append([lower(pixels), upper(pixels)])

    # Sanity check:  check for extraction limits that are out of bounds,
    # or a lower limit that's above the upper limit (limit curves just
    # swapped, or crossing each other).
    # Truncate extraction limits that are out of bounds, but log a warning.
    shape = image.shape
    for i in range(n_srclim):
        lower = srclim[i][0]
        upper = srclim[i][1]
        diff = upper - lower
        if diff.min() < 0.:
            if diff.max() < 0.:
                log.error("Lower and upper source extraction limits"
                          " appear to be swapped.")
                raise ValueError("Lower and upper source extraction limits"
                                 " appear to be swapped.")
            else:
                log.error("Lower and upper source extraction limits"
                          " cross each other.")
                raise ValueError("Lower and upper source extraction limits"
                                 " cross each other.")
        del diff
        if np.any(lower < -0.5) or np.any(upper < -0.5):
            log.warning("Source extraction limit extends below -0.5")
            srclim[i][0][:] = np.where(lower < -0.5, -0.5, lower)
            srclim[i][1][:] = np.where(upper < -0.5, -0.5, upper)
        upper_limit = float(shape[0]) - 0.5
        if np.any(lower > upper_limit) or np.any(upper > upper_limit):
            log.warning("Source extraction limit extends above %g", upper_limit)
            srclim[i][0][:] = np.where(lower > upper_limit, upper_limit, lower)
            srclim[i][1][:] = np.where(upper > upper_limit, upper_limit, upper)
    for i in range(nbkglim):
        lower = bkglim[i][0]
        upper = bkglim[i][1]
        diff = upper - lower
        if diff.min() < 0.:
            if diff.max() < 0.:
                log.error("Lower and upper background extraction limits"
                          " appear to be swapped.")
                raise ValueError("Lower and upper background extraction limits"
                                 " appear to be swapped.")
            else:
                log.error("Lower and upper background extraction limits"
                          " cross each other.")
                raise ValueError("Lower and upper background extraction limits"
                                 " cross each other.")
        del diff
        if np.any(lower < -0.5) or np.any(upper < -0.5):
            log.warning("Background limit extends below -0.5")
            bkglim[i][0][:] = np.where(lower < -0.5, -0.5, lower)
            bkglim[i][1][:] = np.where(upper < -0.5, -0.5, upper)
        upper_limit = float(shape[0]) - 0.5
        if np.any(lower > upper_limit) or np.any(upper > upper_limit):
            log.warning("Background limit extends above %g", upper_limit)
            bkglim[i][0][:] = np.where(lower > upper_limit, upper_limit, lower)
            bkglim[i][1][:] = np.where(upper > upper_limit, upper_limit, upper)

    # Smooth the input image, and use the smoothed image for extracting
    # the background.  temp_image is only needed for background data.
    if nbkglim > 0 and smoothing_length > 1:
        temp_image = bxcar(image, smoothing_length)
    else:
        temp_image = image

    #################################################
    #          Perform spectral extraction:         #
    #################################################

    bkg_model = None
    b_var_poisson_model = None
    b_var_rnoise_model = None
    b_var_flat_model = None

    countrate = np.zeros(nl, dtype=np.float64)
    f_var_poisson = np.zeros(nl, dtype=np.float64)
    f_var_rnoise = np.zeros(nl, dtype=np.float64)
    f_var_flat = np.zeros(nl, dtype=np.float64)
    background = np.zeros(nl, dtype=np.float64)
    b_var_poisson = np.zeros(nl, dtype=np.float64)
    b_var_rnoise = np.zeros(nl, dtype=np.float64)
    b_var_flat = np.zeros(nl, dtype=np.float64)
    npixels = np.zeros(nl, dtype=np.float64)
    # x is an index (column number) within `image`, while j is an index in
    # lambdas, countrate, background, npixels, and the arrays in
    # srclim and bkglim.
    x = disp_range[0]
    for j in range(nl):
        lam = lambdas[j]

        if nbkglim > 0:

            # Compute the background for the current column,
            # using the (optionally) smoothed background.
            (bkg_model, b_var_poisson_model, b_var_rnoise_model,
             b_var_flat_model, bkg_npts) = _fit_background_model(
                temp_image, var_poisson, var_rnoise, var_flat,
                j, bkglim, bkg_fit, bkg_order
            )

            if bkg_npts == 0:
                bkg_model = None
                log.warning(f"Not enough valid pixels to determine background "
                            f"for lambda={lam:.6f} (column {x:d})")

            elif len(bkg_model) < bkg_order:
                log.warning(f"Not enough valid pixels to determine background "
                            f"with the required order for lambda={lam:.6f} "
                            f"(column {x:d})\n"
                            f"Lowering background order to {len(bkg_model)}")

        # Extract the source, and optionally subtract background using the
        # fit to the background for this column.  Even if
        # background smoothing was done, we must extract the source from
        # the original, unsmoothed image.
        # source total flux, background total flux, area, total weight
        (countrate[j], f_var_poisson[j], f_var_rnoise[j], f_var_flat[j],
         bkg_flux, b_var_poisson_val, b_var_rnoise_val, b_var_flat_val,
         npixels[j], twht) = _extract_src_flux(
            image, var_poisson, var_rnoise, var_flat, x, j, lam, srclim,
            weights=weights, bkgmodels=[bkg_model, b_var_poisson_model,
                                        b_var_rnoise_model, b_var_flat_model]
        )
        if nbkglim > 0:
            background[j] = bkg_flux
            b_var_poisson[j] = b_var_poisson_val
            b_var_rnoise[j] = b_var_rnoise_val
            b_var_flat[j] = b_var_flat_val

        x += 1
        continue

    return (countrate, f_var_poisson, f_var_rnoise, f_var_flat,
            background, b_var_poisson, b_var_rnoise, b_var_flat, npixels)


def bxcar(image, smoothing_length):
    """Smooth with a 1-D interval, along the last axis.

    Extended summary
    ----------------
    Note that the entire input array will be smoothed, including the
    region containing the source.  The source extraction must therefore
    be done from the original, unsmoothed array.

    Parameters:
    -----------
    image : 2-D ndarray
        The input data array.

    Returns:
    --------
    ndarray, 1-D
        The smoothed input array.
    """

    half = smoothing_length // 2

    shape0 = image.shape
    width = shape0[-1]
    shape = shape0[0:-1] + (width + 2 * half,)
    temp_im = np.zeros(shape, dtype=np.float64)

    i = 0
    for k in range(smoothing_length):
        temp_im[..., i:i + width] += image
        i += 1
    temp_im /= float(smoothing_length)

    return temp_im[..., half:half + width].astype(image.dtype)


def _extract_src_flux(image, var_poisson, var_rnoise, var_flat, x, j, lam, srclim, weights, bkgmodels):
    """Extract the source and subtract background.

    Parameters:
    -----------
    image : 2-D ndarray
        The input data array.

    var_poisson : 2-D ndarray
        The input poisson variance array.

    var_rnoise : 2-D ndarray
        The input read noise variance array.

    var_flat : 2-D ndarray
        The input flat field variance array.

    x : int
        This is an index (column number) within `image`.

    j : int
        This is an index starting with 0 at the first pixel of the
        extracted spectrum.  See `disp_range` in function `extract1d`.
        j = 0 when x = disp_range[0].

    lam : float

    srclim : list of lists of ndarrays
        For each i, srclim[i] is a two-element list.  Those two elements
        are arrays of the lower and upper limits of one of the extraction
        regions.  Of course, there may only be one extraction region.

    weights : function or None
        If not None, this function gives the weights for pixels within
        an extraction region.

    bkgmodels : function

    Returns:
    --------
    total_flux : float or NaN
        Sum of counts within the source extraction region for the current
        column.  This will be NaN if there is no data in the source
        extraction region for the current column.

    f_var_poisson : float or NaN
        Sum of variance within source extraction region for current column.

    f_var_rnoise : float or NaN
        Sum of variance within source extraction region for current column.

    f_var_flat : float or NaN
        Sum of variance within source extraction region for current column.

    bkg_flux : float
        Sum of counts within the background extraction regions for the
        current column.

    b_var_poisson : float or NaN
        Sum of variance within background extraction region for current column.

    b_var_rnoise : float or NaN
        Sum of variance within background extraction region for current column.

    b_var_flat : float or NaN
        Sum of variance within background extraction region for current column.

    tarea : float
        The sum of the number of pixels in the source extraction region for
        the current column.  If only a fraction of a pixel is included at
        an endpoint, that fraction is what would be included in the sum.
        For example, if the source limits (in `srclim`) are (3, 7), then
        the extraction region extends from the middle of pixel 3 to the
        middle of pixel 7, so the pixel areas in the extraction region
        would be:  0.5, 1.0, 1.0, 1.0, 0.5, resulting in `tarea` = 4.

    twht : float
        Two different weights are applied to the pixels.  One is the
        fraction of a pixel that is included in the extraction (this will
        be 1.0 except possibly at the endpoints); see also `tarea`.  The
        other weight depends on the `weights` argument.  If `weights` is
        None, then this weight will be 1.0 for each pixel, and `twht` will
        be the sum of these values.  If the source limits are (3, 7) as in
        the example for `tarea`, `twht` would be 5.0.
    """

    # extract pixel values along the column that are within
    # source limits:
    y, val, area = _extract_colpix(image, x, j, srclim)

    # find indices of "good" (finite) values:
    good = np.isfinite(val)
    npts = good.sum()

    if npts == 0:
        return (np.nan, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0)

    # filter-out bad values:
    # TODO: in the future we may need to develop a way of interpolating
    #       over missing values either from a model or from adjacent columns
    val = val[good]
    area = area[good]
    y = y[good]
    if bkgmodels[0] is None:
        bkg = np.zeros_like(val, dtype=np.float64)
        b_var_poisson = np.zeros_like(val, dtype=np.float64)
        b_var_rnoise = np.zeros_like(val, dtype=np.float64)
        b_var_flat = np.zeros_like(val, dtype=np.float64)
    else:
        bkg = bkgmodels[0](y)
        b_var_poisson = bkgmodels[1](y) * area
        b_var_rnoise = bkgmodels[2](y) * area
        b_var_flat = bkgmodels[3](y) * area

    dummy_y, f_var_poisson_val, dummy_area = _extract_colpix(var_poisson, x, j, srclim)
    f_var_poisson = f_var_poisson_val[good] * area
    dummy_y, f_var_rnoise_val, dummy_area = _extract_colpix(var_rnoise, x, j, srclim)
    f_var_rnoise = f_var_rnoise_val[good] * area
    dummy_y, f_var_flat_val, dummy_area = _extract_colpix(var_flat, x, j, srclim)
    f_var_flat = f_var_flat_val[good] * area

    # subtract background per pixel:
    val -= bkg

    # scale per pixel values by pixel area included in extraction
    val *= area
    bkg *= area

    # compute weights:
    if weights is None:
        wht = np.ones_like(y, dtype=np.float64)
    else:
        wht = weights(lam, y)

    # compute weighted total flux
    # NOTE: the correct formulae must be derived depending on
    #       final interpretation of weights [not available at
    #       initial release v0.0.1]
    tarea = area.sum(dtype=np.float64)
    twht = wht.sum(dtype=np.float64)
    mwht = twht / wht.shape[0]
    total_flux = (val * wht).sum(dtype=np.float64) / mwht
    f_var_poisson = (f_var_poisson * wht).sum(dtype=np.float64) / mwht
    f_var_rnoise = (f_var_rnoise * wht).sum(dtype=np.float64) / mwht
    f_var_flat = (f_var_flat * wht).sum(dtype=np.float64) / mwht
    bkg_flux = bkg.sum(dtype=np.float64)
    b_var_poisson = b_var_poisson.sum(dtype=np.float64)
    b_var_rnoise = b_var_rnoise.sum(dtype=np.float64)
    b_var_flat = b_var_flat.sum(dtype=np.float64)

    return (total_flux, f_var_poisson, f_var_rnoise, f_var_flat,
            bkg_flux, b_var_poisson, b_var_rnoise, b_var_flat, tarea, twht)


def _fit_background_model(image, var_poisson, var_rnoise, var_flat,
                          x, j, bkglim, bkg_fit, bkg_order):
    """Extract background pixels and fit a polynomial. If the number of good data
    points is <= 1, the fit model will be forced to 0 to avoid divergence.

    Parameters:
    -----------
    image : 2-D ndarray
        The input data array.

    var_poisson : 2-D ndarray
        The input poisson variance array.

    var_rnoise : 2-D ndarray
        The input read noise variance array.

    var_flat : 2-D ndarray
        The input flat field variance array.

    x : int
        This is an index (column number) within `image`.

    j : int
        This is an index starting with 0 at the first pixel of the
        extracted spectrum.  See `disp_range` in function `extract1d`.
        j = 0 when x = disp_range[0].

    bkglim : list of lists of arrays
        For each i, bkglim[i] is a two-element list.  Those two elements
        are arrays of the lower and upper limits of one of the background
        extraction regions.

    bkg_fit : str
        Type of "fitting" to perform: "poly" = polynomial, "mean", or
        "median". Note that mathematically the result for "mean" is
        identical to "poly" with `bkg_order`=0.

    bkg_order : int
        Polynomial order for fitting to the background regions of the
        current column.

    Returns:
    --------
    bkg_model : function
        Polynomial fit to the background regions for the current column.
        When the requested fit is either "mean" or "median", the returned
        model is a 0th-order polynomial with c0 equal to the mean/median.

    b_var_poisson_model : function
        Polynomial fit to the background regions for var_poisson values.

    b_var_rnoise_model : function
        Polynomial fit to the background regions for var_rnoise values.

    b_var_flat_model : function
        Polynomial fit to the background regions for var_flat values.

    npts : int
        This is intended to be the number of good values in the background
        regions.  If the background limits are at pixel edges, however,
        `npts` can include a pixel with zero weight; that is, `npts` can be
        1 larger than one might expect.
    """

    # extract pixel values along the column that are within
    # background limits:
    y, val, wht = _extract_colpix(image, x, j, bkglim)

    # find indices of "good" (finite) values:
    good = np.isfinite(val)
    npts = good.sum()

    if npts <= 1 or not np.any(good):
        return (models.Polynomial1D(0), models.Polynomial1D(0),
                models.Polynomial1D(0), models.Polynomial1D(0), 0)

    # filter-out bad values:
    val = val[good]
    wht = wht[good]
    y = y[good]

    if wht.sum() == 0:
        return (models.Polynomial1D(0), models.Polynomial1D(0),
                models.Polynomial1D(0), models.Polynomial1D(0), 0)

    # Find values for each variance array according to locations of
    # "good" image values
    dummy_y, var_poisson_val, dummy_area = _extract_colpix(var_poisson, x, j, bkglim)
    var_poisson_val = var_poisson_val[good]
    dummy_y, var_rnoise_val, dummy_area = _extract_colpix(var_rnoise, x, j, bkglim)
    var_rnoise_val = var_rnoise_val[good]
    dummy_y, var_flat_val, dummy_area = _extract_colpix(var_flat, x, j, bkglim)
    var_flat_val = var_flat_val[good]

    # Compute the fit
    if bkg_fit == 'poly':

        # Fit the background values with a polynomial of the requested order
        lsqfitter = fitting.LinearLSQFitter()
        bkg_model = lsqfitter(models.Polynomial1D(min(bkg_order, npts - 1)),
                              y, val, weights=wht)
        b_var_poisson_model = lsqfitter(models.Polynomial1D(min(bkg_order, npts - 1)),
                                        y, var_poisson_val, weights=wht)
        b_var_rnoise_model = lsqfitter(models.Polynomial1D(min(bkg_order, npts - 1)),
                                       y, var_rnoise_val, weights=wht)
        b_var_flat_model = lsqfitter(models.Polynomial1D(min(bkg_order, npts - 1)),
                                     y, var_flat_val, weights=wht)

    elif bkg_fit == 'mean':

        # Compute the mean of the (good) background values
        # only use values with weight=1
        bkg_model = models.Polynomial1D(degree=0, c0=np.mean(val[wht == 1]))
        b_var_poisson_model = models.Polynomial1D(degree=0, c0=np.mean(var_poisson_val[wht == 1]))
        b_var_rnoise_model = models.Polynomial1D(degree=0, c0=np.mean(var_rnoise_val[wht == 1]))
        b_var_flat_model = models.Polynomial1D(degree=0, c0=np.mean(var_flat_val[wht == 1]))

    elif bkg_fit == 'median':

        # Compute the median of the (good) background values
        # only use values with weight=1
        bkg_model = models.Polynomial1D(degree=0, c0=np.median(val[wht == 1]))
        b_var_poisson_model = models.Polynomial1D(degree=0, c0=np.median(var_poisson_val[wht == 1]))
        b_var_rnoise_model = models.Polynomial1D(degree=0, c0=np.median(var_rnoise_val[wht == 1]))
        b_var_flat_model = models.Polynomial1D(degree=0, c0=np.median(var_flat_val[wht == 1]))

    return bkg_model, b_var_poisson_model, b_var_rnoise_model, b_var_flat_model, npts


def _extract_colpix(image_data, x, j, limits):
    """Extract either the source or background data.

    Parameters:
    -----------
    image_data : 2-D ndarray
        The input data array.

    x : int
        This is an index (column number) within `image_data`.

    j : int
        This is an index starting with 0 at the first pixel of the
        extracted spectrum.  See `disp_range` in function `extract1d`.
        j = 0 when x = disp_range[0].

    limits : list of lists of ndarrays
        For each i, limits[i] is a two-element list.  Those two elements
        are 1-D arrays of the pixel coordinates for the lower and upper
        limits of one of the source or background extraction regions.  The
        number of elements in each of these arrays is the number of pixels
        in the domain from disp_range[0] to and including disp_range[1].

    Returns:
    --------
    y : ndarray, float64
        Y pixel coordinates within the current column, for every pixel that
        is included in any of the intervals in `limits`.

    val : ndarray, float64
        The image values at the pixels given by `y`.  That is,
        val[i] = image_data[y[i], x].

    wht : ndarray, float64
        The weight associated with each element in `val`.  The weight
        ranges from 0 to 1, giving the fraction of a pixel that is included
        within an interval.  For example, suppose one of the elements in
        `limits` has values 3.0 and 9.0 for the lower and upper limits for
        pixel number `x`.  That corresponds to this list of Y pixel values:
        [3., 4., 5., 6., 7., 8., 9.].  (You would then see this list as a
        section in `y`.)  Because the integer value is the center of the
        pixel, the lower and upper limits are in the middle of those
        pixels, so the corresponding weights would be:
        [0.5, 1., 1., 1., 1., 1., 0.5].
    """

    # These are the extraction limits in image pixel coordinates:
    intervals = []
    for lim in limits:
        intervals.append([lim[0][j], lim[1][j]])

    if len(intervals) == 0:
        return ([], [], [])

    # optimize limits:
    intervals = _coalesce_bounds(intervals)

    # compute number of data points:
    ns = image_data.shape[0] - 1
    ns12 = ns + 0.5

    npts = sum(map(lambda x: min(ns, int(math.floor(x[1] + 0.5))) -
                   max(0, int(math.floor(x[0] + 0.5))) + 1, intervals))
    npts = max(npts, 1)

    # pre-allocate data arrays:
    y = np.empty(npts, dtype=np.float64)
    val = np.empty(npts, dtype=np.float64)
    wht = np.ones(npts, dtype=np.float64)

    # populate data and weights:
    k = 0
    for i in intervals:
        i1 = i[0] if i[0] >= -0.5 else -0.5
        i2 = i[1] if i[1] <= ns12 else ns12

        ii1 = max(0, int(math.floor(i1 + 0.5)))
        ii1 = min(ii1, ns)
        ii2 = min(ns, int(math.floor(i2 + 0.5)))

        # special case: ii1 == ii2:
        if ii1 == ii2:
            v = image_data[ii1, x]
            val[k] = v
            wht[k] = i2 - i1
            k += 1
            continue

        # bounds in different pixels:
        # a. lower bound:
        v = image_data[ii1, x]
        val[k] = v
        wht[k] = 1.0 - divmod(i1 - 0.5, 1)[1] if i1 >= -0.5 else 1.0

        # b. upper bound:
        v = image_data[ii2, x]
        kn = k + ii2 - ii1
        val[kn] = v
        wht[kn] = divmod(i2 + 0.5, 1)[1] if i2 < ns12 else 1.0

        # c. all other intermediate pixels:
        val[k + 1:kn] = image_data[ii1 + 1:ii2, x]
        y[k:kn + 1] = np.arange(ii1, ii2 + 1, 1, dtype=np.float64)

        k += ii2 - ii1 + 1

    return (y, val, wht)  # pixel coordinate, value, wheight=fractional pixel area


def _coalesce_bounds(segments):
    """Optimize limits.

    Parameters:
    -----------
    segments : list of two-element lists of float
        Each element of `segments` is a list containing the lower and upper
        limits of a source or background extraction region for one of the
        columns in the input image.

    Returns:
    --------
    list of two-element lists of float
        A copy of `segments`, but sorted, and with overlapping intervals
        merged into a smaller number of equivalent intervals.
    """
    if not isinstance(segments, list):
        raise TypeError("'segments' must be a list")

    intervals = copy.deepcopy(segments)

    if all([isinstance(x, list) for x in intervals]):
        # make sure each nested list is a list of two numbers:
        for x in intervals:
            if len(x) != 2:
                raise ValueError("Each list in the 'segments' list must have "
                                 "exactly two elements")
            try:
                map(float, x)
            except TypeError:
                raise TypeError("Each segment in 'segments' must be a list of "
                                "two numbers")
    else:
        # we have a single "interval" (or "segment"):
        if len(intervals) != 2:
            raise ValueError("'segments' list must be a list of lists of two "
                             "numbers or 'segments' must be a list of exactly "
                             "two numbers")
        try:
            map(float, intervals)
        except TypeError:
            raise TypeError("Each bound in 'segments' must be a number")

        intervals.sort()
        return intervals

    # sort each "segment" in an increasing order, then sort the entire list
    # in the increasing order of lower limit:
    for segment in intervals:
        segment.sort()
    intervals.sort(key=lambda x: x[0])

    # coalesce intervals/segments:
    if len(intervals) > 0:
        cint = [intervals.pop(0)]
    else:
        return [[]]

    while len(intervals) > 0:
        pint = cint[-1]
        interval = intervals.pop(0)
        if interval[0] <= pint[1]:
            pint[1] = interval[1]
            continue
        cint.append(interval)

    return cint
