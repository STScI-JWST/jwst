#! /usr/bin/env python

# !!!!!!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!!!!!
# Needs work.
# Also, this code makes reference to `nreads` as a the second dimension
# of the 4-D data set, while `ngroups` makes reference to the NGROUPS
# key word in the exposure metadata.  This should be changed, removing
# reference to the NGROUPS key word and using ngroups as the second
# dimension of the 4-D data set.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


'''
import logging
from multiprocessing.pool import Pool as Pool
import numpy as np
import numpy.linalg as la
import time

from .. import datamodels
from ..datamodels import dqflags

from . import utils

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DO_NOT_USE = dqflags.group['DO_NOT_USE']
JUMP_DET = dqflags.group['JUMP_DET']
SATURATED = dqflags.group['SATURATED']
UNRELIABLE_SLOPE = dqflags.pixel['UNRELIABLE_SLOPE']

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# This is the number of iterations that will be done with use_extra_terms
# set to False.  If this is zero, use_extra_terms will be set to True even
# for the first iteration.
# NUM_ITER_NO_EXTRA_TERMS = 1
NUM_ITER_NO_EXTRA_TERMS = 0
# These are the lower and upper limits of the number of iterations that
# will be done by determine_slope.
# MIN_ITER = NUM_ITER_NO_EXTRA_TERMS + 1
# MAX_ITER = 3
MIN_ITER = 1
MAX_ITER = 1

# This is a term to add for saturated pixels to give them low weight.
HUGE_FOR_LOW_WEIGHT = 1.e20

# This is a value to replace zero or negative values in a fit, to make
# all values of the fit positive and to give low weight where the fit was
# zero or negative.
FIT_MUST_BE_POSITIVE = 1.e10


def gls_ramp_fit(input_model, buffsize, save_opt, readnoise_model, gain_model, max_cores):
    """Fit a ramp using generalized least squares.

    Extended Summary
    ----------------
    Calculate the count rate for each pixel in the data ramp, for every
    integration.  Generalized least squares is used for fitting the ramp
    in order to take into account the correlation between reads.  If the
    input file contains multiple integrations, a second output file will
    be written, containing per-integration count rates.

    One additional file can optionally be written (if save_opt is True),
    containing per-integration data.

    Parameters
    ----------
    model : data model
        Input data model, assumed to be of type RampModel.

    buffsize : int
        Size of data section (buffer) in bytes.

    save_opt : boolean
        Calculate optional fitting results.

    readnoise_model : instance of data Model
        Readnoise for all pixels.

    gain_model : instance of gain model
        Gain for all pixels.

    max_cores : string
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    new_model : Data Model object
        DM object containing a rate image averaged over all integrations in
        the exposure.

    int_model : Data Model object or None
        DM object containing rate images for each integration in the exposure,
        or None if there is only one integration.

    gls_opt_model : GLS_RampFitModel object or None
        Object containing optional GLS-specific ramp fitting data for the
        exposure; this will be None if save_opt is False.
    """
    number_slices = utils.compute_slices(max_cores)

    # Get needed sizes and shapes
    nreads, npix, imshape, cubeshape, n_int, instrume, frame_time, ngroups, \
        group_time = utils.get_dataset_info(input_model)

    (group_time, frames_per_group, saturated_flag, jump_flag) = \
        utils.get_more_info(input_model)
    # Get readnoise array for calculation of variance of noiseless ramps, and
    #   gain array in case optimal weighting is to be done
    #   KDG - not sure what this means and no optimal weigting in GLS
    readnoise_2d, gain_2d = utils.get_ref_subs(input_model, readnoise_model,
                                               gain_model, frames_per_group)
    # Flag any bad pixels in the gain
    pixeldq = utils.reset_bad_gain(input_model.pixeldq, gain_2d)
    log.info("number of processes being used is %d" % number_slices)

    total_rows = input_model.data.shape[2]

    tstart = time.time()

    # Determine the maximum number of cosmic ray hits for any pixel.
    max_num_cr = -1                     # invalid initial value
    for num_int in range(n_int):
        i_max_num_cr = utils.get_max_num_cr(input_model.groupdq[num_int, :, :, :], jump_flag)
        max_num_cr = max(max_num_cr, i_max_num_cr)

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim, nframes, groupgap, dropframes1 = utils.get_efftim_ped(input_model)

    if number_slices == 1:
        rows_per_slice = total_rows
        slopes, slope_int, slope_err_int, pixeldq_sect, dq_int, sum_weight, \
            intercept_int, intercept_err_int, pedestal_int, ampl_int, ampl_err_int = \
            gls_fit_all_integrations(frame_time, gain_2d, input_model.groupdq,
                                     group_time, jump_flag, max_num_cr, input_model.data,
                                     input_model.err, nframes, pixeldq, readnoise_2d,
                                     saturated_flag, save_opt)
    else:
        rows_per_slice = round(total_rows / number_slices)
        pool = Pool(processes=number_slices)
        slices = []
        slopes = np.zeros(imshape, dtype=np.float32)
        sum_weight = np.zeros(imshape, dtype=np.float32)

        # For multiple-integration datasets, will output integration-specific
        # results to separate file named <basename> + '_rateints.fits'.
        # Even if there's only one integration, the output results will be
        # saved in these arrays.
        slope_int = np.zeros((n_int,) + imshape, dtype=np.float32)
        slope_err_int = np.zeros((n_int,) + imshape, dtype=np.float32)
        dq_int = np.zeros((n_int,) + imshape, dtype=np.uint32)
        out_pixeldq = np.zeros(imshape, dtype=np.uint32)
        if save_opt:
            # Create arrays for the fitted values of zero-point intercept and
            # cosmic-ray amplitudes, and their errors.
            intercept_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            intercept_err_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            # The pedestal is the extrapolation of the first group back to zero
            # time, for each integration.
            pedestal_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            # If there are no cosmic rays, set the last axis length to 1.
            shape_ampl = (n_int, imshape[0], imshape[1], max(1, max_num_cr))
            ampl_int = np.zeros(shape_ampl, dtype=np.float32)
            ampl_err_int = np.zeros(shape_ampl, dtype=np.float32)

# Loop over number of processes
        for i in range(number_slices - 1):
            start_row = i * rows_per_slice
            stop_row = (i + 1) * rows_per_slice
            readnoise_slice = readnoise_2d[start_row: stop_row, :]
            gain_slice = gain_2d[start_row: stop_row, :]
            data_slice = input_model.data[:, :, start_row:stop_row, :].copy()
            err_slice = input_model.err[:, :, start_row: stop_row, :].copy()
            groupdq_slice = input_model.groupdq[:, :, start_row: stop_row, :].copy()
            pixeldq_slice = pixeldq[start_row: stop_row, :].copy()
            slices.insert(i,
                          (frame_time, gain_slice, groupdq_slice, group_time,
                           jump_flag, max_num_cr, data_slice, err_slice, frames_per_group, pixeldq_slice,
                           readnoise_slice, saturated_flag, save_opt))
        # The last slice takes the remainder of the rows
        start_row = (number_slices - 1) * rows_per_slice
        readnoise_slice = readnoise_2d[start_row: total_rows, :]
        gain_slice = gain_2d[start_row: total_rows, :]
        data_slice = input_model.data[:, :, start_row: total_rows, :].copy()
        err_slice = input_model.err[:, :, start_row: total_rows, :].copy()
        groupdq_slice = input_model.groupdq[:, :, start_row: total_rows, :].copy()
        pixeldq_slice = input_model.pixeldq[start_row: total_rows, :].copy()
        slices.insert(number_slices - 1,
                      (frame_time, gain_slice, groupdq_slice, group_time,
                       jump_flag, max_num_cr, data_slice, err_slice, frames_per_group, pixeldq_slice,
                       readnoise_slice, saturated_flag, save_opt))

        log.debug("Creating %d processes for ramp fitting " % number_slices)
        real_results = pool.starmap(gls_fit_all_integrations, slices)
        pool.close()
        pool.join()
        k = 0
        log.debug("All processes complete")
        for resultslice in real_results:
            start_row = k * rows_per_slice
            if len(real_results) == k + 1:  # last result
                slopes[start_row:total_rows, :] = resultslice[0]
                slope_int[:, start_row:total_rows, :] = resultslice[1]
                slope_err_int[:, start_row:total_rows, :] = resultslice[2]
                out_pixeldq[start_row:total_rows, :] = resultslice[3]
                if resultslice[4] is not None:
                    dq_int[:, start_row:total_rows, :] = resultslice[4]  # nint > 1
                    sum_weight[start_row:total_rows, :] = resultslice[5]  # nint > 1
                if resultslice[6] is not None:
                    intercept_int[:, start_row: total_rows, :] = resultslice[6]  # optional
                    intercept_err_int[:, start_row:total_rows, :] = resultslice[7]  # optional
                    pedestal_int[:, start_row: total_rows, :] = resultslice[8]  # optional
                    ampl_int[:, start_row:total_rows, :] = resultslice[9]  # optional
                    ampl_err_int[:, start_row: total_rows, :] = resultslice[10]  # optional
            else:
                stop_row = (k + 1) * rows_per_slice
                slopes[start_row:stop_row, :] = resultslice[0]
                slope_int[:, start_row:stop_row, :] = resultslice[1]
                slope_err_int[:, start_row:stop_row, :] = resultslice[2]
                out_pixeldq[start_row:stop_row, :] = resultslice[3]
                if resultslice[4] is not None:
                    dq_int[:, start_row:stop_row, :] = resultslice[4]  # nint > 1
                    sum_weight[start_row:stop_row, :] = resultslice[5]  # nint > 1
                if resultslice[6] is not None:
                    intercept_int[:, start_row: stop_row, :] = resultslice[6]  # optional
                    intercept_err_int[:, start_row:stop_row, :] = resultslice[7]  # optional
                    pedestal_int[:, start_row: stop_row, :] = resultslice[8]  # optional
                    ampl_int[:, start_row:stop_row, :] = resultslice[9]  # optional
                    ampl_err_int[:, start_row: stop_row, :] = resultslice[10]  # optional
            k = k + 1
    # Average the slopes over all integrations.
    if n_int > 1:
        sum_weight = np.where(sum_weight <= 0., 1., sum_weight)
        recip_sum_weight = 1. / sum_weight
        slopes *= recip_sum_weight
        gls_err = np.sqrt(recip_sum_weight)

    # Convert back from electrons to DN.
    slope_int /= gain_2d
    slope_err_int /= gain_2d
    if n_int > 1:
        slopes /= gain_2d
        gls_err /= gain_2d
    if save_opt:
        intercept_int /= gain_2d
        intercept_err_int /= gain_2d
        pedestal_int /= gain_2d
        gain_shape = gain_2d.shape
        gain_4d = gain_2d.reshape((1, gain_shape[0], gain_shape[1], 1))
        ampl_int /= gain_4d
        ampl_err_int /= gain_4d
        del gain_4d
    del gain_2d

    # Compress all integration's dq arrays to create 2D PIXELDDQ array for
    #   primary output
    final_pixeldq = utils.dq_compress_final(dq_int, n_int)

    int_model = utils.gls_output_integ(input_model, slope_int, slope_err_int, dq_int)

    if save_opt:  # collect optional results for output
        # Get the zero-point intercepts and the cosmic-ray amplitudes for
        # each integration (even if there's only one integration).
        gls_opt_model = utils.gls_output_optional(
            input_model, intercept_int, intercept_err_int, pedestal_int, ampl_int, ampl_err_int)
    else:
        gls_opt_model = None

    tstop = time.time()

    if n_int > 1:
        utils.log_stats(slopes)
    else:
        utils.log_stats(slope_int[0])

    log.debug('Instrument: %s' % instrume)
    log.debug('Number of pixels in 2D array: %d' % npix)
    log.debug('Shape of 2D image: (%d, %d)' % imshape)
    log.debug('Shape of data cube: (%d, %d, %d)' % cubeshape)
    log.debug('Buffer size (bytes): %d' % buffsize)
    log.debug('Number of rows per slice: %d' % rows_per_slice)
    log.info('Number of groups per integration: %d' % nreads)
    log.info('Number of integrations: %d' % n_int)
    log.debug('The execution time in seconds: %f' % (tstop - tstart,))

    # Create new model...
    if n_int > 1:
        new_model = datamodels.ImageModel(
            data=slopes.astype(np.float32), dq=final_pixeldq, err=gls_err.astype(np.float32))
    else:
        new_model = datamodels.ImageModel(
            data=slope_int[0], dq=final_pixeldq, err=slope_err_int[0])

    new_model.update(input_model)     # ... and add all keys from input

    return new_model, int_model, gls_opt_model


def gls_fit_all_integrations(
        frame_time, gain_2d, gdq_cube, group_time, jump_flag, max_num_cr, data_sect,
        input_var_sect, nframes_used, pixeldq, readnoise_2d, saturated_flag, save_opt):
    """
    This method will fit the rate for all pixels and all integrations using the Generalized Least
    Squares (GLS) method.
     Parameters
    ----------
    frame_time : float32
        The time to read one frame
    gain_2d : 2D float32
        The gain in electrons per DN for each pixel
    gdq_cube : 4-D DQ Flags
        The group dq flag values for all groups in the exposure
    group_time : float32
        The time to read one group
    jump_flag : DQ flag
        The DQ value to mark a jump
    max_num_cr : int
        The largest number of cosmic rays found in any integration
    data_sect : 4-D float32
        The input ramp cube with the sample values for each group of each integration for each pixel
    input_var_sect: 4-D float32
        The input variance for each group of each integration for each pixel
    nframes_used : int
        The number of frames used to form each group average
    pixel_dq : 2-D DQ flags
        The pixel DQ flags for all pixels
    readnoise_2d : 2-D float32
        The read noise for each pixel
    saturated_flag : DQ flag
        The DQ flag value to mark saturation
    save_opt : boolean
        Set to true to return the optional output model
    Returns
    --------
    slopes : 2-D float32
        The output rate for each pixel
    slope_int : 2-D float32
        The output y-intercept for each pixel
    slope_var_sect : 2-D float32
        The variance of the rate for each pixel
    pixeldq_sect : 2-D DQ flag
        The pixel dq for each pixel
    dq_int : 3-D DQ flag
        The pixel dq for each integration for each pixel
    sum_weight : 2-D float32
        The sum of the weights for each pixel
    intercept_int : 3-D float32
        The y-intercept for each integration for each pixel
    intercept_err_int : 3-D float32
        The uncertainty of the y-intercept for each pixel of each integration
    pedestal_int : 3-D float32
        The pedestal value for each integration for each pixel
    ampl_int : 3-D float32
        The amplitude of each cosmic ray for each pixel
    ampl_err_int :
        The variance of the amplitude of each cosmic ray for each pixel
    """
    number_ints = data_sect.shape[0]
    number_rows = data_sect.shape[2]
    number_cols = data_sect.shape[3]
    imshape = (data_sect.shape[2], data_sect.shape[3])
    slope_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.float32)
    slope_err_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.float32)
    dq_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.uint32)
    temp_dq = np.zeros((number_rows, number_cols), dtype=np.uint32)
    slopes = np.zeros((number_rows, number_cols), dtype=np.float32)
    sum_weight = np.zeros((number_rows, number_cols), dtype=np.float32)
    if save_opt:
        # Create arrays for the fitted values of zero-point intercept and
        # cosmic-ray amplitudes, and their errors.
        intercept_int = np.zeros((number_ints,) + imshape, dtype=np.float32)
        intercept_err_int = np.zeros((number_ints,) + imshape, dtype=np.float32)
        # The pedestal is the extrapolation of the first group back to zero
        # time, for each integration.
        pedestal_int = np.zeros((number_ints,) + imshape, dtype=np.float32)
        # The first group, for calculating the pedestal.  (This only needs
        # to be nrows high, but we don't have nrows yet.  xxx)
        first_group = np.zeros(imshape, dtype=np.float32)
        # If there are no cosmic rays, set the last axis length to 1.
        shape_ampl = (number_ints, imshape[0], imshape[1], max(1, max_num_cr))
        ampl_int = np.zeros(shape_ampl, dtype=np.float32)
        ampl_err_int = np.zeros(shape_ampl, dtype=np.float32)
    else:
        intercept_int = None
        intercept_err_int = None
        pedestal_int = None
        first_group = None
        shape_ampl = None
        ampl_int = None
        ampl_err_int = None
    # loop over data integrations
    for num_int in range(number_ints):
        if save_opt:
            first_group[:, :] = 0.  # re-use this for each integration

            # We'll propagate error estimates from previous steps to the
            # current step by using the variance.
        input_var_sect = input_var_sect ** 2

        # Convert the data section from DN to electrons.
        data_sect *= gain_2d
        if save_opt:
            first_group[:, :] = data_sect[num_int, 0, :, :].copy()

        intercept_sect, intercept_var_sect, slope_sect, slope_var_sect, cr_sect, cr_var_sect = \
            determine_slope(data_sect[num_int, :, :, :], input_var_sect[num_int, :, :, :],
                            gdq_cube[num_int, :, :, :], readnoise_2d, gain_2d, frame_time, group_time,
                            nframes_used, max_num_cr, saturated_flag, jump_flag)

        slope_int[num_int, :, :] = slope_sect.copy()
        v_mask = (slope_var_sect <= 0.)
        if v_mask.any():
            # Replace negative or zero variances with a large value.
            slope_var_sect[v_mask] = utils.LARGE_VARIANCE
            # Also set a flag in the pixel dq array.
            temp_dq[:, :][v_mask] = UNRELIABLE_SLOPE
            del v_mask
            # If a pixel was flagged (by an earlier step) as saturated in
            # the first group, flag the pixel as bad.
            # Note:  save s_mask until after the call to utils.gls_pedestal.
        s_mask = (gdq_cube[0] == saturated_flag)
        if s_mask.any():
            temp_dq[:, :][s_mask] = UNRELIABLE_SLOPE
        slope_err_int[num_int, :, :] = np.sqrt(slope_var_sect)

        # We need to take a weighted average if (and only if) number_ints > 1.
        # Accumulate sum of slopes and sum of weights.
        if number_ints > 1:
            weight = 1. / slope_var_sect
            slopes[:, :] += (slope_sect * weight)
            sum_weight[:, :] += weight

        if save_opt:
            # Save the intercepts and cosmic-ray amplitudes for the
            # current integration.
            intercept_int[num_int, :, :] = intercept_sect.copy()
            intercept_err_int[num_int, :, :] = np.sqrt(np.abs(intercept_var_sect))
            pedestal_int[num_int, :, :] = utils.gls_pedestal(first_group[:, :],
                                                             slope_int[num_int, :, :],
                                                             s_mask,
                                                             frame_time, nframes_used)
            ampl_int[num_int, :, :, :] = cr_sect.copy()
            ampl_err_int[num_int, :, :, :] = np.sqrt(np.abs(cr_var_sect))

        # Compress 4D->2D dq arrays for saturated and jump-detected
        # pixels
        pixeldq_sect = pixeldq[:, :].copy()
        dq_int[num_int, :, :] = \
            utils.dq_compress_sect(gdq_cube[num_int, :, :, :], pixeldq_sect).copy()

        dq_int[num_int, :, :] |= temp_dq
        temp_dq[:, :] = 0  # initialize for next integration

    return slopes, slope_int, slope_var_sect, pixeldq_sect, dq_int, sum_weight, \
        intercept_int, intercept_err_int, pedestal_int, ampl_int, ampl_err_int


def determine_slope(data_sect, input_var_sect,
                    gdq_sect, readnoise_sect, gain_sect,
                    frame_time, group_time, nframes_used,
                    max_num_cr, saturated_flag, jump_flag):
    """Iteratively fit a slope, intercept, and cosmic rays to a ramp.

    This function fits a ramp, possibly with discontinuities (cosmic-ray
    hits), to a 3-D data "cube" with shape (number of groups, number of
    pixels high, number of pixels wide).  The fit will be done multiple
    times, with the previous fit being used to assign weights (via the
    covariance matrix) for the current fit.  The iterations stop either
    when the maximum number of iterations has been reached or when the
    maximum difference between the previous fit and the current fit is
    below a cutoff.  This function calls compute_slope and evaluate_fit.

    compute_slope creates arrays for the slope, intercept, and cosmic-ray
    amplitudes (the arrays that will be returned by determine_slope).  Then
    it loops over the number of cosmic rays, from 0 to max_num_cr
    inclusive.  Within this loop, compute_slope copies to temporary arrays
    the ramp data for all the pixels that have the current number of cosmic
    ray hits, calls gls_fit to compute the fit, then copies the results
    of the fit (slope, etc.) to the output arrays for just those pixels.

    The input to gls_fit is ramp data for a subset of pixels (nz in number)
    that all have the same number of cosmic-ray hits.  gls_fit solves
    matrix equations (one for each of the nz pixels) of the form:

        y = x * p

    where y is a column vector containing the observed data values in
    electrons for each group (the length of y is ngroups, the number of
    groups); x is a matrix with ngroups rows and 2 + num_cr columns, where
    num_cr is the number of cosmic rays being included in the fit; and p
    is the solution, a column vector containing the intercept, slope, and
    the amplitude of each of the num_cr cosmic rays.  The first column of
    x is all ones, for fitting to the intercept.  The second column of x
    is the time (seconds) at the beginning of each group.  The remaining
    num_cr columns (if num_cr > 0) are Heaviside functions, 0 for the
    first rows and 1 for all rows at and following the group containing a
    cosmic-ray hit (each row corresponds to a group).  There will be one
    such column for each cosmic ray, so that the cosmic rays will be fit
    independently of each other.  Whether a cosmic ray hit the detector
    during a particular group was determined by a previous step, and the
    affected groups are flagged in the group data quality array.  In order
    to account for the variance of each observed value and the covariance
    between them (since they're measurements along a ramp), the solution
    is computed in this form (the @ sign represents matrix multiplication):

        (xT @ C^-1 @ x)^-1 @ [xT @ C^-1 @ y]

    where C is the ngroups by ngroups covariance matrix, ^-1 means matrix
    inverse, and xT is the transpose of x.

    Summary of the notation:

    data_sect is 3-D, (ngroups, ny, nx); this is the ramp of science data.
    cr_flagged is 3-D, (ngroups, ny, nx); 1 indicates a cosmic ray, e.g.:
        cr_flagged = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)
    cr_flagged_2d is 2-D, (ngroups, nz); this gives the location within
        the ramp of each cosmic ray, for the subset of pixels (nz of them)
        that have a total of num_cr cosmic ray hits at each pixel.  This
        is passed to gls_fit(), which fits a slope to the ramp.

    ramp_data has shape (ngroups, nz); this will be a ramp with a 1-D
    array of pixels copied out of data_sect.  The pixels will be those
    that have a particular number of cosmic-ray hits, somewhere within
    the ramp.

    Sum cr_flagged over groups to get an (ny, nx) image of the number of
    cosmic rays (i.e. accumulated over the ramp) in each pixel.
    sum_flagged = cr_flagged.sum(axis=0, ...)
    sum_flagged is used to extract the nz pixels from (ny, nx) that have a
    specified number of cosmic ray hits, e.g.:
        for num_cr in range(max_num_cr + 1):
            ncr_mask = (sum_flagged == num_cr)
            nz = ncr_mask.sum(dtype=np.int32)
            for k in range(ngroups):
                ramp_data[k] = data_sect[k][ncr_mask]
                cr_flagged_2d[k] = cr_flagged[k][ncr_mask]

    gls_fit is called for the subset of pixels (nz of them) that have
    num_cr cosmic ray hits within the ramp, the same number for every
    pixel.

    Parameters
    ----------
    data_sect: 3-D ndarray, shape (ngroups, ny, nx)
        The ramp data for one integration.  This may be a subarray in
        detector coordinates, but covering all groups.  ngroups is the
        number of groups; ny is the number of pixels in the Y direction;
        nx is the number of pixels in the X (more rapidly varying)
        direction.  The units should be electrons.

    input_var_sect: 3-D ndarray, shape (ngroups, ny, nx)
        The square of the input ERR array, matching data_sect.

    gdq_sect: 3-D ndarray, shape (ngroups, ny, nx)
        The group data quality array.  This may be a subarray, matching
        data_sect.

    readnoise_sect: 2-D ndarray, shape (ny, nx)
        The read noise in electrons at each detector pixel (i.e. not a
        ramp).  This may be a subarray, similar to data_sect.

    gain_sect: 2-D ndarray, or None, shape (ny, nx)
        The gain in electrons per DN at each detector pixel (i.e. not a
        ramp).  This may be a subarray, matching readnoise_sect.  If
        gain_sect is None, a value of 1 will be assumed.

    frame_time: float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time: float
        Time increment between groups, in seconds.

    nframes_used: int
        Number of frames that were averaged together to make a group.
        Note that this value does not include the number (if any) of
        skipped frames.

    max_num_cr: non-negative int
        The maximum number of cosmic rays that should be handled.  This
        must be specified by the caller, because determine_slope may be
        called for different sections of the input data, and those sections
        may have differing maximum numbers of cosmic rays.

    saturated_flag: int
        dqflags.group['SATURATED']

    jump_flag: int
        dqflags.group['JUMP_DET']

    Returns
    -------
    tuple:  (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
             cr_sect, cr_var_sect)
        intercept_sect: 2-D ndarray, shape (ny, nx)
            The intercept of the ramp at each pixel.
        int_var_sect: 2-D ndarray, shape (ny, nx)
            The variance of the intercept at each pixel.
        slope_sect: 2-D ndarray, shape (ny, nx)
            The ramp slope at each pixel of data_sect.
        slope_var_sect: 2-D ndarray, shape (ny, nx)
            The variance of the slope at each pixel.
        cr_sect: 3-D ndarray, shape (ny, nx, cr_dimen)
            The amplitude of each cosmic ray at each pixel.  cr_dimen is
            max_num_cr or 1, whichever is larger.
        cr_var_sect: 3-D ndarray, shape (ny, nx, cr_dimen)
            The variance of each cosmic-ray amplitude.
    """

    slope_diff_cutoff = 1.e-5

    # These will be updated in the loop.
    prev_slope_sect = (data_sect[1, :, :] - data_sect[0, :, :]) / group_time
    prev_fit = data_sect.copy()

    use_extra_terms = True

    iter = 0
    done = False
    if NUM_ITER_NO_EXTRA_TERMS <= 0:
        # Even the first iteration uses the extra terms.
        temp_use_extra_terms = True
    else:
        temp_use_extra_terms = False
    while not done:
        (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
         cr_sect, cr_var_sect) = \
            compute_slope(data_sect, input_var_sect,
                          gdq_sect, readnoise_sect, gain_sect,
                          prev_fit, prev_slope_sect,
                          frame_time, group_time, nframes_used,
                          max_num_cr, saturated_flag, jump_flag,
                          temp_use_extra_terms)
        iter += 1
        if iter == NUM_ITER_NO_EXTRA_TERMS:
            temp_use_extra_terms = use_extra_terms
        if iter >= MAX_ITER:
            done = True
        else:
            # If there are pixels with zero or negative variance, ignore
            # them when taking the difference between the slopes computed
            # in the current and previous iterations.
            slope_diff = np.where(slope_var_sect > 0.,
                                  prev_slope_sect - slope_sect, 0.)
            max_slope_diff = np.abs(slope_diff).max()
            if iter >= MIN_ITER and max_slope_diff < slope_diff_cutoff:
                done = True
            current_fit = evaluate_fit(intercept_sect, slope_sect, cr_sect,
                                       frame_time, group_time,
                                       gdq_sect, jump_flag)
            prev_fit = positive_fit(current_fit)      # use for next iteration
            del current_fit
            prev_slope_sect = slope_sect.copy()

    return (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
            cr_sect, cr_var_sect)


def evaluate_fit(intercept_sect, slope_sect, cr_sect,
                 frame_time, group_time,
                 gdq_sect, jump_flag):
    """Evaluate the fit (intercept, slope, cosmic-ray amplitudes).

    Parameters
    ----------
    intercept_sect: 2-D ndarray
        The intercept of the ramp at each pixel of data_sect (one of the
        arguments to determine_slope).

    slope_sect: 2-D ndarray
        The ramp slope at each pixel of data_sect.

    cr_sect: 3-D ndarray
        The amplitude of each cosmic ray at each pixel of data_sect.

    frame_time: float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time: float
        Time increment between groups, in seconds.

    gdq_sect: 3-D ndarray; indices:  group, y, x
        The group data quality array.  This may be a subarray, matching
        data_sect.

    jump_flag: int
        dqflags.group['JUMP_DET']

    Returns
    -------
    fit_model: 3-D ndarray, shape (ngroups, ny, nx)
        This is the same shape as data_sect, and if the fit is good,
        fit_model and data_sect should not differ by much.
    """

    shape_3d = gdq_sect.shape           # the ramp, (ngroups, ny, nx)
    ngroups = gdq_sect.shape[0]
    # This array is also created in function compute_slope.
    cr_flagged = np.empty(shape_3d, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)

    sum_flagged = cr_flagged.sum(axis=0, dtype=np.int32)
    # local_max_num_cr is local to this function.  It may be smaller than
    # the max_num_cr that's an argument to determine_slope, and it can even
    # be zero.
    local_max_num_cr = sum_flagged.max()
    del sum_flagged

    # The independent variable, in seconds at each image pixel.
    ind_var = np.zeros(shape_3d, dtype=np.float64)
    M = round(group_time / frame_time)
    iv = np.arange(ngroups, dtype=np.float64) * group_time + \
        frame_time * (M + 1.) / 2.
    iv = iv.reshape((ngroups, 1, 1))
    ind_var += iv

    # No cosmic rays yet; these will be accounted for below.
    # ind_var has a different shape (ngroups, ny, nx) from slope_sect and
    # intercept_sect, but their last dimensions are the same.
    fit_model = ind_var * slope_sect + intercept_sect

    # heaviside and cr_flagged have shape (ngroups, ny, nx).
    heaviside = np.zeros(shape_3d, dtype=np.float64)
    cr_cumsum = cr_flagged.cumsum(axis=0, dtype=np.int16)

    # Add an offset for each cosmic ray.
    for n in range(local_max_num_cr):
        heaviside[:] = np.where(cr_cumsum > n, 1., 0.)
        fit_model += (heaviside * cr_sect[:, :, n])

    return fit_model


def positive_fit(current_fit):
    """Replace zero and negative values with a positive number.

    Ramp data should be positive, since they are based on counts.  The
    fit to a ramp can go negative, however, due e.g. to extrapolation
    beyond where the data are saturated.  To avoid negative elements in
    the covariance matrix (populated in part with the fit to the ramp),
    this function replaces zero or negative values in the fit with a
    positive number.

    Parameters
    ----------
    current_fit: 3-D ndarray, shape (ngroups, ny, nx)
        The fit returned by evaluate_fit.

    Returns
    -------
    current_fit: 3-D ndarray, shape (ngroups, ny, nx)
        This is the same as the input current_fit, except that zero and
        negative values will have been replaced by a positive value.
    """

    return np.where(current_fit <= 0., FIT_MUST_BE_POSITIVE, current_fit)


def compute_slope(data_sect, input_var_sect,
                  gdq_sect, readnoise_sect, gain_sect,
                  prev_fit, prev_slope_sect,
                  frame_time, group_time, nframes_used,
                  max_num_cr, saturated_flag, jump_flag,
                  use_extra_terms):
    """Set up the call to fit a slope to ramp data.

    This loops over the number of cosmic rays (jumps).  That is, all the
    ramps with no cosmic rays are processed first, then all the ramps with
    one cosmic ray, then with two, etc.

    Parameters
    ----------
    data_sect: 3-D ndarray; shape (ngroups, ny, nx)
        The ramp data for one of the integrations in an exposure.  This
        may be a subarray in detector coordinates, but covering all groups.

    input_var_sect: 3-D ndarray, shape (ngroups, ny, nx)
        The square of the input ERR array, matching data_sect.

    gdq_sect: 3-D ndarray; shape (ngroups, ny, nx)
        The group data quality array.  This may be a subarray, matching
        data_sect.

    readnoise_sect: 2-D ndarray; shape (ny, nx)
        The read noise in electrons at each detector pixel (i.e. not a
        ramp).  This may be a subarray, similar to data_sect.

    gain_sect: 2-D ndarray, or None; shape (ny, nx)
        The gain in electrons per DN at each detector pixel (i.e. not a
        ramp).  This may be a subarray, matching readnoise_sect.  If
        gain_sect is None, a value of 1 will be assumed.

    prev_fit: 3-D ndarray; shape (ngroups, ny, nx)
        The previous fit (intercept, slope, cosmic-ray amplitudes)
        evaluated for each pixel in the subarray.  data_sect itself may be
        used for the first iteration.

    prev_slope_sect: 2-D ndarray; shape (ny, nx)
        An estimate (e.g. from a previous iteration) of the slope at each
        pixel, in electrons per second.  This may be a subarray, similar to
        data_sect.

    frame_time: float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time: float
        Time increment between groups, in seconds.

    nframes_used: int
        Number of frames that were averaged together to make a group.
        This value does not include the number (if any) of skipped frames.

    max_num_cr: non-negative int
        The maximum number of cosmic rays that should be handled.

    saturated_flag: int
        dqflags.group['SATURATED']

    jump_flag: int
        dqflags.group['JUMP_DET']

    use_extra_terms: bool
        True if we should include Massimo Robberto's terms in the
        covariance matrix.
        See JWST-STScI-003193.pdf

    Returns
    -------
    tuple:  (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
             cr_sect, cr_var_sect)
        intercept_sect is a 2-D ndarray, the intercept of the ramp at each
        pixel of data_sect.
        int_var_sect is a 2-D ndarray, the variance of the intercept at
        each pixel of data_sect.
        slope_sect is a 2-D ndarray, the ramp slope at each pixel of
        data_sect.
        slope_var_sect is a 2-D ndarray, the variance of the slope at each
        pixel of data_sect.
        cr_sect is a 3-D ndarray, shape (ny, nx, cr_dimen), the amplitude
        of each cosmic ray at each pixel of data_sect.  cr_dimen is
        max_num_cr or 1, whichever is larger.
        cr_var_sect is a 3-D ndarray, the variance of each cosmic ray
        amplitude.
    """

    cr_flagged = np.empty(data_sect.shape, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)

    # If a pixel is flagged as a jump in the first group, we can't fit to
    # the ramp, because a matrix that we need to invert would be singular.
    # If there's only one group, we can't fit a ramp to it anyway, so
    # at this point we wouldn't need to be concerned about a jump.  If
    # there is more than one group, just ignore any jump the first group.
    if data_sect.shape[0] > 1:
        cr_flagged[0, :, :] = 0

    # Sum over groups to get an (ny, nx) image of the number of cosmic
    # rays in each pixel, accumulated over the ramp.
    sum_flagged = cr_flagged.sum(axis=0, dtype=np.int32)

    # If a pixel is flagged as saturated in the first or second group, we
    # don't want to even attempt to fit a slope to the ramp for that pixel.
    # Handle this case by setting the corresponding pixel in sum_flagged to
    # a negative number.  The test `ncr_mask = (sum_flagged == num_cr)`
    # will therefore never match, since num_cr is zero or larger, and the
    # pixel will not be included in any ncr_mask.
    mask1 = (gdq_sect[0, :, :] == saturated_flag)
    sum_flagged[mask1] = -1
    # one_group_mask flags pixels that are not saturated in the first
    # group but are saturated in the second group (if there is a second
    # group).  For these pixels, we will assign a value to the slope
    # image by just dividing the value in the first group by group_time.
    if len(gdq_sect) > 1:
        mask2 = (gdq_sect[1, :, :] == saturated_flag)
        sum_flagged[mask2] = -1
        one_group_mask = np.bitwise_and(mask2, np.bitwise_not(mask1))
        del mask2
    else:
        one_group_mask = np.bitwise_not(mask1)
    del mask1

    # Set elements of this array to a huge value if the corresponding
    # pixels are saturated.  This is not a flag, it's a value to be
    # added to the diagonal of the covariance matrix.
    saturated = np.empty(data_sect.shape, dtype=np.float64)
    saturated[:] = np.where(np.bitwise_and(gdq_sect, saturated_flag),
                            HUGE_FOR_LOW_WEIGHT, 0.)

    # Create arrays to be populated and then returned.
    shape = data_sect.shape
    # Lower limit of one, in case there are no cosmic rays at all.
    cr_dimen = max(1, max_num_cr)
    intercept_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    slope_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    cr_sect = np.zeros((shape[1], shape[2], cr_dimen),
                       dtype=data_sect.dtype)
    int_var_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    slope_var_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    cr_var_sect = np.zeros((shape[1], shape[2], cr_dimen),
                           dtype=data_sect.dtype)

    # This takes care of the case that there's only one group, as well as
    # pixels that are saturated in the second but not the first group of a
    # multi-group file
    if one_group_mask.any():
        slope_sect[one_group_mask] = data_sect[0, one_group_mask] / group_time
    del one_group_mask

    # Fit slopes for all pixels that have no cosmic ray hits anywhere in
    # the ramp, then fit slopes with one CR hit, then with two, etc.
    for num_cr in range(max_num_cr + 1):
        ngroups = len(data_sect)
        ncr_mask = (sum_flagged == num_cr)
        # Number of detector pixels flagged with num_cr CRs within the ramp.
        nz = ncr_mask.sum(dtype=np.int32)
        if nz <= 0:
            continue

        # ramp_data will be a ramp with a 1-D array of pixels copied out
        # of data_sect.
        ramp_data = np.empty((ngroups, nz), dtype=data_sect.dtype)
        input_var_data = np.empty((ngroups, nz), dtype=data_sect.dtype)
        prev_fit_data = np.empty((ngroups, nz), dtype=prev_fit.dtype)
        prev_slope_data = np.empty(nz, dtype=prev_slope_sect.dtype)
        prev_slope_data[:] = prev_slope_sect[ncr_mask]
        readnoise = np.empty(nz, dtype=readnoise_sect.dtype)
        readnoise[:] = readnoise_sect[ncr_mask]
        if gain_sect is None:
            gain = None
        else:
            gain = np.empty(nz, dtype=gain_sect.dtype)
            gain[:] = gain_sect[ncr_mask]
        cr_flagged_2d = np.empty((ngroups, nz), dtype=cr_flagged.dtype)
        saturated_data = np.empty((ngroups, nz), dtype=prev_fit.dtype)
        for k in range(ngroups):
            ramp_data[k] = data_sect[k][ncr_mask]
            input_var_data[k] = input_var_sect[k][ncr_mask]
            prev_fit_data[k] = prev_fit[k][ncr_mask]
            cr_flagged_2d[k] = cr_flagged[k][ncr_mask]
            # This is for clobbering saturated pixels.
            saturated_data[k] = saturated[k][ncr_mask]

        (result, variances) = \
            gls_fit(ramp_data,
                    prev_fit_data, prev_slope_data,
                    readnoise, gain,
                    frame_time, group_time, nframes_used,
                    num_cr, cr_flagged_2d, saturated_data)
        # Copy the intercept, slope, and cosmic-ray amplitudes and their
        # variances to the arrays to be returned.
        # ncr_mask is a mask array that is True for each pixel that has the
        # current number (num_cr) of cosmic rays.  Thus, the output arrays
        # are being populated here in sets, a different set of pixels with
        # each iteration of this loop.
        intercept_sect[ncr_mask] = result[:, 0].copy()
        int_var_sect[ncr_mask] = variances[:, 0].copy()
        slope_sect[ncr_mask] = result[:, 1].copy()
        slope_var_sect[ncr_mask] = variances[:, 1].copy()
        # In this loop, i is just an index.  cr_sect is populated for
        # number of cosmic rays = 1 to num_cr, inclusive.
        for i in range(num_cr):
            cr_sect[ncr_mask, i] = result[:, 2 + i].copy()
            cr_var_sect[ncr_mask, i] = variances[:, 2 + i].copy()

    return (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
            cr_sect, cr_var_sect)


def gls_fit(ramp_data,
            prev_fit_data, prev_slope_data,
            readnoise, gain,
            frame_time, group_time, nframes_used,
            num_cr, cr_flagged_2d, saturated_data):
    """Generalized least squares linear fit.

    It is assumed that every input pixel has num_cr cosmic-ray hits
    somewhere within the ramp.  This function should be called separately
    for different values of num_cr.

    Notes
    -----
    Curently the noise model is assumed to be a combination of
    read and photon noise alone.
    Same technique could be used with more complex noise models, but then
    the ramp covariance matrix should be input.

    Parameters
    ----------
    ramp_data: 2-D ndarray; indices:  group, pixel number
        The ramp data for one of the integrations in an exposure.  This
        may be a subset in detector coordinates, but covering all groups.
        The shape is (ngroups, nz), where ngroups is the length of the
        ramp, and nz is the number of pixels in the current subset.

    prev_fit_data: 2-D ndarray, shape (ngroups, nz)
        The fit to ramp_data, based on applying the values of intercept,
        slope, and cosmic-ray amplitudes that were determined in a previous
        call to gls_fit.  This array is only used for setting up the
        covariance matrix.

    prev_slope_data: 1-D ndarray, length nz.
        An estimate (e.g. from a previous iteration) of the slope at each
        pixel, in electrons per second.

    readnoise: 1-D ndarray, length nz.
        The read noise in electrons at each detector pixel.

    gain: 1-D ndarray, shape (nz,)
        The analog-to-digital gain (electrons per dn) at each detector
        pixel.

    frame_time: float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time: float
        Time increment between groups, in seconds.

    nframes_used: int
        Number of frames that were averaged together to make a group.
        Note that this value does not include the number (if any) of
        skipped frames.

    num_cr: int
        The number of cosmic rays that will be handled.  All pixels in the
        current set (ramp_data) are assumed to have this many cosmic ray
        hits somewhere within the ramp.

    cr_flagged_2d: 2-D ndarray, shape (ngroups, nz)
        The values should be 0 or 1; 1 indicates that a cosmic ray was
        detected (by another step) at that point.

    saturated_data: 2-D ndarray, shape (ngroups, nz)
        Normal values are zero; the value will be a huge number for
        saturated pixels.  This will be added to the main diagonal of the
        inverse of the weight matrix to greatly reduce the weight for
        saturated pixels.

    Returns
    -------
    tuple:  (result2d, variances)
        result2d is a 2-D ndarray; shape (nz, 2 + num_cr)
        The computed values of intercept, slope, and cosmic-ray amplitudes
        (there will be num_cr cosmic-ray amplitudes) for each of the nz
        pixels.

        variances is a 2-D ndarray; shape (nz, 2 + num_cr)
        The variance for the intercept, slope, and for the amplitude of
        each cosmic ray that was detected.
    """

    M = float(nframes_used)

    ngroups = ramp_data.shape[0]
    nz = ramp_data.shape[1]
    num_cr = int(num_cr)

    # x is an array (length nz) of matrices, each of which is the
    # independent variable of a linear equation.  Each such matrix
    # has ngroups rows and 2 + num_cr columns.  The first column is set
    # to 1, for finding the intercept.  The second column is the time at
    # each group, for finding the slope.  The remaining columns (if any),
    # are 0 for all rows prior to a certain point, then 1 for all
    # subsequent rows (i.e. the Heaviside function).  The transition from
    # 0 to 1 is the location of a cosmic ray hit; the first 1 in a column
    # corresponds to the value in cr_flagged_2d being 1.
    x = np.zeros((nz, ngroups, 2 + num_cr), dtype=np.float64)
    x[:, :, 0] = 1.
    x[:, :, 1] = np.arange(ngroups, dtype=np.float64) * group_time + \
        frame_time * (M + 1.) / 2.

    if num_cr > 0:
        sum_crs = cr_flagged_2d.cumsum(axis=0)
        for k in range(ngroups):
            s = slice(k, ngroups)
            for n in range(1, num_cr + 1):
                temp = np.where(np.logical_and(cr_flagged_2d[k] == 1,
                                               sum_crs[k] == n))
                if len(temp[0]) > 0:
                    index = (temp[0], s, n + 1)
                    x[index] = 1
        del temp, index

    y = np.transpose(ramp_data, (1, 0)).reshape((nz, ngroups, 1))

    # ramp_cov is an array of nz matrices, each ngroups x ngroups.
    # each matrix gives the covariance of that pixel's ramp data
    ramp_cov = np.ones((nz, ngroups, ngroups), dtype=np.float64)

    # Use the previous fit to the data to populate the covariance matrix,
    # for each of the nz pixels.  prev_fit_data has shape (ngroups, nz),
    # similar to the ramp data, but we want the nz axis to be the first
    # (we're constructing an array of nz matrix equations), so transpose
    # prev_fit_data.
    prev_fit_T = np.transpose(prev_fit_data, (1, 0))
    for k in range(ngroups):
        # Populate the upper right, row by row.
        ramp_cov[:, k, k:ngroups] = prev_fit_T[:, k:k + 1]
        # Populate the lower left, column by column.
        ramp_cov[:, k:ngroups, k] = prev_fit_T[:, k:k + 1]
        # Give saturated pixels a very high high variance (hence a low weight)
        ramp_cov[:, k, k] += saturated_data[k, :]
    del prev_fit_T

    # iden is 2-D, but it can broadcast to 4-D.  This is used to add terms to
    # the diagonal of the covariance matrix.
    iden = np.identity(ngroups)

    rn3d = readnoise.reshape((nz, 1, 1))
    ramp_cov += (iden * rn3d**2)

    # prev_slope_data must be non-negative.
    flags = prev_slope_data < 0.
    prev_slope_data[flags] = 1.

    # The resulting fit parameters are
    #  (xT @ ramp_cov^-1 @ x)^-1 @ [xT @ ramp_cov^-1 @ y]
    #  = [y-intercept, slope, cr_amplitude_1, cr_amplitude_2, ...]
    # where @ means matrix multiplication.

    # shape of xT is (nz, 2 + num_cr, ngroups)
    xT = np.transpose(x, (0, 2, 1))

    # shape of `ramp_invcov` is (nz, ngroups, ngroups)
    iden = iden.reshape((1, ngroups, ngroups))
    ramp_invcov = la.solve(ramp_cov, iden)

    del iden

    # temp1 = xT @ ramp_invcov
    # np.einsum use is equivalent to matrix multiplication
    # shape of temp1 is (nz, 2 + num_cr, ngroups)
    temp1 = np.einsum('...ij,...jk->...ik', xT, ramp_invcov)

    # temp_var = xT @ ramp_invcov @ x
    # shape of temp_var is (nz, 2 + num_cr, 2 + num_cr)
    temp_var = np.einsum('...ij,...jk->...ik', temp1, x)

    # `fitparam_cov` is an array of nz covariance matrices.
    # fitparam_cov = (xT @ ramp_invcov @ x)^-1
    # shape of fitparam_covar is (nz, 2 + num_cr, 2 + num_cr)
    I_2 = np.eye(2 + num_cr).reshape((1, 2 + num_cr, 2 + num_cr))
    try:
        # inverse of temp_var
        fitparam_cov = la.solve(temp_var, I_2)
    except la.LinAlgError:
        # find the pixel with the singular matrix
        for z in range(nz):
            try:
                la.solve(temp_var[z], I_2)
            except la.LinAlgError as msg2:
                log.warning("singular matrix, z = %d" % z)
                raise la.LinAlgError(msg2)
    del I_2

    # [xT @ ramp_invcov @ y]
    # shape of temp2 is (nz, 2 + num_cr, 1)
    temp2 = np.einsum('...ij,...jk->...ik', temp1, y)

    # shape of fitparam is (nz, 2 + num_cr, 1)
    fitparam = np.einsum('...ij,...jk->...ik', fitparam_cov, temp2)
    r_shape = fitparam.shape
    fitparam2d = fitparam.reshape((r_shape[0], r_shape[1]))
    del fitparam

    # shape of both result2d and variances is (nz, 2 + num_cr)
    fitparam_uncs = fitparam_cov.diagonal(axis1=1, axis2=2).copy()

    return (fitparam2d, fitparam_uncs)
'''
