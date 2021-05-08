#! /usr/bin/env python


import logging
# from multiprocessing.pool import Pool as Pool
import numpy as np
import time

import warnings

from ..datamodels import dqflags  # TODO here for flags

from . import utils

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# TODO Change dependencies when moved to STCAL
DO_NOT_USE = dqflags.group['DO_NOT_USE']
JUMP_DET = dqflags.group['JUMP_DET']
SATURATED = dqflags.group['SATURATED']
UNRELIABLE_SLOPE = dqflags.pixel['UNRELIABLE_SLOPE']

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def ols_ramp_fit_multi(
        input_model, buffsize, save_opt, readnoise_2d, gain_2d, weighting, max_cores):
    """
    Setup the inputs to ols_ramp_fit with and without multiprocessing. The
    inputs will be sliced into the number of cores that are being used for
    multiprocessing. Because the data models cannot be pickled, only numpy
    arrays are passed and returned as parameters to ols_ramp_fit.

    Parameters
    ----------
    input_model : data model
        input data model, assumed to be of type RampModel

    buffsize : int
        size of data section (buffer) in bytes (not used)

    save_opt : boolean
       calculate optional fitting results

    readnoise_2d : instance of data Model
        readnoise for all pixels

    gain_2d : instance of gain model
        gain for all pixels

    algorithm : string
        'OLS' specifies that ordinary least squares should be used;
        'GLS' specifies that generalized least squares should be used.

    weighting : string
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : string
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.

    gls_opt_model : GLS_RampFitModel object or None
        Object containing optional GLS-specific ramp fitting data for the
        exposure
    """

    # Determine number of slices to use for multi-processor computations
    number_slices = utils.compute_slices(max_cores)

    # Copy the int_times table for TSO data
    int_times = input_model.int_times

    # total_rows = input_model.data.shape[2]
    # total_cols = input_model.data.shape[3]
    number_of_integrations = input_model.data.shape[0]

    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    #   the input model arrays will be resized appropriately. If all pixels in
    #   all groups are flagged, return None for the models.
    if input_model.meta.instrument.name == 'MIRI' and input_model.data.shape[1] > 1:
        miri_ans = discard_miri_groups(input_model)
        # The function returns False if the removed groups leaves no data to be
        # processed.  If this is the case, return None for all expected variables
        # returned by ramp_fit
        if miri_ans is not True:
            return [None] * 3

    # Call ramp fitting for the single processor (1 data slice) case
    if number_slices == 1:
        max_segments, max_CRs = calc_num_seg(input_model.groupdq, number_of_integrations)
        log.debug(f"Max segments={max_segments}")

        # Single threaded computation
        image_info, integ_info, opt_info = ols_ramp_fit_single(
            input_model, int_times, buffsize, save_opt, readnoise_2d, gain_2d, weighting)
        if image_info is None:
            return None, None, None

        return image_info, integ_info, opt_info

    # Call ramp fitting for multi-processor (multiple data slices) case
    else:
        return None, None, None


'''
# Remove BEGIN multiprocessing
        log.debug(f'number of processes being used is {number_slices}')
        rows_per_slice = round(total_rows / number_slices)
        pool = Pool(processes=number_slices)
        slices = []

        # Populate the first n-1 slices
        for i in range(number_slices - 1):
            start_row = i * rows_per_slice
            stop_row = (i + 1) * rows_per_slice
            readnoise_slice = readnoise_2d[start_row: stop_row, :]
            gain_slice = gain_2d[start_row: stop_row, :]
            data_slice = input_model.data[:, :, start_row: stop_row, :].copy()
            err_slice = input_model.err[:, :, start_row: stop_row, :].copy()
            groupdq_slice = input_model.groupdq[:, :, start_row:stop_row, :].copy()
            pixeldq_slice = input_model.pixeldq[start_row:stop_row, :].copy()

            slices.insert(
                i,
                (data_slice, err_slice, groupdq_slice, pixeldq_slice, buffsize, save_opt,
                 readnoise_slice, gain_slice, weighting,
                 input_model.meta.instrument.name, input_model.meta.exposure.frame_time,
                 input_model.meta.exposure.ngroups, input_model.meta.exposure.group_time,
                 input_model.meta.exposure.groupgap, input_model.meta.exposure.nframes,
                 input_model.meta.exposure.drop_frames1, int_times))

        # last slice gets the rest
        start_row = (number_slices - 1) * rows_per_slice
        readnoise_slice = readnoise_2d[start_row: total_rows, :]
        gain_slice = gain_2d[start_row: total_rows, :]
        data_slice = input_model.data[:, :, start_row: total_rows, :].copy()
        err_slice = input_model.err[:, :, start_row: total_rows, :].copy()
        groupdq_slice = input_model.groupdq[:, :, start_row: total_rows, :].copy()
        pixeldq_slice = input_model.pixeldq[start_row: total_rows, :].copy()
        slices.insert(number_slices - 1,
                      (data_slice, err_slice, groupdq_slice, pixeldq_slice, buffsize, save_opt,
                       readnoise_slice, gain_slice, weighting,
                       input_model.meta.instrument.name, input_model.meta.exposure.frame_time,
                       input_model.meta.exposure.ngroups, input_model.meta.exposure.group_time,
                       input_model.meta.exposure.groupgap, input_model.meta.exposure.nframes,
                       input_model.meta.exposure.drop_frames1, int_times))

        # Start up the processes for each slice
        log.debug("Creating %d processes for ramp fitting " % number_slices)

        # Use starmap because input is iterable as well.
        real_results = pool.starmap(ols_ramp_fit_sliced, slices)
        pool.close()
        pool.join()
        k = 0
        log.debug("All processes complete")

        # Check that all slices got processed properly
        for resultslice in real_results:
            if resultslice[0] is None:
                return None, None, None

        # Create new model for the primary output.
        actual_segments = real_results[0][20]
        actual_CRs = real_results[0][21]
        int_model, opt_model, out_model = \
            create_output_models(input_model, number_of_integrations, save_opt,
                                 total_cols, total_rows, actual_segments, actual_CRs)
        int_model.int_times = int_times

        # iterate over the number of slices and place the results into the output models
        for resultslice in real_results:
            start_row = k * rows_per_slice
            if len(real_results) == k + 1:  # last result
                out_model.data[start_row: total_rows, :] = resultslice[0]
                out_model.dq[start_row:total_rows, :] = resultslice[1]
                out_model.var_poisson[start_row:total_rows, :] = resultslice[2]
                out_model.var_rnoise[start_row:total_rows, :] = resultslice[3]
                out_model.err[start_row:total_rows, :] = resultslice[4]
                if resultslice[5] is not None:  # Integration results exist
                    int_model.data[:, start_row:total_rows, :] = resultslice[5]
                    int_model.dq[:, start_row:total_rows, :] = resultslice[6]
                    int_model.var_poisson[:, start_row:total_rows, :] = resultslice[7]
                    int_model.var_rnoise[:, start_row:total_rows, :] = resultslice[8]
                    int_model.err[:, start_row:total_rows, :] = resultslice[9]
                if resultslice[11] is not None:  # Optional results exist
                    opt_model.slope[:, :, start_row:total_rows, :] = resultslice[11]
                    opt_model.sigslope[:, :, start_row:total_rows, :] = resultslice[12]
                    opt_model.var_poisson[:, :, start_row:total_rows, :] = resultslice[13]
                    opt_model.var_rnoise[:, :, start_row:total_rows, :] = resultslice[14]
                    opt_model.yint[:, :, start_row:total_rows, :] = resultslice[15]
                    opt_model.sigyint[:, :, start_row:total_rows, :] = resultslice[16]
                    opt_model.pedestal[:, start_row:total_rows, :] = resultslice[17]
                    opt_model.weights[:, :, start_row:total_rows, :] = resultslice[18]
                    opt_model.crmag[:, :, start_row:total_rows, :] = resultslice[19]
            else:  # all but last slice
                stop_row = (k + 1) * rows_per_slice
                out_model.data[start_row: stop_row, :] = resultslice[0]
                out_model.dq[start_row: stop_row, :] = resultslice[1]
                out_model.var_poisson[start_row: stop_row, :] = resultslice[2]
                out_model.var_rnoise[start_row: stop_row, :] = resultslice[3]
                out_model.err[start_row: stop_row, :] = resultslice[4]
                if resultslice[5] is not None:  # Multiple integration results exist
                    int_model.data[:, start_row: stop_row, :] = resultslice[5]
                    int_model.dq[:, start_row: stop_row, :] = resultslice[6]
                    int_model.var_poisson[:, start_row: stop_row, :] = resultslice[7]
                    int_model.var_rnoise[:, start_row: stop_row, :] = resultslice[8]
                    int_model.err[:, start_row: stop_row, :] = resultslice[9]
                if resultslice[11] is not None:  # Optional Results exist
                    opt_model.slope[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[11]
                    opt_model.sigslope[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[12]
                    opt_model.var_poisson[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[13]
                    opt_model.var_rnoise[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[14]
                    opt_model.yint[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[15]
                    opt_model.sigyint[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[16]
                    opt_model.pedestal[:, start_row: (k + 1) * rows_per_slice, :] = resultslice[17]
                    opt_model.weights[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[18]
                    opt_model.crmag[:, :, start_row: (k + 1) * rows_per_slice, :] = resultslice[19]
            k = k + 1

        return out_model, int_model, opt_model


def create_output_models(input_model, number_of_integrations, save_opt,
                         total_cols, total_rows, actual_segments, actual_CRs):
    """
    Create_output_models is used to make blank output models to hold the results from the OLS
    ramp fitting.

    Parameters
       ----------
    input_model : DataModel
        The input ramp model
    number_of_integrations : int
        The number of integration in the input model
    save_opt : Boolean
        Whether to save the optional outputs
    total_cols : int
        The number of columns in the input image
    total_rows : int
        The number of rows in the input image
    actual_segments : int
        The largest number of segments in the integration resulting from cosmic rays
    actual_CRs : int
        The largest number of cosmic rays jumps found in any integration
    Returns
    ------------
    int_model : DataModel
        The per integration output model
    opt_model : DataModel
        The optional output model
    out_model : RampFitOutputModel
        The standard rate output model
    """
    # TODO Remove function
    imshape = (total_rows, total_cols)
    out_model = datamodels.ImageModel(data=np.zeros(imshape, dtype=np.float32),
                                      dq=np.zeros(imshape, dtype=np.uint32),
                                      var_poisson=np.zeros(imshape, dtype=np.float32),
                                      var_rnoise=np.zeros(imshape, dtype=np.float32),
                                      err=np.zeros(imshape, dtype=np.float32))
    # ... and add all keys from input
    out_model.update(input_model)

    # create per integrations model
    # TODO Remove function
    int_model = datamodels.CubeModel(
        data=np.zeros((number_of_integrations,) + imshape, dtype=np.float32),
        dq=np.zeros((number_of_integrations,) + imshape, dtype=np.uint32),
        var_poisson=np.zeros((number_of_integrations,) + imshape, dtype=np.float32),
        var_rnoise=np.zeros((number_of_integrations,) + imshape, dtype=np.float32),
        err=np.zeros((number_of_integrations,) + imshape, dtype=np.float32))
    int_model.int_times = None
    int_model.update(input_model)  # ... and add all keys from input

    # Create model for the optional output
    # TODO Remove function
    if save_opt:
        opt_model = datamodels.RampFitOutputModel(
            slope=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            yint=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            sigyint=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            sigslope=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            weights=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            firstf_int=np.zeros((number_of_integrations,) + imshape, dtype=np.float32),
            pedestal=np.zeros((number_of_integrations,) + imshape, dtype=np.float32),
            crmag=np.zeros((number_of_integrations,) + (actual_CRs,) + imshape, dtype=np.float32),
            var_poisson=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
            var_rnoise=np.zeros((number_of_integrations,) + (actual_segments,) + imshape, dtype=np.float32),
        )

        opt_model.meta.filename = input_model.meta.filename
        opt_model.update(input_model)  # ... and add all keys from input
    else:
        opt_model = None

    return int_model, opt_model, out_model


def ols_ramp_fit_sliced(
        data, err, groupdq, inpixeldq, buffsize, save_opt, readnoise_2d, gain_2d,
        weighting, instrume, frame_time, ngroups, group_time, groupgap, nframes,
        dropframes1, int_times):

    """
    Fit a ramp using ordinary least squares. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    slope for all sections (intervals between cosmic rays) of the pixel's ramp
    divided by the effective integration time.  This function wraps the single
    threaded ols_ramp_fit_single function and is used in order to properly handle
    the slicing of the data for multiprocessing.

    Parameters
    ----------
    data : The input 4-D array with ramp data (num_integrations, num_groups, num_rows, num_cols)
        The input ramp data

    err : ndarray
        The input 4-D error that matches the ramp data

    groupdq : ndarray
        The input 4-D group DQ flags

    inpixeldq : ndarray
        The input 2-D pixel DQ flags

    buffsize : int
        The working buffer size

    save_opt : Boolean
            Whether to return the optional output model

    readnoise_2d : 2D float32
        The read noise of each pixel

    gain_2d : 2D float32
        The gain of each pixel

    weighting : string
        'optimal' is the only valid value

    instrume : string
        Instrument name

    frame_time : float32
        The time to read one frame.

    ngroups : int
        The number of groups in each integration

    group_time : float32
        The time to read one group.

    groupgap : int
        The number of frames that are not included in the group average

    nframes : int
        The number of frames that are included in the group average

    dropframes1 :
        The number of frames dropped at the beginning of every integration

    int_times : None
        Not used

    Returns
    -------
    new_model.data : 2-D float32
        The output final rate of each pixel

    new_model.dq : 2-D DQflag
        The output pixel dq for each pixel

    new_model.var_poisson : 2-D float32
        The variance in each pixel due to Poisson noise

    new_model.var_rnoise : 2-D float32
        The variance in each pixel due to read noise

    new_model.err : 2-D float32
        The output total variance for each pixel

    int_data : 3-D float32
        The rate for each pixel in each integration

    int_dq : 3-D float32
        The pixel dq flag for each integration

    int_var_poisson : 3-D float32
        The variance of the rate for each integration due to Poisson noise

    int_var_rnoise : 3-D float32
        The variance of the rate for each integration due to read noise

    int_err : 3-D float32
        The total variance of the rate for each integration

    int_int_times : 3-D
        The total time for each integration

    opt_slope : 4-D float32
        The rate of each segment in each integration

    opt_sigslope : 4-D float32
        The total variance of the rate for each pixel in each segment of each integration

    opt_var_poisson : 4-D float32
        The Poisson variance of the rate for each pixel in each segment of each integration

    opt_var_rnoise : 4-D float32
        The read noise variance of the rate for each pixel in each segment of each integration

    opt_yint : 4-D float32
        The y-intercept for each pixel in each segment of each integration

    opt_sigyint : 4-D float32
        The variance for each pixel in each segment of each integration

    opt_pedestal : 4-D float32
        The zero point for each pixel in each segment of each integration

    opt_weights : 4-D float32
        The weight of each pixel to use in combining the segments

    opt_crmag : 4-D float32
        The magnitude of each CR in each integration

    actual_segments : int
        The actual maximum number of segments in any integration

    actual_CRs : int
        The actual maximum number of CRs in any integration
    """
    # Package image data into a RampModel
    input_model = RampModel()

    input_model.data = data
    input_model.err = err
    input_model.groupdq = groupdq
    input_model.pixeldq = inpixeldq

    # Capture exposure and instrument data into the RampModel
    input_model.meta.instrument.name = instrume

    input_model.meta.exposure.frame_time = frame_time
    input_model.meta.exposure.ngroups = ngroups
    input_model.meta.exposure.group_time = group_time
    input_model.meta.exposure.groupgap = groupgap
    input_model.meta.exposure.nframes = nframes
    input_model.meta.exposure.drop_frames1 = dropframes1

    # Compute ramp fitting
    new_model, int_model, opt_res = ols_ramp_fit_single(
        input_model, int_times, buffsize, save_opt, readnoise_2d, gain_2d, weighting)

    if new_model is None:
        return [None] * 22

    # Package computed data for return
    if int_model is not None:
        int_data = int_model.data.copy()
        int_dq = int_model.dq.copy()
        int_var_poisson = int_model.var_poisson.copy()
        int_var_rnoise = int_model.var_rnoise.copy()
        int_err = int_model.err.copy()
        int_int_times = int_model.int_times.copy()
    else:
        int_data = None
        int_dq = None
        int_var_poisson = None
        int_var_rnoise = None
        int_err = None
        int_int_times = None

    if opt_res is not None:
        opt_slope = opt_res.slope.copy()
        opt_sigslope = opt_res.sigslope.copy()
        opt_var_poisson = opt_res.var_poisson.copy()
        opt_var_rnoise = opt_res.var_rnoise.copy()
        opt_yint = opt_res.yint.copy()
        opt_sigyint = opt_res.sigyint.copy()
        opt_pedestal = opt_res.pedestal.copy()
        opt_weights = opt_res.weights.copy()
        opt_crmag = opt_res.crmag.copy()
        actual_segments = opt_slope.shape[1]
        actual_CRs = opt_crmag.shape[1]
    else:
        opt_slope = None
        opt_sigslope = None
        opt_var_poisson = None
        opt_var_rnoise = None
        opt_yint = None
        opt_sigyint = None
        opt_pedestal = None
        opt_weights = None
        opt_crmag = None
        actual_segments = 0
        actual_CRs = 0

    return new_model.data, new_model.dq, new_model.var_poisson, \
        new_model.var_rnoise, new_model.err, \
        int_data, int_dq, int_var_poisson, int_var_rnoise, int_err, int_int_times, \
        opt_slope, opt_sigslope, opt_var_poisson, opt_var_rnoise, opt_yint, opt_sigyint, \
        opt_pedestal, opt_weights, opt_crmag, actual_segments, actual_CRs
# END Multiprocessing
'''


def ols_ramp_fit_single(
        input_model, int_times, buffsize, save_opt, readnoise_2d, gain_2d, weighting):
    """
    Fit a ramp using ordinary least squares. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    slope for all sections (intervals between cosmic rays) of the pixel's ramp
    divided by the effective integration time.

    Parameter
    ---------
    input_model: RampModel

    int_times : None
        Not used

    buffsize : int
        The working buffer size

    save_opt : Boolean
            Whether to return the optional output model

    readnoise_2d : 2D float32
        The read noise of each pixel

    gain_2d : 2D float32
        The gain of each pixel

    weighting : string
        'optimal' is the only valid value

    Return
    ------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.
    """
    tstart = time.time()

    # Save original shapes for writing to log file, as these may change for MIRI
    n_int, ngroups, nrows, ncols = input_model.data.shape
    orig_ngroups = ngroups
    orig_cubeshape = (ngroups, nrows, ncols)

    if ngroups == 1:
        log.warning('Dataset has NGROUPS=1, so count rates for each integration')
        log.warning('will be calculated as the value of that 1 group divided by')
        log.warning('the group exposure time.')

    # In this 'First Pass' over the data, loop over integrations and data
    #   sections to calculate the estimated median slopes, which will be used
    #   to calculate the variances. This is the same method to estimate slopes
    #   as is done in the jump detection step, except here CR-affected and
    #   saturated groups have already been flagged. The actual, fit, slopes for
    #   each segment are also calculated here.
    fit_slopes_ans = ramp_fit_slopes(
        input_model, gain_2d, readnoise_2d, save_opt, weighting)
    if fit_slopes_ans[0] == "saturated":
        return fit_slopes_ans[1:]

    # In this 'Second Pass' over the data, loop over integrations and data
    #   sections to calculate the variances of the slope using the estimated
    #   median slopes from the 'First Pass'. These variances are due to Poisson
    #   noise only, read noise only, and the combination of Poisson noise and
    #   read noise. The integration-specific variances are 3D arrays, and the
    #   segment-specific variances are 4D arrays.
    variances_ans = \
        ramp_fit_compute_variances(input_model, gain_2d, readnoise_2d, fit_slopes_ans)

    # Now that the segment-specific and integration-specific variances have
    #   been calculated, the segment-specific, integration-specific, and
    #   overall slopes will be calculated. The integration-specific slope is
    #   calculated as a weighted average of the segments in the integration:
    #     slope_int = sum_over_segs(slope_seg/var_seg)/ sum_over_segs(1/var_seg)
    #  The overall slope is calculated as a weighted average of the segments in
    #     all integrations:
    #     slope = sum_over_integs_and_segs(slope_seg/var_seg)/
    #                    sum_over_integs_and_segs(1/var_seg)
    image_info, integ_info, opt_info = ramp_fit_overall(
        input_model, orig_cubeshape, orig_ngroups, buffsize, fit_slopes_ans,
        variances_ans, save_opt, int_times, tstart)

    return image_info, integ_info, opt_info


def discard_miri_groups(input_model):
    """
    For MIRI datasets having >1 group, if all pixels in the final group are
    flagged as DO_NOT_USE, resize the input model arrays to exclude the
    final group.  Similarly, if leading groups 1 though N have all pixels
    flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    the input model arrays will be resized appropriately. If all pixels in
    all groups are flagged, return None for the models.

    Parameters
    ----------
    input_model: RampModel
        The input model containing the image data.

    Returns
    -------
    boolean:
        False if no data to process after discarding unusable data.
        True if useable data available for further processing.
    """
    data = input_model.data
    err = input_model.err
    groupdq = input_model.groupdq

    n_int, ngroups, nrows, ncols = data.shape

    num_bad_slices = 0  # number of initial groups that are all DO_NOT_USE

    while np.all(np.bitwise_and(groupdq[:, 0, :, :], DO_NOT_USE)):
        num_bad_slices += 1
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error('1. All groups have all pixels flagged as DO_NOT_USE,')
            log.error('  so will not process this dataset.')
            return False

        groupdq = groupdq[:, 1:, :, :]

        # Where the initial group of the just-truncated data is a cosmic ray,
        #   remove the JUMP_DET flag from the group dq for those pixels so
        #   that those groups will be included in the fit.
        wh_cr = np.where(np.bitwise_and(groupdq[:, 0, :, :], JUMP_DET))
        num_cr_1st = len(wh_cr[0])

        for ii in range(num_cr_1st):
            groupdq[wh_cr[0][ii], 0, wh_cr[1][ii], wh_cr[2][ii]] -= JUMP_DET

    if num_bad_slices > 0:
        data = data[:, num_bad_slices:, :, :]
        err = err[:, num_bad_slices:, :, :]

    log.info('Number of leading groups that are flagged as DO_NOT_USE: %s', num_bad_slices)

    # If all groups were flagged, the final group would have been picked up
    #   in the while loop above, ngroups would have been set to 0, and Nones
    #   would have been returned.  If execution has gotten here, there must
    #   be at least 1 remaining group that is not all flagged.
    if np.all(np.bitwise_and(groupdq[:, -1, :, :], DO_NOT_USE)):
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error('2. All groups have all pixels flagged as DO_NOT_USE,')
            log.error('  so will not process this dataset.')
            return False

        data = data[:, :-1, :, :]
        err = err[:, :-1, :, :]
        groupdq = groupdq[:, :-1, :, :]

        log.info('MIRI dataset has all pixels in the final group flagged as DO_NOT_USE.')

    # Next block is to satisfy github issue 1681:
    # "MIRI FirstFrame and LastFrame minimum number of groups"
    if ngroups < 2:
        log.warning('MIRI datasets require at least 2 groups/integration')
        log.warning('(NGROUPS), so will not process this dataset.')
        return False

    input_model.data = data
    input_model.err = err
    input_model.groupdq = groupdq
    return True


def ramp_fit_slopes(input_model, gain_2d, readnoise_2d, save_opt, weighting):
    """
    Calculate effective integration time (once EFFINTIM has been populated accessible, will
    use that instead), and other keywords that will needed if the pedestal calculation is
    requested. Note 'nframes' is the number of given by the NFRAMES keyword, and is the
    number of frames averaged on-board for a group, i.e., it does not include the groupgap.

    Parameter
    ---------
    input_model: RampModel
        The input model containing the image data.

    gain_2d : instance of gain model
        gain for all pixels

    readnoise_2d : instance of data Model
        readnoise for all pixels

    save_opt : boolean
       calculate optional fitting results

    weighting : string
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    Return
    ------
    max_seg : int
        Maximum possible number of segments over all groups and segments

    gdq_cube_shape : ndarray
        Group DQ dimensions

    effintim : float
        effective integration time for a single group

    f_max_seg : int
        Actual maximum number of segments over all groups and segments

    dq_int : ndarray
        The pixel dq for each integration for each pixel

    sum_weight : ndarray
        The sum of the weights for each pixel

    num_seg_per_int : integer, 3D array
        Cube of numbers of segments for all integrations and pixels

    sat_0th_group_int : uint8, 3D array
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated

    opt_res : OptRes
        Object to hold optional results for all good pixels.

    pixeldq : ndarray
        The input 2-D pixel DQ flags

    inv_var : float, 1D array
        values of 1/variance for good pixels

    med_rates : ndarray
        Rate array
    """

    # Get image data information
    data = input_model.data
    err = input_model.err
    groupdq = input_model.groupdq
    inpixeldq = input_model.pixeldq

    # Get instrument and exposure data
    frame_time = input_model.meta.exposure.frame_time
    group_time = input_model.meta.exposure.group_time
    groupgap = input_model.meta.exposure.groupgap
    nframes = input_model.meta.exposure.nframes

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)
    cubeshape = (ngroups,) + imshape

    # If all the pixels have their initial groups flagged as saturated, the DQ
    #   in the primary and integration-specific output products are updated,
    #   the other arrays in all output products are populated with zeros, and
    #   the output products are returned to ramp_fit(). If the initial group of
    #   a ramp is saturated, it is assumed that all groups are saturated.
    first_gdq = groupdq[:, 0, :, :]
    if np.all(np.bitwise_and(first_gdq, SATURATED)):
        image_info, integ_info, opt_info = utils.do_all_sat(
            inpixeldq, groupdq, imshape, n_int, save_opt)

        return "saturated", image_info, integ_info, opt_info

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim = (nframes + groupgap) * frame_time

    # Get GROUP DQ and ERR arrays from input file
    gdq_cube = groupdq
    gdq_cube_shape = gdq_cube.shape

    # Get max number of segments fit in all integrations
    max_seg, num_CRs = calc_num_seg(gdq_cube, n_int)
    del gdq_cube

    f_max_seg = 0  # final number to use, usually overwritten by actual value

    dq_int, median_diffs_2d, num_seg_per_int, sat_0th_group_int =\
        utils.alloc_arrays_1(n_int, imshape)

    opt_res = utils.OptRes(n_int, imshape, max_seg, ngroups, save_opt)

    # Get Pixel DQ array from input file. The incoming RampModel has uint32
    #   PIXELDQ, but ramp fitting will update this array here by flagging
    #   the 2D PIXELDQ locations where the ramp data has been previously
    #   flagged as jump-detected or saturated. These additional bit values
    #   require this local variable to be uint16, and it will be used as the
    #   (uint16) PIXELDQ in the outgoing ImageModel.
    pixeldq = inpixeldq.copy()
    pixeldq = utils.reset_bad_gain(pixeldq, gain_2d)  # Flag bad pixels in gain

    # In this 'First Pass' over the data, loop over integrations and data
    #   sections to calculate the estimated median slopes, which will be used
    #   to calculate the variances. This is the same method to estimate slopes
    #   as is done in the jump detection step, except here CR-affected and
    #   saturated groups have already been flagged. The actual, fit, slopes for
    #   each segment are also calculated here.

    # Loop over data integrations:
    for num_int in range(0, n_int):
        # Loop over data sections
        for rlo in range(0, cubeshape[1], nrows):
            rhi = rlo + nrows

            if rhi > cubeshape[1]:
                rhi = cubeshape[1]

            # Skip data section if it is all NaNs
            data_sect = np.float32(data[num_int, :, :, :])
            if np.all(np.isnan(data_sect)):
                log.error('Current data section is all nans, so not processing the section.')
                continue

            # first frame section for 1st group of current integration
            ff_sect = data[num_int, 0, rlo:rhi, :]

            # Get appropriate sections
            gdq_sect = groupdq[num_int, :, :, :]
            rn_sect = readnoise_2d[rlo:rhi, :]
            gain_sect = gain_2d[rlo:rhi, :]

            # Reset all saturated groups in the input data array to NaN
            where_sat = np.where(np.bitwise_and(gdq_sect, SATURATED))

            data_sect[where_sat] = np.NaN
            del where_sat

            # Compute the first differences of all groups
            first_diffs_sect = np.diff(data_sect, axis=0)

            # If the dataset has only 1 group/integ, assume the 'previous group'
            #   is all zeros, so just use data as the difference
            if first_diffs_sect.shape[0] == 0:
                first_diffs_sect = data_sect.copy()
            else:
                # Similarly, for datasets having >1 group/integ and having
                #   single-group segments, just use the data as the difference
                wh_nan = np.where(np.isnan(first_diffs_sect[0, :, :]))

                if len(wh_nan[0]) > 0:
                    first_diffs_sect[0, :, :][wh_nan] = data_sect[0, :, :][wh_nan]

                del wh_nan

                # Mask all the first differences that are affected by a CR,
                #   starting at group 1.  The purpose of starting at index 1 is
                #   to shift all the indices down by 1, so they line up with the
                #   indices in first_diffs.
                i_group, i_yy, i_xx, = np.where(np.bitwise_and(gdq_sect[1:, :, :], JUMP_DET))
                first_diffs_sect[i_group - 1, i_yy, i_xx] = np.NaN

                del i_group, i_yy, i_xx

                # Check for pixels in which there is good data in 0th group, but
                #   all first_diffs for this ramp are NaN because there are too
                #   few good groups past the 0th. Due to the shortage of good
                #   data, the first_diffs will be set here equal to the data in
                #   the 0th group.
                wh_min = np.where(np.logical_and(
                    np.isnan(first_diffs_sect).all(axis=0), np.isfinite(data_sect[0, :, :])))
                if len(wh_min[0] > 0):
                    first_diffs_sect[0, :, :][wh_min] = data_sect[0, :, :][wh_min]

                del wh_min

            # All first differences affected by saturation and CRs have been set
            #  to NaN, so compute the median of all non-NaN first differences.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN.*", RuntimeWarning)
                nan_med = np.nanmedian(first_diffs_sect, axis=0)
            nan_med[np.isnan(nan_med)] = 0.  # if all first_diffs_sect are nans
            median_diffs_2d[rlo:rhi, :] += nan_med

            # Calculate the slope of each segment
            # note that the name "opt_res", which stands for "optional results",
            # is deceiving; this in fact contains all the per-integration and
            # per-segment results that will eventually be used to compute the
            # final slopes, sigmas, etc. for the main (non-optional) products
            t_dq_cube, inv_var, opt_res, f_max_seg, num_seg = \
                calc_slope(data_sect, gdq_sect, frame_time, opt_res, save_opt, rn_sect,
                           gain_sect, max_seg, ngroups, weighting, f_max_seg)

            del gain_sect

            # Populate 3D num_seg { integ, y, x } with 2D num_seg for this data
            #  section (y,x) and integration (num_int)
            sect_shape = data_sect.shape[-2:]
            num_seg_per_int[num_int, rlo:rhi, :] = num_seg.reshape(sect_shape)

            # Populate integ-spec slice which is set if 0th group has SAT
            wh_sat0 = np.where(np.bitwise_and(gdq_sect[0, :, :], SATURATED))
            if len(wh_sat0[0]) > 0:
                sat_0th_group_int[num_int, rlo:rhi, :][wh_sat0] = 1

            del wh_sat0

            pixeldq_sect = pixeldq[rlo:rhi, :].copy()
            dq_int[num_int, rlo:rhi, :] = utils.dq_compress_sect(t_dq_cube, pixeldq_sect).copy()

            del t_dq_cube

            # Loop over the segments and copy the reshaped 2D segment-specific
            #   results for the current data section to the 4D output arrays.
            opt_res.reshape_res(num_int, rlo, rhi, sect_shape, ff_sect, save_opt)

            if save_opt:
                # Calculate difference between each slice and the previous slice
                #   as approximation to cosmic ray amplitude for those pixels
                #   having their DQ set for cosmic rays
                data_diff = data_sect - utils.shift_z(data_sect, -1)
                dq_cr = np.bitwise_and(JUMP_DET, gdq_sect)

                opt_res.cr_mag_seg[num_int, :, rlo:rhi, :] = data_diff * (dq_cr != 0)

                del data_diff

            del data_sect
            del ff_sect
            del gdq_sect

    if pixeldq_sect is not None:
        del pixeldq_sect

    # Compute the final 2D array of differences; create rate array
    median_diffs_2d /= n_int
    med_rates = median_diffs_2d / group_time

    del median_diffs_2d
    del first_diffs_sect

    input_model.data = data
    input_model.err = err
    input_model.groupdq = groupdq
    input_model.pixeldq = inpixeldq

    return max_seg, gdq_cube_shape, effintim, f_max_seg, dq_int, num_seg_per_int,\
        sat_0th_group_int, opt_res, pixeldq, inv_var, med_rates


def ramp_fit_compute_variances(input_model, gain_2d, readnoise_2d, fit_slopes_ans):
    """
    In this 'Second Pass' over the data, loop over integrations and data
    sections to calculate the variances of the slope using the estimated
    median slopes from the 'First Pass'. These variances are due to Poisson
    noise only, read noise only, and the combination of Poisson noise and
    read noise. The integration-specific variances are 3D arrays, and the
    segment-specific variances are 4D arrays.

    The naming convention for the arrays:
        'var': a variance
        'p3': intermediate 3D array for variance due to Poisson noise
        'r4': intermediate 4D array for variance due to read noise
        'both4': intermediate 4D array for combined variance due to both Poisson and read noise
        'inv_<X>': intermediate array = 1/<X>
        's_inv_<X>': intermediate array = 1/<X>, summed over integrations


    Parameters
    ----------
    input_model: RampModel
        The input model containing the image data.

    gain_2d : instance of gain model
        gain for all pixels

    readnoise_2d : 2-D float32
        The read noise for each pixel

    fit_slopes_ans : tuple
        Contains intermediate values computed in the first pass over the data.

    Return
    ------
    var_p3 : ndarray
        3-D variance based on Poisson noise

    var_r3 : ndarray
        3-D variance based on read noise

    var_p4 : ndarray
        4-D variance based on Poisson noise

    var_r4 : ndarray
        4-D variance based on read noise

    var_both4 : ndarray
        4-D array for combined variance due to both Poisson and read noise

    var_both3 : ndarray
        3-D array for combined variance due to both Poisson and read noise

    inv_var_both4 : ndarray
        1 / var_both4

    s_inv_var_p3 : ndarray
        1 / var_p3, summed over integrations

    s_inv_var_r3 : ndarray
        1 / var_r3, summed over integrations

    s_inv_var_both3 : ndarray
        1 / var_both3, summed over integrations
    """

    # Get image data information
    data = input_model.data
    err = input_model.err
    groupdq = input_model.groupdq
    inpixeldq = input_model.pixeldq

    # Get instrument and exposure data
    group_time = input_model.meta.exposure.group_time

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)
    cubeshape = (ngroups,) + imshape

    max_seg = fit_slopes_ans[0]
    num_seg_per_int = fit_slopes_ans[5]
    med_rates = fit_slopes_ans[10]

    var_p3, var_r3, var_p4, var_r4, var_both4, var_both3, \
        inv_var_both4, s_inv_var_p3, s_inv_var_r3, s_inv_var_both3, segs_4 = \
        utils.alloc_arrays_2(n_int, imshape, max_seg)

    # Loop over data integrations
    for num_int in range(n_int):

        # Loop over data sections
        for rlo in range(0, cubeshape[1], nrows):
            rhi = rlo + nrows

            if rhi > cubeshape[1]:
                rhi = cubeshape[1]

            gdq_sect = groupdq[num_int, :, rlo:rhi, :]
            rn_sect = readnoise_2d[rlo:rhi, :]
            gain_sect = gain_2d[rlo:rhi, :]

            # Calculate results needed to compute the variance arrays
            den_r3, den_p3, num_r3, segs_beg_3 = \
                utils.calc_slope_vars(rn_sect, gain_sect, gdq_sect, group_time, max_seg)

            segs_4[num_int, :, rlo:rhi, :] = segs_beg_3

            # Suppress harmless arithmetic warnings for now
            warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
            var_p4[num_int, :, rlo:rhi, :] = den_p3 * med_rates[rlo:rhi, :]

            # Find the segment variance due to read noise and convert back to DN
            var_r4[num_int, :, rlo:rhi, :] = num_r3 * den_r3 / gain_sect**2

            # Reset the warnings filter to its original state
            warnings.resetwarnings()

            del den_r3, den_p3, num_r3, segs_beg_3
            del gain_sect
            del gdq_sect

        # The next 4 statements zero out entries for non-existing segments, and
        #   set the variances for segments having negative slopes (the segment
        #   variance is proportional to the median estimated slope) to
        #   outrageously large values so that they will have negligible
        #   contributions.
        var_p4[num_int, :, :, :] *= (segs_4[num_int, :, :, :] > 0)

        # Suppress, then re-enable harmless arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        var_p4[var_p4 <= 0.] = utils.LARGE_VARIANCE

        var_r4[num_int, :, :, :] *= (segs_4[num_int, :, :, :] > 0)
        var_r4[var_r4 <= 0.] = utils.LARGE_VARIANCE

        # The sums of inverses of the variances are needed for later
        #   variance calculations.
        s_inv_var_p3[num_int, :, :] = (1. / var_p4[num_int, :, :, :]).sum(axis=0)
        var_p3[num_int, :, :] = 1. / s_inv_var_p3[num_int, :, :]
        s_inv_var_r3[num_int, :, :] = (1. / var_r4[num_int, :, :, :]).sum(axis=0)
        var_r3[num_int, :, :] = 1. / s_inv_var_r3[num_int, :, :]

        # Huge variances correspond to non-existing segments, so are reset to 0
        #  to nullify their contribution.
        var_p3[var_p3 > 0.1 * utils.LARGE_VARIANCE] = 0.
        warnings.resetwarnings()

        var_both4[num_int, :, :, :] = var_r4[num_int, :, :, :] + var_p4[num_int, :, :, :]
        inv_var_both4[num_int, :, :, :] = 1. / var_both4[num_int, :, :, :]

        # Want to retain values in the 4D arrays only for the segments that each
        #   pixel has, so will zero out values for the higher indices. Creating
        #   and manipulating intermediate arrays (views, such as var_p4_int
        #   will zero out the appropriate indices in var_p4 and var_r4.)
        # Extract the slice of 4D arrays for the current integration
        var_p4_int = var_p4[num_int, :, :, :]   # [ segment, y, x ]
        inv_var_both4_int = inv_var_both4[num_int, :, :, :]

        # Zero out non-existing segments
        var_p4_int *= (segs_4[num_int, :, :, :] > 0)
        inv_var_both4_int *= (segs_4[num_int, :, :, :] > 0)

        # reshape these arrays to simplify masking [ segment, 1D pixel ]
        var_p4_int2 = var_p4_int.reshape(
            (var_p4_int.shape[0], var_p4_int.shape[1] * var_p4_int.shape[2]))

        s_inv_var_both3[num_int, :, :] = (inv_var_both4[num_int, :, :, :]).sum(axis=0)

        # Suppress, then re-enable harmless arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        var_both3[num_int, :, :] = 1. / s_inv_var_both3[num_int, :, :]
        warnings.resetwarnings()

        del var_p4_int
        del var_p4_int2

    del gain_2d

    var_p4 *= (segs_4[:, :, :, :] > 0)  # Zero out non-existing segments
    var_r4 *= (segs_4[:, :, :, :] > 0)

    # Delete lots of arrays no longer needed
    if inv_var_both4_int is not None:
        del inv_var_both4_int

    if med_rates is not None:
        del med_rates

    if num_seg_per_int is not None:
        del num_seg_per_int

    if readnoise_2d is not None:
        del readnoise_2d

    if rn_sect is not None:
        del rn_sect

    if segs_4 is not None:
        del segs_4

    input_model.data = data
    input_model.err = err
    input_model.groupdq = groupdq
    input_model.pixeldq = inpixeldq

    return var_p3, var_r3, var_p4, var_r4, var_both4, var_both3, inv_var_both4, \
        s_inv_var_p3, s_inv_var_r3, s_inv_var_both3


def ramp_fit_overall(
        input_model, orig_cubeshape, orig_ngroups, buffsize, fit_slopes_ans,
        variances_ans, save_opt, int_times, tstart):
    """
    Parameter
    ---------
    input_model : data model
        input data model, assumed to be of type RampModel

    orig_cubeshape : (int, int, int) tuple
       Original shape of input dataset

    orig_ngroups: int
       Original number of groups

    buffsize : int
        Size of data section (buffer) in bytes

    fit_slopes_ans : tuple
        Return values from ramp_fit_slopes

    variances_ans : tuple
        Return values from ramp_fit_compute_variances

    save_opt : boolean
        Calculate optional fitting results.

    int_times : bintable, or None
        The INT_TIMES table, if it exists in the input, else None

    tstart : float
        Start time.

    Return
    ------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Get image data information
    data = input_model.data
    groupdq = input_model.groupdq

    # Get instrument and exposure data
    instrume = input_model.meta.instrument.name

    groupgap = input_model.meta.exposure.groupgap
    nframes = input_model.meta.exposure.nframes
    dropframes1 = input_model.meta.exposure.drop_frames1
    if dropframes1 is None:    # set to default if missing
        dropframes1 = 0
        log.debug('Missing keyword DRPFRMS1, so setting to default value of 0')

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)

    # Unpack intermediate computations from preious steps
    max_seg, gdq_cube_shape, effintim, f_max_seg, dq_int, num_seg_per_int = fit_slopes_ans[:6]
    sat_0th_group_int, opt_res, pixeldq, inv_var, med_rates = fit_slopes_ans[6:]

    var_p3, var_r3, var_p4, var_r4, var_both4, var_both3 = variances_ans[:6]
    inv_var_both4, s_inv_var_p3, s_inv_var_r3, s_inv_var_both3 = variances_ans[6:]

    slope_by_var4 = opt_res.slope_seg.copy() / var_both4

    del var_both4

    s_slope_by_var3 = slope_by_var4.sum(axis=1)  # sum over segments (not integs)
    s_slope_by_var2 = s_slope_by_var3.sum(axis=0)  # sum over integrations
    s_inv_var_both2 = s_inv_var_both3.sum(axis=0)

    # Compute the 'dataset-averaged' slope
    # Suppress, then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    slope_dataset2 = s_slope_by_var2 / s_inv_var_both2
    warnings.resetwarnings()

    del s_slope_by_var2, s_slope_by_var3, slope_by_var4
    del s_inv_var_both2, s_inv_var_both3

    #  Replace nans in slope_dataset2 with 0 (for non-existing segments)
    slope_dataset2[np.isnan(slope_dataset2)] = 0.

    # Compute the integration-specific slope
    the_num = (opt_res.slope_seg * inv_var_both4).sum(axis=1)

    the_den = (inv_var_both4).sum(axis=1)

    # Suppress, then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    slope_int = the_num / the_den
    warnings.resetwarnings()

    del the_num, the_den

    # Clean up ramps that are SAT on their initial groups; set ramp parameters
    #   for variances and slope so they will not contribute
    var_p3, var_both3, slope_int, dq_int = utils.fix_sat_ramps(
        sat_0th_group_int, var_p3, var_both3, slope_int, dq_int)

    if sat_0th_group_int is not None:
        del sat_0th_group_int

    # Loop over data integrations to calculate integration-specific pedestal
    if save_opt:
        dq_slice = np.zeros((gdq_cube_shape[2], gdq_cube_shape[3]), dtype=np.uint32)

        for num_int in range(0, n_int):
            dq_slice = groupdq[num_int, 0, :, :]
            opt_res.ped_int[num_int, :, :] = \
                utils.calc_pedestal(num_int, slope_int, opt_res.firstf_int,
                                    dq_slice, nframes, groupgap, dropframes1)

        del dq_slice

    # Collect optional results for output
    if save_opt:
        gdq_cube = groupdq
        opt_res.shrink_crmag(n_int, gdq_cube, imshape, ngroups)
        del gdq_cube

        # Some contributions to these vars may be NaN as they are from ramps
        # having PIXELDQ=DO_NOT_USE
        var_p4[np.isnan(var_p4)] = 0.
        var_r4[np.isnan(var_r4)] = 0.

        # Truncate results at the maximum number of segments found
        opt_res.slope_seg = opt_res.slope_seg[:, :f_max_seg, :, :]
        opt_res.sigslope_seg = opt_res.sigslope_seg[:, :f_max_seg, :, :]
        opt_res.yint_seg = opt_res.yint_seg[:, :f_max_seg, :, :]
        opt_res.sigyint_seg = opt_res.sigyint_seg[:, :f_max_seg, :, :]
        opt_res.weights = (inv_var_both4[:, :f_max_seg, :, :])**2.
        opt_res.var_p_seg = var_p4[:, :f_max_seg, :, :]
        opt_res.var_r_seg = var_r4[:, :f_max_seg, :, :]

        opt_info = opt_res.output_optional(effintim)
    else:
        opt_info = None

    if inv_var_both4 is not None:
        del inv_var_both4

    if var_p4 is not None:
        del var_p4

    if var_r4 is not None:
        del var_r4

    if inv_var is not None:
        del inv_var

    if pixeldq is not None:
        del pixeldq

    # Output integration-specific results to separate file
    integ_info = utils.output_integ(
        slope_int, dq_int, effintim, var_p3, var_r3, var_both3, int_times)

    if opt_res is not None:
        del opt_res

    if slope_int is not None:
        del slope_int
    del var_p3
    del var_r3
    del var_both3
    if int_times is not None:
        del int_times

    # Divide slopes by total (summed over all integrations) effective
    #   integration time to give count rates.
    c_rates = slope_dataset2 / effintim

    # Compress all integration's dq arrays to create 2D PIXELDDQ array for
    #   primary output
    final_pixeldq = utils.dq_compress_final(dq_int, n_int)

    if dq_int is not None:
        del dq_int

    tstop = time.time()

    utils.log_stats(c_rates)

    log.debug('Instrument: %s', instrume)
    log.debug('Number of pixels in 2D array: %d', nrows * ncols)
    log.debug('Shape of 2D image: (%d, %d)' % (imshape))
    log.debug('Shape of data cube: (%d, %d, %d)' % (orig_cubeshape))
    log.debug('Buffer size (bytes): %d', buffsize)
    log.debug('Number of rows per buffer: %d', nrows)
    log.info('Number of groups per integration: %d', orig_ngroups)
    log.info('Number of integrations: %d', n_int)
    log.debug('The execution time in seconds: %f', tstop - tstart)

    # Compute the 2D variances due to Poisson and read noise
    var_p2 = 1 / (s_inv_var_p3.sum(axis=0))
    var_r2 = 1 / (s_inv_var_r3.sum(axis=0))

    # Huge variances correspond to non-existing segments, so are reset to 0
    #  to nullify their contribution.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        var_p2[var_p2 > 0.1 * utils.LARGE_VARIANCE] = 0.
        var_r2[var_r2 > 0.1 * utils.LARGE_VARIANCE] = 0.

    # Some contributions to these vars may be NaN as they are from ramps
    # having PIXELDQ=DO_NOT_USE
    var_p2[np.isnan(var_p2)] = 0.
    var_r2[np.isnan(var_r2)] = 0.

    # Suppress, then re-enable, harmless arithmetic warning
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    err_tot = np.sqrt(var_p2 + var_r2)
    warnings.resetwarnings()

    del s_inv_var_p3
    del s_inv_var_r3

    # Create new model for the primary output.
    data = c_rates.astype(np.float32)
    dq = final_pixeldq.astype(np.uint32)
    var_poisson = var_p2.astype(np.float32)
    var_rnoise = var_r2.astype(np.float32)
    err = err_tot.astype(np.float32)
    image_info = (data, dq, var_poisson, var_rnoise, err)

    return image_info, integ_info, opt_info


def calc_power(snr):
    """
    Using the given SNR, calculate the weighting exponent, which is from
    `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C, Nieto,
    Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP, 112, 1350`.

    Parameters
    ----------
    snr : float32, 1D array
        signal-to-noise for the ramp segments

    Returns
    -------
    pow_wt.ravel() : float32, 1D array
        weighting exponent
    """
    pow_wt = snr.copy() * 0.0
    pow_wt[np.where(snr > 5.)] = 0.4
    pow_wt[np.where(snr > 10.)] = 1.0
    pow_wt[np.where(snr > 20.)] = 3.0
    pow_wt[np.where(snr > 50.)] = 6.0
    pow_wt[np.where(snr > 100.)] = 10.0

    return pow_wt.ravel()


def interpolate_power(snr):
    pow_wt = snr.copy() * 0.0
    pow_wt[np.where(snr > 5.)] = ((snr[snr > 5] - 5) / (10 - 5)) * 0.6 + 0.4
    pow_wt[np.where(snr > 10.)] = ((snr[snr > 10] - 10) / (20 - 10)) * 2.0 + 1.0
    pow_wt[np.where(snr > 20.)] = ((snr[snr > 20] - 20)) / (50 - 20) * 3.0 + 3.0
    pow_wt[np.where(snr > 50.)] = ((snr[snr > 50] - 50)) / (100 - 50) * 4.0 + 6.0
    pow_wt[np.where(snr > 100.)] = 10.0

    return pow_wt.ravel()


def calc_slope(data_sect, gdq_sect, frame_time, opt_res, save_opt, rn_sect,
               gain_sect, i_max_seg, ngroups, weighting, f_max_seg):
    """
    Compute the slope of each segment for each pixel in the data cube section
    for the current integration. Each segment has its slope fit in fit_lines();
    that slope and other quantities from the fit are added to the 'optional
    result' object by append_arr() from the appropriate 'CASE' (type of segment)
    in fit_next_segment().

    Parameters
    ----------
    data_sect : float, 3D array
        section of input data cube array

    gdq_sect : int, 3D array
        section of GROUPDQ data quality array

    frame_time : float
        integration time

    opt_res : OptRes object
        Contains all quantities derived from fitting all segments in all
        pixels in all integrations, which will eventually be used to compute
        per-integration and per-exposure quantities for all pixels. It's
        also used to populate the optional product, when requested.

    save_opt : boolean
       save optional fitting results

    rn_sect : float, 2D array
        read noise values for all pixels in data section

    gain_sect : float, 2D array
        gain values for all pixels in data section

    i_max_seg : int
        used for size of initial allocation of arrays for optional results;
        maximum possible number of segments within the ramp, based on the
        number of CR flags

    ngroups : int
        number of groups per integration

    weighting : string
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    f_max_seg : int
        actual maximum number of segments within a ramp, based on the fitting
        of all ramps; later used when truncating arrays before output.

    Returns
    -------
    gdq_sect : int, 3D array
        data quality flags for pixels in section

    inv_var : float, 1D array
        values of 1/variance for good pixels

    opt_res : OptRes object
        contains all quantities related to fitting for use in computing final
        slopes, variances, etc. and is used to populate the optional output

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : int, 1D array
        numbers of segments for good pixels
    """
    ngroups, nrows, ncols = data_sect.shape
    npix = nrows * ncols  # number of pixels in section of 2D array

    all_pix = np.arange(npix)
    arange_ngroups_col = np.arange(ngroups)[:, np.newaxis]
    start = np.zeros(npix, dtype=np.int32)  # lowest channel in fit

    # Highest channel in fit initialized to last read
    end = np.zeros(npix, dtype=np.int32) + (ngroups - 1)

    pixel_done = (end < 0)  # False until processing is done

    inv_var = np.zeros(npix, dtype=np.float32)  # inverse of fit variance
    num_seg = np.zeros(npix, dtype=np.int32)  # number of segments per pixel

    # End stack array - endpoints for each pixel
    # initialize with ngroups for each pixel; set 1st channel to 0
    end_st = np.zeros((ngroups + 1, npix), dtype=np.int32)
    end_st[0, :] = ngroups - 1

    # end_heads is initially a tuple populated with every pixel that is
    # either saturated or contains a cosmic ray based on the input DQ
    # array, so is sized to accomodate the maximum possible number of
    # pixels flagged. It is later compressed to be an array denoting
    # the number of endpoints per pixel.
    end_heads = np.ones(npix * ngroups, dtype=np.int32)

    # Create nominal 2D ERR array, which is 1st slice of
    #    avged_data_cube * readtime
    err_2d_array = data_sect[0, :, :] * frame_time

    # Suppress, then re-enable, harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    err_2d_array[err_2d_array < 0] = 0
    warnings.resetwarnings()

    # Frames >= start and <= end will be masked. However, the first channel
    #   to be included in fit will be the read in which a cosmic ray has
    #   been flagged
    mask_2d = ((arange_ngroups_col >= start[np.newaxis, :]) &
               (arange_ngroups_col <= end[np.newaxis, :]))

    end = 0  # array no longer needed

    # Section of GROUPDQ dq section, excluding bad dq values in mask
    gdq_sect_r = np.reshape(gdq_sect, (ngroups, npix))
    mask_2d[gdq_sect_r != 0] = False  # saturated or CR-affected
    mask_2d_init = mask_2d.copy()  # initial flags for entire ramp

    wh_f = np.where(np.logical_not(mask_2d))

    these_p = wh_f[1]  # coordinates of pixels flagged as False
    these_r = wh_f[0]  # reads of pixels flagged as False

    del wh_f

    # Populate end_st to contain the set of end points for each pixel.
    # Populate end_heads to initially include every pixel that is either
    # saturated or contains a cosmic ray. Skips the duplicated final group
    # for saturated pixels. Saturated pixels resulting in a contiguous set
    # of intervals of length 1 will later be flagged as too short
    # to fit well.
    for ii, val in enumerate(these_p):
        if these_r[ii] != (ngroups - 1):
            end_st[end_heads[these_p[ii]], these_p[ii]] = these_r[ii]
            end_heads[these_p[ii]] += 1

    # Sort and reverse array to handle the order that saturated pixels
    # were added
    end_st.sort(axis=0)
    end_st = end_st[::-1]

    # Reformat to designate the number of endpoints per pixel; compress
    # to specify number of groups per pixel
    end_heads = (end_st > 0).sum(axis=0)

    # Create object to hold optional results
    opt_res.init_2d(npix, i_max_seg, save_opt)

    # LS fit until 'ngroups' iterations or all pixels in
    #    section have been processed
    for iter_num in range(ngroups):
        if pixel_done.all():
            break

        # frames >= start and <= end_st will be included in fit
        mask_2d = \
            ((arange_ngroups_col >= start)
             & (arange_ngroups_col < (end_st[end_heads[all_pix] - 1, all_pix] + 1)))

        mask_2d[gdq_sect_r != 0] = False  # RE-exclude bad group dq values

        # for all pixels, update arrays, summing slope and variance
        f_max_seg, num_seg = \
            fit_next_segment(start, end_st, end_heads, pixel_done, data_sect, mask_2d,
                             mask_2d_init, inv_var, num_seg, opt_res, save_opt, rn_sect,
                             gain_sect, ngroups, weighting, f_max_seg)

        if f_max_seg is None:
            f_max_seg = 1

    arange_ngroups_col = 0
    all_pix = 0

    return gdq_sect, inv_var, opt_res, f_max_seg, num_seg


def fit_next_segment(start, end_st, end_heads, pixel_done, data_sect, mask_2d,
                     mask_2d_init, inv_var, num_seg, opt_res, save_opt, rn_sect,
                     gain_sect, ngroups, weighting, f_max_seg):
    """
    Call routine to LS fit masked data for a single segment for all pixels in
    data section. Then categorize each pixel's fitting interval based on
    interval length, and whether the interval is at the end of the array.
    Update the start array, the end stack array, the end_heads array which
    contains the number of endpoints. For pixels in which the fitting intervals
    are long enough, the resulting slope and variance are added to the
    appropriate stack arrays.  The first channel to fit in a segment is either
    the first group in the ramp, or a group in which a cosmic ray has been
    flagged.

    Parameters
    ----------
    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    data_sect : float, 3D array
        data cube section

    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    rn_sect : float, 2D array
        read noise values for all pixels in data section

    gain_sect : float, 2D array
        gain values for all pixels in data section

    ngroups : int
        number of groups per integration

    weighting : string
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : int, 1D array
        numbers of segments for good pixels
    """
    ngroups, nrows, ncols = data_sect.shape
    all_pix = np.arange(nrows * ncols)

    ramp_mask_sum = mask_2d_init.sum(axis=0)

    # Compute fit quantities for the next segment of all pixels
    # Each returned array below is 1D, for all npix pixels for current segment
    slope, intercept, variance, sig_intercept, sig_slope = \
        fit_lines(data_sect, mask_2d, rn_sect, gain_sect, ngroups, weighting)

    end_locs = end_st[end_heads[all_pix] - 1, all_pix]

    # Set the fitting interval length; for a segment having >1 groups, this is
    #   the number of groups-1
    l_interval = end_locs - start

    wh_done = (start == -1)  # done pixels
    l_interval[wh_done] = 0  # set interval lengths for done pixels to 0

    # Create array to set when each good pixel is classified for the current
    #   semiramp (to enable unclassified pixels to have their arrays updated)
    got_case = np.zeros((ncols * nrows), dtype=bool)

    # Special case fit with NGROUPS being 1 or 2.
    if ngroups == 1 or ngroups == 2:
        return fit_short_ngroups(
            ngroups, start, end_st, end_heads, pixel_done, all_pix,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, ramp_mask_sum)

    # CASE: Long enough (semiramp has >2 groups), at end of ramp
    wh_check = np.where((l_interval > 1) & (end_locs == ngroups - 1) & (~pixel_done))
    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_long_end_of_ramp(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt)

    # CASE: Long enough (semiramp has >2 groups ), not at array end (meaning
    #  final group for this semiramp is not final group of the whole ramp)
    wh_check = np.where((l_interval > 2) & (end_locs != ngroups - 1) & ~pixel_done)
    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_long_not_end_of_ramp(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, end_locs, ngroups)

    # CASE: interval too short to fit normally (only 2 good groups)
    #    At end of array, NGROUPS>1, but exclude NGROUPS==2 datasets
    #    as they are covered in `fit_short_ngroups`.
    wh_check = np.where((l_interval == 1) & (end_locs == ngroups - 1)
                        & (ngroups > 2) & (~pixel_done))

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_short_seg_at_end(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init)

    # CASE: full-length ramp has 2 good groups not at array end
    wh_check = np.where((l_interval == 2) & (ngroups > 2)
                        & (end_locs != ngroups - 1) & ~pixel_done)

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_short_seg_not_at_end(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, end_locs, ngroups)

    # CASE: full-length ramp has a good group on 0th group of the entire ramp,
    #    and no later good groups. Will use single good group data as the slope.
    wh_check = np.where(
        mask_2d_init[0, :] & ~mask_2d_init[1, :] & (ramp_mask_sum == 1) & ~pixel_done)

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_only_good_0th_group(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init)

    # CASE: the segment has a good 0th group and a bad 1st group.
    wh_check = np.where(mask_2d_init[0, :] & ~mask_2d_init[1, :] & ~pixel_done
                        & (end_locs == 1) & (start == 0))

    if len(wh_check[0]) > 0:
        fit_next_segment_good_0th_bad_1st(
            wh_check, start, end_st, end_heads, got_case, ngroups)

    # CASE OTHER: all other types of segments not covered earlier. No segments
    #   handled here have adequate data, but the stack arrays are updated.
    wh_check = np.asarray(np.where(~pixel_done & ~got_case))
    if len(wh_check[0]) > 0:
        fit_next_segment_all_other(wh_check, start, end_st, end_heads, ngroups)

    return f_max_seg, num_seg


def fit_next_segment_all_other(wh_check, start, end_st, end_heads, ngroups):
    """
    Catch all other types of segments not covered earlier. No segments
    handled here have adequate data, but the stack arrays are updated.
        - increment start array
        - remove current end from end stack
        - decrement number of ends

    Parameter
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    ngroups: int
        number of groups in exposure
    """
    these_pix = wh_check[0]
    start[these_pix] += 1
    start[start > ngroups - 1] = ngroups - 1  # to keep at max level
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.


def fit_next_segment_good_0th_bad_1st(
        wh_check, start, end_st, end_heads, got_case, ngroups):
    """
    The segment has a good 0th group and a bad 1st group. For the
    data from the 0th good group of this segment to possibly be used as a
    slope, that group must necessarily be the 0th group of the entire ramp.
    It is possible to have a single 'good' group segment after the 0th group
    of the ramp; in that case the 0th group and the 1st group would both have
    to be CRs, and the data of the 0th group would not be included as a slope.
    For a good 0th group in a ramp followed by a bad 1st group there must be
    good groups later in the segment because if there were not, the segment
    would be done in `fit_next_segment_only_good_0th_group`. In this situation,
    since here are later good groups in the segment, those later good groups
    will be used in the slope computation, and the 0th good group will not be.
    As a result, for all instances of these types of segments, the data in the
    initial good group will not be used in the slope calculation, but the
    arrays for the indices for the ramp (end_st, etc) are appropriately
    adjusted.
        - increment start array
        - remove current end from end stack
        - decrement number of ends

    Parameter
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    got_case: 1D array
        classification of pixel for current semiramp

    ngroups: int
        number of groups in exposure
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True
    start[these_pix] += 1
    start[start > ngroups - 1] = ngroups - 1  # to keep at max level
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.


def fit_next_segment_only_good_0th_group(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init):
    """
    Full-length ramp has a good group on 0th group of the entire ramp,
    and no later good groups. Will use single good group data as the slope.
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of end to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done

    Parameters
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    got_case: 1D array
        classification of pixel for current semiramp

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True

    start[these_pix] = -1
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] = 0
    pixel_done[these_pix] = True  # all processing for pixel is completed
    inv_var[these_pix] += 1.0 / variance[these_pix]

    # Append results to arrays
    opt_res.append_arr(num_seg, these_pix, intercept, slope,
                       sig_intercept, sig_slope, inv_var, save_opt)

    num_seg[these_pix] += 1
    f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_short_seg_not_at_end(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, end_locs, ngroups):
    """
    Special case
    Full-length ramp has 2 good groups not at array end
        - use the 2 good reads to get the slope
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of end to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done
    For segments of this type, the final good group in the segment is
    followed by a group that is flagged as a CR and/or SAT and is not the
    final group in the ramp, and the variable `l_interval` used below is
    equal to 2, which is the number of the segment's groups.

    Parameters
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    got_case: 1D array
        classification of pixel for current semiramp

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    end_locs: 1D array
        end locations

    ngroups: int
        number of groups in exposure

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    # Copy mask, as will modify when calculating the number of later good groups
    c_mask_2d_init = mask_2d_init.copy()

    these_pix = wh_check[0]
    got_case[these_pix] = True

    # Suppress, then re-enable, harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    inv_var[these_pix] += 1.0 / variance[these_pix]
    warnings.resetwarnings()

    # create array: 0...ngroups-1 in a column for each pixel
    arr_ind_all = np.array(
        [np.arange(ngroups), ] * c_mask_2d_init.shape[1]).transpose()
    wh_c_start_all = np.zeros(mask_2d_init.shape[1], dtype=np.uint8)
    wh_c_start_all[these_pix] = start[these_pix]

    # set to False all groups before start group
    c_mask_2d_init[arr_ind_all < wh_c_start_all] = 0
    tot_good_groups = c_mask_2d_init.sum(axis=0)

    # Select pixels having at least 2 later good groups (these later good
    #   groups are a segment whose slope will be calculated)
    wh_more = np.where(tot_good_groups[these_pix] > 1)
    pix_more = these_pix[wh_more]
    start[pix_more] = end_locs[pix_more]
    end_st[end_heads[pix_more] - 1, pix_more] = 0
    end_heads[pix_more] -= 1

    # Select pixels having less than 2 later good groups (these later good
    #   groups will not be used)
    wh_only = np.where(tot_good_groups[these_pix] <= 1)
    pix_only = these_pix[wh_only]
    start[pix_only] = -1
    end_st[end_heads[pix_only] - 1, pix_only] = 0
    end_heads[pix_only] = 0
    pixel_done[pix_only] = True  # all processing for pixel is completed
    end_heads[(end_heads < 0.)] = 0.

    # Append results to arrays
    opt_res.append_arr(num_seg, these_pix, intercept, slope,
                       sig_intercept, sig_slope, inv_var, save_opt)

    num_seg[these_pix] += 1
    f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_short_seg_at_end(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init):
    """
    Interval too short to fit normally (only 2 good groups)
    At end of array, NGROUPS>1, but exclude NGROUPS==2 datasets
    as they are covered in `fit_short_groups`.
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of ends to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done
    For segments of this type, the final good group is the final group in the
    ramp, and the variable `l_interval` used below = 1, and the number of
    groups in the segment = 2

    Parameters
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    got_case: 1D array
        classification of pixel for current semiramp

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    # Require that pixels to be processed here have at least 1 good group out
    #   of the final 2 groups (these ramps have 2 groups and are at the end of
    #   the array).
    wh_list = []

    num_wh = len(wh_check[0])
    for ii in range(num_wh):  # locate pixels with at least 1 good group
        this_pix = wh_check[0][ii]
        sum_final_2 = mask_2d_init[start[this_pix]:, this_pix].sum()

        if sum_final_2 > 0:
            wh_list.append(wh_check[0][ii])  # add to list to be fit

    if len(wh_list) > 0:
        these_pix = np.asarray(wh_list)
        got_case[these_pix] = True

        start[these_pix] = -1
        end_st[end_heads[these_pix] - 1, these_pix] = 0
        end_heads[these_pix] = 0
        pixel_done[these_pix] = True

        g_pix = these_pix[variance[these_pix] > 0.]  # good pixels

        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            # Append results to arrays
            opt_res.append_arr(num_seg, g_pix, intercept, slope,
                               sig_intercept, sig_slope, inv_var, save_opt)

            num_seg[g_pix] += 1
            f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_long_not_end_of_ramp(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, end_locs, ngroups):
    """
    Special case fitting long segment at the end of ramp.
    Long enough (semiramp has >2 groups ), not at array end (meaning
    final group for this semiramp is not final group of the whole ramp)
        - remove current end from end stack
        - decrement number of ends
        - add slopes and variances to running sums
    For segments of this type, the final good group in the segment is a CR
    and/or SAT and is not the final group in the ramp, and the variable
    `l_interval` used below is equal to the number of the segment's groups.

    Parameters
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    got_case: 1D array
        classification of pixel for current semiramp

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    end_locs: 1D array
        end locations

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    ngroups: int
        number of groups in exposure

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True

    start[these_pix] = end_locs[these_pix]
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.

    g_pix = these_pix[variance[these_pix] > 0.]  # good pixels

    if len(g_pix) > 0:
        inv_var[g_pix] += 1.0 / variance[g_pix]

        # Append results to arrays
        opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                           sig_slope, inv_var, save_opt)

        num_seg[g_pix] += 1
        f_max_seg = max(f_max_seg, num_seg.max())

        # If there are pixels with no later good groups, update stack
        #   arrays accordingly
        c_mask_2d_init = mask_2d_init.copy()

        # create array: 0...ngroups-1 in a column for each pixel
        arr_ind_all = np.array(
            [np.arange(ngroups), ] * c_mask_2d_init.shape[1]).transpose()

        wh_c_start_all = np.zeros(c_mask_2d_init.shape[1], dtype=np.uint8)
        wh_c_start_all[g_pix] = start[g_pix]

        # set to False all groups before start group
        c_mask_2d_init[arr_ind_all < wh_c_start_all] = False

        # select pixels having all groups False from start to ramp end
        wh_rest_false = np.where(c_mask_2d_init.sum(axis=0) == 0)
        if len(wh_rest_false[0]) > 0:
            pix_rest_false = wh_rest_false[0]
            start[pix_rest_false] = -1
            end_st[end_heads[pix_rest_false] - 1, pix_rest_false] = 0
            end_heads[pix_rest_false] = 0
            pixel_done[pix_rest_false] = True  # all processing is complete

    return f_max_seg


def fit_next_segment_long_end_of_ramp(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt):
    """
    Long enough (semiramp has >2 groups), at end of ramp
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of ends to 0
        - add slopes and variances to running sums
    For segments of this type, the final good group is the final group in the
    ramp, and the variable `l_interval` used below is equal to the number of
    the segment's groups minus 1.

    Parameters
    ----------
    wh_check: 1D array
        pixels for current segment processing and updating

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    got_case: 1D array
        classification of pixel for current semiramp

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    start[these_pix] = -1   # all processing for this pixel is completed
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] = 0
    pixel_done[these_pix] = True  # all processing for pixel is completed
    got_case[these_pix] = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        g_pix = these_pix[variance[these_pix] > 0.]  # good pixels
    if len(g_pix) > 0:
        inv_var[g_pix] += 1.0 / variance[g_pix]

        # Append results to arrays
        opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                           sig_slope, inv_var, save_opt)

        num_seg[g_pix] += 1
        f_max_seg = max(f_max_seg, num_seg.max())
    return f_max_seg


def fit_short_ngroups(
        ngroups, start, end_st, end_heads, pixel_done, all_pix,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, ramp_mask_sum):
    """
    Special case fitting for short ngroups fit.

    Parameters
    ----------
    ngroups: int
        number of groups in exposure

    start : int, 1D array
        lowest channel in fit

    end_st : int, 2D array
        stack array of endpoints

    end_heads : int, 1D array
        number of endpoints for each pixel

    pixel_done : boolean, 1D array
        whether each pixel's calculations are completed

    all_pix: 1D array
        all pixels in image

    inv_var : float, 1D array
        values of 1/variance for good pixels

    num_seg : int, 1D array
        numbers of segments for good pixels

    slope: float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept: float, 1D array
       y-intercepts from fit for data section

    variance: float, 1D array
       variance of residuals for fit for data section

    sig_intercept: float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope: float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : boolean
       save optional fitting results

    mask_2d_init : bool, 2D array
        copy of intial mask_2d

    ramp_mask_sum: int, 1D array
        number of channels to fit for each pixel

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : int, 1D array
        numbers of segments for good pixels
    """

    # Dataset has NGROUPS=2, so special fitting is done for all pixels.
    # All segments are at the end of the array.
    #    - set start to -1 to designate all fitting done
    #    - remove current end from end stack
    #    - set number of ends to 0
    #    - add slopes and variances to running sums
    #    - set pixel_done to True to designate all fitting done
    if ngroups == 2:
        start[all_pix] = -1
        end_st[end_heads[all_pix] - 1, all_pix] = 0
        end_heads[all_pix] = 0
        pixel_done[all_pix] = True

        g_pix = all_pix[variance[all_pix] > 0.]
        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                               sig_slope, inv_var, save_opt)

            num_seg[g_pix] = 1

            return 1, num_seg

    # Dataset has NGROUPS=1 ; so special fitting is done for all pixels
    # and all intervals are at the end of the array.
    #    - set start to -1 to designate all fitting done
    #    - remove current end from end stack
    #    - set number of ends to 0
    #    - add slopes and variances to running sums
    #    - set pixel_done to True to designate all fitting done
    start[all_pix] = -1
    end_st[end_heads[all_pix] - 1, all_pix] = 0
    end_heads[all_pix] = 0
    pixel_done[all_pix] = True

    wh_check = np.where(mask_2d_init[0, :] & (ramp_mask_sum == 1))
    if len(wh_check[0]) > 0:
        g_pix = wh_check[0]

        # Ignore all pixels having no good groups (so the single group is bad)
        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            # Append results to arrays
            opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                               sig_slope, inv_var, save_opt)

            num_seg[g_pix] = 1

    return 1, num_seg


def fit_lines(data, mask_2d, rn_sect, gain_sect, ngroups, weighting):
    """
    Do linear least squares fit to data cube in this integration for a single
    segment for all pixels.  In addition to applying the mask due to identified
    cosmic rays, the data is also masked to exclude intervals that are too short
    to fit well. The first channel to fit in a segment is either the first group
    in the ramp, or a group in which a cosmic ray has been flagged.

    Parameters
    ----------
    data : float, 3D array
       array of values for current data section

    mask_2d : boolean, 2D array
       delineates which channels to fit for each pixel

    rn_sect : float, 2D array
        read noise values for all pixels in data section

    gain_sect : float, 2D array
        gain values for all pixels in data section

    ngroups : int
        number of groups per integration

    weighting : string
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    Returns
    -------
    Note - all of these pertain to a single segment (hence '_s')

    slope_s : float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
       y-intercepts from fit for data section

    variance_s : float, 1D array
       variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
       sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
       sigma of slopes from fit for data section (for a single segment)

    """
    # To ensure that the first channel to be fit is the cosmic-ray-affected
    #   group, the channel previous to each channel masked as good is
    #   also masked as good. This is only for the local purpose of setting
    #   the first channel, and will not propagate beyond this current function
    #   call.
    c_mask_2d = mask_2d.copy()
    wh_mask_2d = np.where(c_mask_2d)
    c_mask_2d[np.maximum(wh_mask_2d[0] - 1, 0), wh_mask_2d[1]] = True

    del wh_mask_2d

    # num of reads/pixel unmasked
    nreads_1d = c_mask_2d.astype(np.int16).sum(axis=0)
    npix = c_mask_2d.shape[1]

    slope_s = np.zeros(npix, dtype=np.float32)
    variance_s = np.zeros(npix, dtype=np.float32)
    intercept_s = np.zeros(npix, dtype=np.float32)
    sig_intercept_s = np.zeros(npix, dtype=np.float32)
    sig_slope_s = np.zeros(npix, dtype=np.float32)

    # Calculate slopes etc. for datasets having either 1 or 2 groups per
    #   integration, and return
    if ngroups == 1:  # process all pixels in 1 group/integration dataset
        slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s = \
            fit_1_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                        sig_slope_s, npix, data, c_mask_2d)

        return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s

    if ngroups == 2:  # process all pixels in 2 group/integration dataset
        rn_sect_1d = rn_sect.reshape(npix)
        slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s = \
            fit_2_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                        sig_slope_s, npix, data, c_mask_2d, rn_sect_1d)

        return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s

    # reshape data_masked
    data_masked = data * np.reshape(c_mask_2d, data.shape)
    data_masked = np.reshape(data_masked, (data_masked.shape[0], npix))

    # For datasets having >2 groups/integration, for any semiramp in which the
    #   0th group is good and the 1st group is bad, determine whether or not to
    #   use the 0th group.
    wh_pix_1r = np.where(c_mask_2d[0, :] & (np.logical_not(c_mask_2d[1, :])))

    if len(wh_pix_1r[0]) > 0:
        slope_s, intercept_s, variance_s, sig_intercept_s, \
            sig_slope_s = fit_single_read(slope_s, intercept_s, variance_s,
                                          sig_intercept_s, sig_slope_s, npix,
                                          data, wh_pix_1r)

    del wh_pix_1r

    # For datasets having >2 groups/integrations, for any semiramp in which only
    #   the 0th and 1st group are good, set slope, etc
    wh_pix_2r = np.where(c_mask_2d.sum(axis=0) == 2)  # ramps with 2 good groups
    slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s = \
        fit_double_read(c_mask_2d, wh_pix_2r, data_masked, slope_s, intercept_s,
                        variance_s, sig_slope_s, sig_intercept_s, rn_sect)

    del wh_pix_2r

    # Select ramps having >2 good groups
    wh_pix_to_use = np.where(c_mask_2d.sum(axis=0) > 2)

    good_pix = wh_pix_to_use[0]  # Ramps with >2 good groups
    data_masked = data_masked[:, good_pix]

    del wh_pix_to_use

    xvalues = np.arange(data_masked.shape[0])[:, np.newaxis] * c_mask_2d
    xvalues = xvalues[:, good_pix]  # set to those pixels to be used

    c_mask_2d = c_mask_2d[:, good_pix]
    nreads_1d = nreads_1d[good_pix]

    if weighting.lower() == 'optimal':  # fit using optimal weighting
        # get sums from optimal weighting
        sumx, sumxx, sumxy, sumy, nreads_wtd, xvalues = \
            calc_opt_sums(rn_sect, gain_sect, data_masked, c_mask_2d, xvalues, good_pix)

        slope, intercept, sig_slope, sig_intercept = \
            calc_opt_fit(nreads_wtd, sumxx, sumx, sumxy, sumy)

        variance = sig_slope**2.  # variance due to fit values

    elif weighting.lower() == 'unweighted':  # fit using unweighted weighting
        # get sums from unweighted weighting
        sumx, sumxx, sumxy, sumy = calc_unwtd_sums(data_masked, xvalues)

        slope, intercept, sig_slope, sig_intercept, line_fit =\
            calc_unwtd_fit(xvalues, nreads_1d, sumxx, sumx, sumxy, sumy)

        denominator = nreads_1d * sumxx - sumx**2

        # In case this branch is ever used again, disable, and then re-enable
        #   harmless arithmetic warrnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        variance = nreads_1d / denominator
        warnings.resetwarnings()

        denominator = 0

    else:  # unsupported weighting type specified
        log.error('FATAL ERROR: unsupported weighting type specified.')

    slope_s[good_pix] = slope
    variance_s[good_pix] = variance
    intercept_s[good_pix] = intercept
    sig_intercept_s[good_pix] = sig_intercept
    sig_slope_s[good_pix] = sig_slope

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def fit_single_read(slope_s, intercept_s, variance_s, sig_intercept_s,
                    sig_slope_s, npix, data, wh_pix_1r):
    """
    For datasets having >2 groups/integrations, for any semiramp in which the
    0th group is good and the 1st group is either SAT or CR, set slope, etc.

    Parameters
    ----------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    npix : int
        number of pixels in 2D array

    data : float
        array of values for current data section

    wh_pix_1r : tuple
        locations of pixels whose only good group is the 0th group

    Returns
    -------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section
    """
    data0_slice = data[0, :, :].reshape(npix)
    slope_s[wh_pix_1r] = data0_slice[wh_pix_1r]

    # The following arrays will have values correctly calculated later; for
    #   now they are just place-holders
    variance_s[wh_pix_1r] = utils.LARGE_VARIANCE
    sig_slope_s[wh_pix_1r] = 0.
    intercept_s[wh_pix_1r] = 0.
    sig_intercept_s[wh_pix_1r] = 0.

    return slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s


def fit_double_read(mask_2d, wh_pix_2r, data_masked, slope_s, intercept_s,
                    variance_s, sig_slope_s, sig_intercept_s, rn_sect):
    """
    Process all semi-ramps having exactly 2 good groups. May need to optimize
    later to remove loop over pixels.

    Parameters
    ----------
    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel

    wh_pix_2r : tuple
        locations of pixels whose only good groups are the 0th and the 1st

    data_masked : float, 2D array
        masked values for all pixels in data section

    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    rn_sect : float, 2D array
        read noise values for all pixels in data section

    Returns
    -------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section
    """
    rn_sect_flattened = rn_sect.flatten()

    for ff in range(len(wh_pix_2r[0])):  # loop over the pixels
        pixel_ff = wh_pix_2r[0][ff]  # pixel index (1d)

        rn = rn_sect_flattened[pixel_ff]  # read noise for this pixel

        read_nums = np.where(mask_2d[:, pixel_ff])
        second_read = read_nums[0][1]
        data_ramp = data_masked[:, pixel_ff] * mask_2d[:, pixel_ff]
        data_semi = data_ramp[mask_2d[:, pixel_ff]]  # picks only the 2
        diff_data = data_semi[1] - data_semi[0]

        slope_s[pixel_ff] = diff_data
        intercept_s[pixel_ff] = \
            data_semi[1] * (1. - second_read) + data_semi[0] * second_read  # by geometry
        variance_s[pixel_ff] = 2.0 * rn * rn
        sig_slope_s[pixel_ff] = np.sqrt(2) * rn
        sig_intercept_s[pixel_ff] = np.sqrt(2) * rn

    return slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s


def calc_unwtd_fit(xvalues, nreads_1d, sumxx, sumx, sumxy, sumy):
    """
    Do linear least squares fit to data cube in this integration, using
    unweighted fits to the segments. Currently not supported.

    Parameters
    ----------
    xvalues : int, 1D array
        indices of valid pixel values for all groups

    nreads_1d : int, 1D array
        number of reads in an integration

    sumxx : float
        sum of squares of xvalues

    sumx : float
        sum of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data

    Returns
    -------
    slope : float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept : float, 1D array
       y-intercepts from fit for data section

    sig_slope : float, 1D array
       sigma of slopes from fit for data section

    sig_intercept : float, 1D array
       sigma of y-intercepts from fit for data section

    line_fit : float, 1D array
       values of fit using slope and intercept
    """

    denominator = nreads_1d * sumxx - sumx**2

    # In case this branch is ever used again, suppress, and then re-enable
    #   harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    slope = (nreads_1d * sumxy - sumx * sumy) / denominator
    intercept = (sumxx * sumy - sumx * sumxy) / denominator
    sig_intercept = (sumxx / denominator)**0.5
    sig_slope = (nreads_1d / denominator)**0.5
    warnings.resetwarnings()

    line_fit = (slope * xvalues) + intercept

    return slope, intercept, sig_slope, sig_intercept, line_fit


def calc_opt_fit(nreads_wtd, sumxx, sumx, sumxy, sumy):
    """
    Do linear least squares fit to data cube in this integration for a single
    semi-ramp for all pixels, using optimally weighted fits to the semi_ramps.
    The weighting uses the formulation by Fixsen (Fixsen et al, PASP, 112, 1350).
    Note - these weights, sigmas, and variances pertain only to the fitting, and
    the variances are *NOT* the variances of the slope due to noise.

    Parameters
    ----------
    nreads_wtd : float, 1D array
        sum of product of data and optimal weight

    sumxx : float, 1D array
        sum of squares of xvalues

    sumx : float, 1D array
        sum of xvalues

    sumxy : float, 1D array
        sum of product of xvalues and data

    sumy : float, 1D array
        sum of data

    Returns
    -------
    slope : float, 1D array
       weighted slope for current iteration's pixels for data section

    intercept : float, 1D array
       y-intercepts from fit for data section

    sig_slope : float, 1D array
       sigma of slopes from fit for data section

    sig_intercept : float, 1D array
       sigma of y-intercepts from fit for data section
    """
    denominator = nreads_wtd * sumxx - sumx**2

    # Suppress, and then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    slope = (nreads_wtd * sumxy - sumx * sumy) / denominator
    intercept = (sumxx * sumy - sumx * sumxy) / denominator
    sig_intercept = (sumxx / denominator)**0.5
    sig_slope = (nreads_wtd / denominator)**0.5  # STD of the slope's fit

    warnings.resetwarnings()

    return slope, intercept, sig_slope, sig_intercept


def fit_1_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                sig_slope_s, npix, data, mask_2d):
    """
    This function sets the fitting arrays for datasets having only 1 group
    per integration.

    Parameters
    ----------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    npix : int
        number of pixels in 2d array

    data : float
        array of values for current data section

    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel

    Returns
    -------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section
    """
    # For pixels not saturated, recalculate the slope as the value of the SCI
    #   data in that group, which will later be divided by the group exposure
    #   time to give the count rate. Recalculate other fit quantities to be
    #   benign.
    slope_s = data[0, :, :].reshape(npix)

    # The following arrays will have values correctly calculated later; for
    #    now they are just place-holders
    variance_s = np.zeros(npix, dtype=np.float32) + utils.LARGE_VARIANCE
    sig_slope_s = slope_s * 0.
    intercept_s = slope_s * 0.
    sig_intercept_s = slope_s * 0.

    # For saturated pixels, overwrite slope with benign values.
    wh_sat0 = np.where(np.logical_not(mask_2d[0, :]))

    if len(wh_sat0[0]) > 0:
        sat_pix = wh_sat0[0]
        slope_s[sat_pix] = 0.

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def fit_2_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                sig_slope_s, npix, data, mask_2d, rn_sect_1d):
    """
    This function sets the fitting arrays for datasets having only 2 groups
    per integration.

    Parameters
    ----------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section

    npix : int
        number of pixels in 2d array

    data : float
        array of values for current data section

    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel

    rn_sect_1d : float, 1D array
        read noise values for all pixels in data section

    Returns
    -------
    slope_s : float, 1D array
        weighted slope for current iteration's pixels for data section

    intercept_s : float, 1D array
        y-intercepts from fit for data section

    variance_s : float, 1D array
        variance of residuals for fit for data section

    sig_intercept_s : float, 1D array
        sigma of y-intercepts from fit for data section

    sig_slope_s : float, 1D array
        sigma of slopes from fit for data section
    """
    # For pixels saturated on the first group, overwrite fit values with
    # benign values to be recalculated later.
    wh_sat0 = np.where(np.logical_not(mask_2d[0, :]))

    if len(wh_sat0[0]) > 0:
        sat_pix = wh_sat0[0]
        slope_s[sat_pix] = 0.
        variance_s[sat_pix] = 0.
        sig_slope_s[sat_pix] = 0.
        intercept_s[sat_pix] = 0.
        sig_intercept_s[sat_pix] = 0.
    del wh_sat0

    # For pixels saturated on the second group, recalculate the slope as
    # the value of the SCI data in the first group, which will later be
    # divided by the group exposure time to give the count rate, and
    # recalculate the other fit quantities to be benign. Note: these pixels
    # will already have been handled earlier (for intervals of arbitrary
    # length) in this function, but are being included here to explicitly
    # cover all possibilities for pixels in datasets with ngroups=2. Will
    # later consider refactoring.
    wh_sat1 = np.where((mask_2d[:, :].sum(axis=0) == 1) & mask_2d[0, :])

    if len(wh_sat1[0]) > 0:
        data0_slice = data[0, :, :].reshape(npix)
        slope_s[wh_sat1] = data0_slice[wh_sat1]
        # set variance non-zero because calling function uses variance=0 to
        # throw out bad results; this is not bad
        variance_s[wh_sat1] = 1.
        sig_slope_s[wh_sat1] = 0.
        intercept_s[wh_sat1] = 0.
        sig_intercept_s[wh_sat1] = 0.
    del wh_sat1

    # For pixels with no saturated values, recalculate the slope as the
    # difference between the values of the second and first groups (1-based),
    # which will later be divided by the group exposure time to give the count
    # rate, and recalculate other fit quantities to be benign.
    wh_sat_no = np.where(mask_2d[:, :].sum(axis=0) == 2)

    if len(wh_sat_no[0]) > 0:
        data0_slice = data[0, :, :].reshape(npix)
        data1_slice = data[1, :, :].reshape(npix)
        slope_s[wh_sat_no] = data1_slice[wh_sat_no] - data0_slice[wh_sat_no]
        sig_slope_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]
        intercept_s[wh_sat_no] = data0_slice[wh_sat_no] -\
            data1_slice[wh_sat_no]  # by geometry
        sig_intercept_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]
        variance_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]

    del wh_sat_no

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def calc_num_seg(gdq, n_int):
    """
    Calculate the maximum number of segments that will be fit within an
    integration, calculated over all pixels and all integrations.  This value
    is based on the locations of cosmic ray-affected pixels in all of the ramps,
    and will be used to allocate arrays used for the optional output product.

    Parameters
    ----------
    gdq : float, 3D array
        cube of GROUPDQ array for a data

    n_int : int (unused)
        total number of integrations in data set

    Return:
    -------
    max_num_seg: int
        The maximum number of segements within an integration
    max_cr: int
        The maximum number of cosmic rays within an integration
    """
    max_cr = 0  # max number of CRS for all integrations

    # For all 2d pixels, get max number of CRs or DO_NOT_USE flags along their
    # ramps, to use as a surrogate for the number of segments along the ramps
    # Note that we only care about flags that are NOT in the first or last groups,
    # because exclusion of a first or last group won't result in an additional segment.
    max_cr = np.count_nonzero(np.bitwise_and(gdq[:, 1:-1], JUMP_DET | DO_NOT_USE), axis=1).max()

    # Do not want to return a value > the number of groups, which can occur if
    #  this is a MIRI dataset in which the first or last group was flagged as
    #  DO_NOT_USE and also flagged as a jump.
    max_num_seg = int(max_cr) + 1  # n CRS implies n+1 segments
    if max_num_seg > gdq.shape[1]:
        max_num_seg = gdq.shape[1]

    return max_num_seg, max_cr


def calc_unwtd_sums(data_masked, xvalues):
    """
    Calculate the sums needed to determine the slope and intercept (and sigma
    of each) using an unweighted fit. Unweighted fitting currently not
    supported.

    Parameters
    ----------
    data_masked : float, 2D array
        masked values for all pixels in data section

    xvalues : int, 1D array
        indices of valid pixel values for all groups

    Return:
    -------
    sumx : float
        sum of xvalues

    sumxx : float
        sum of squares of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data

    """
    sumx = xvalues.sum(axis=0)
    sumxx = (xvalues**2).sum(axis=0)
    sumy = (np.reshape(data_masked.sum(axis=0), sumx.shape))
    sumxy = (xvalues * np.reshape(data_masked, xvalues.shape)).sum(axis=0)

    return sumx, sumxx, sumxy, sumy


def calc_opt_sums(rn_sect, gain_sect, data_masked, mask_2d, xvalues, good_pix):
    """
    Calculate the sums needed to determine the slope and intercept (and sigma of
    each) using the optimal weights.  For each good pixel's segment, from the
    initial and final indices and the corresponding number of counts, calculate
    the SNR. From the SNR, calculate the weighting exponent using the formulation
    by Fixsen (Fixsen et al, PASP, 112, 1350). Using this exponent and the gain
    and the readnoise, the weights are calculated from which the sums are
    calculated.

    Parameters
    ----------
    rn_sect : float, 2D array
        read noise values for all pixels in data section

    gain_sect : float, 2D array
        gain values for all pixels in data section

    data_masked : float, 2D array
        masked values for all pixels in data section

    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel

    xvalues : int, 2D array
        indices of valid pixel values for all groups

    good_pix : int, 1D array
        indices of pixels having valid data for all groups

    Return:
    -------
    sumx : float
        sum of xvalues

    sumxx : float
        sum of squares of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data

    nreads_wtd : float, 1D array
        sum of optimal weights

    xvalues : int, 2D array
        rolled up indices of valid pixel values for all groups
    """
    c_mask_2d = mask_2d.copy()  # copy the mask to prevent propagation
    rn_sect = np.float32(rn_sect)

    # Return 'empty' sums if there is no more data to fit
    if data_masked.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([]), np.array([])

    # get initial group for each good pixel for this semiramp
    fnz = np.argmax(c_mask_2d, axis=0)

    # For those pixels that are all False, set to sentinel value of -1
    fnz[c_mask_2d.sum(axis=0) == 0] = -1

    mask_2d_sum = c_mask_2d.sum(axis=0)   # number of valid groups/pixel

    # get final valid group for each pixel for this semiramp
    ind_lastnz = fnz + mask_2d_sum - 1

    # get SCI value of initial good group for semiramp
    data_zero = data_masked[fnz, range(data_masked.shape[1])]

    # get SCI value of final good group for semiramp
    data_final = data_masked[(ind_lastnz), range(data_masked.shape[1])]
    data_diff = data_final - data_zero  # correctly does *NOT* have nans

    ind_lastnz = 0

    # Use the readnoise and gain for good pixels only
    rn_sect_rav = rn_sect.flatten()[good_pix]
    rn_2_r = rn_sect_rav * rn_sect_rav

    gain_sect_r = gain_sect.flatten()[good_pix]

    # Calculate the sigma for nonzero gain values
    sigma_ir = data_final.copy() * 0.0
    numer_ir = data_final.copy() * 0.0

    # Calculate the SNR for pixels from the readnoise, the gain, and the
    # difference between the last and first reads for pixels where this results
    # in a positive SNR. Otherwise set the SNR to 0.
    sqrt_arg = rn_2_r + data_diff * gain_sect_r
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        wh_pos = np.where((sqrt_arg >= 0.) & (gain_sect_r != 0.))
    numer_ir[wh_pos] = \
        np.sqrt(rn_2_r[wh_pos] + data_diff[wh_pos] * gain_sect_r[wh_pos])
    sigma_ir[wh_pos] = numer_ir[wh_pos] / gain_sect_r[wh_pos]
    snr = data_diff * 0.
    snr[wh_pos] = data_diff[wh_pos] / sigma_ir[wh_pos]
    snr[np.isnan(snr)] = 0.0
    snr[snr < 0.] = 0.0

    del wh_pos

    gain_sect_r = 0
    numer_ir = 0
    data_diff = 0
    sigma_ir = 0

    power_wt_r = calc_power(snr)  # Get the interpolated power for this SNR
    # Make array of number of good groups, and exponents for each pixel
    num_nz = (data_masked != 0.).sum(0)  # number of nonzero groups per pixel
    nrd_data_a = num_nz.copy()
    num_nz = 0

    nrd_prime = (nrd_data_a - 1) / 2.
    nrd_data_a = 0

    # Calculate inverse read noise^2 for use in weights
    # Suppress, then re-enable, harmless arithmetic warning
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    invrdns2_r = 1. / rn_2_r
    warnings.resetwarnings()

    rn_sect = 0
    fnz = 0

    # Set optimal weights for each group of each pixel;
    #    for all pixels at once, loop over the groups
    wt_h = np.zeros(data_masked.shape, dtype=np.float32)

    for jj_rd in range(data_masked.shape[0]):
        wt_h[jj_rd, :] = \
            abs((abs(jj_rd - nrd_prime) / nrd_prime) ** power_wt_r) * invrdns2_r

    wt_h[np.isnan(wt_h)] = 0.
    wt_h[np.isinf(wt_h)] = 0.

    # For all pixels, 'roll' up the leading zeros such that the 0th group of
    #  each pixel is the lowest nonzero group for that pixel
    wh_m2d_f = np.logical_not(c_mask_2d[0, :])  # ramps with initial group False
    while wh_m2d_f.sum() > 0:
        data_masked[:, wh_m2d_f] = np.roll(data_masked[:, wh_m2d_f], -1, axis=0)
        c_mask_2d[:, wh_m2d_f] = np.roll(c_mask_2d[:, wh_m2d_f], -1, axis=0)
        xvalues[:, wh_m2d_f] = np.roll(xvalues[:, wh_m2d_f], -1, axis=0)
        wh_m2d_f = np.logical_not(c_mask_2d[0, :])

    # Create weighted sums for Poisson noise and read noise
    nreads_wtd = (wt_h * c_mask_2d).sum(axis=0)  # using optimal weights

    sumx = (xvalues * wt_h).sum(axis=0)
    sumxx = (xvalues**2 * wt_h).sum(axis=0)

    c_data_masked = data_masked.copy()
    c_data_masked[np.isnan(c_data_masked)] = 0.
    sumy = (np.reshape((c_data_masked * wt_h).sum(axis=0), sumx.shape))
    sumxy = (xvalues * wt_h * np.reshape(c_data_masked, xvalues.shape)).sum(axis=0)

    return sumx, sumxx, sumxy, sumy, nreads_wtd, xvalues
