#! /usr/bin/env python

import numpy as np

from ..stpipe import Step
from .. import datamodels

from . import ramp_fit
# from stcal.ramp_fitting import ramp_fit

from ..lib import reffile_utils  # TODO remove
from ..lib import pipe_utils

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


__all__ = ["RampFitStep"]


def get_ref_subs(model, readnoise_model, gain_model, nframes):
    """
    Get readnoise array for calculation of variance of noiseless ramps, and
    the gain array in case optimal weighting is to be done. The returned
    readnoise has been multiplied by the gain.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    readnoise_model : instance of data Model
        readnoise for all pixels

    gain_model : instance of gain Model
        gain for all pixels

    nframes : int
        number of frames averaged per group; from the NFRAMES keyword. Does
        not contain the groupgap.

    Returns
    -------
    readnoise_2d : float, 2D array
        readnoise subarray

    gain_2d : float, 2D array
        gain subarray
    """
    if reffile_utils.ref_matches_sci(model, gain_model):
        gain_2d = gain_model.data
    else:
        log.info('Extracting gain subarray to match science data')
        gain_2d = reffile_utils.get_subarray_data(model, gain_model)

    if reffile_utils.ref_matches_sci(model, readnoise_model):
        readnoise_2d = readnoise_model.data.copy()
    else:
        log.info('Extracting readnoise subarray to match science data')
        readnoise_2d = reffile_utils.get_subarray_data(model, readnoise_model)

    return readnoise_2d, gain_2d


def create_image_model(input_model, image_info):
    """
    Creates an ImageModel from the computed arrays from ramp_fit.

    Parameter
    ---------
    input_model: RampModel
        Input RampModel for which the output ImageModel is created.

    image_info: tuple
        The ramp fitting arrays needed for the ImageModel.

    Parameter
    ---------
    out_model: ImageModel
        The output ImageModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, err = image_info
    imshape = data.shape
    out_model = datamodels.ImageModel(data=np.zeros(imshape, dtype=np.float32),
                                      dq=np.zeros(imshape, dtype=np.uint32),
                                      var_poisson=np.zeros(imshape, dtype=np.float32),
                                      var_rnoise=np.zeros(imshape, dtype=np.float32),
                                      err=np.zeros(imshape, dtype=np.float32))
    # ... and add all keys from input
    out_model.update(input_model)
    out_model.data = data
    out_model.dq = dq
    out_model.var_poisson = var_poisson
    out_model.var_rnoise = var_rnoise
    out_model.err = err

    return out_model


def create_integration_model(input_model, integ_info):
    """
    Creates an ImageModel from the computed arrays from ramp_fit.

    Parameter
    ---------
    input_model: RampModel
        Input RampModel for which the output CubeModel is created.

    integ_info: tuple
        The ramp fitting arrays needed for the CubeModel for each integration.

    Parameter
    ---------
    int_model: CubeModel
        The output CubeModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, int_times, err = integ_info
    int_model = datamodels.CubeModel(
        data=np.zeros(data.shape, dtype=np.float32),
        dq=np.zeros(data.shape, dtype=np.uint32),
        var_poisson=np.zeros(data.shape, dtype=np.float32),
        var_rnoise=np.zeros(data.shape, dtype=np.float32),
        err=np.zeros(data.shape, dtype=np.float32))
    int_model.int_times = None
    int_model.update(input_model)  # ... and add all keys from input

    int_model.data = data
    int_model.dq = dq
    int_model.var_poisson = var_poisson
    int_model.var_rnoise = var_rnoise
    int_model.err = err
    int_model.int_times = int_times

    return int_model


def create_optional_results_model(opt_info):
    """
    Creates an ImageModel from the computed arrays from ramp_fit.

    Parameter
    ---------
    opt_info: tuple
        The ramp fitting arrays needed for the RampFitOutputModel.

    Parameter
    ---------
    opt_model: RampFitOutputModel
        The optional RampFitOutputModel to be returned from the ramp fit step.
    """
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info
    opt_model = datamodels.RampFitOutputModel(
        slope=slope,
        sigslope=sigslope,
        var_poisson=var_poisson,
        var_rnoise=var_rnoise,
        yint=yint,
        sigyint=sigyint,
        pedestal=pedestal,
        weights=weights,
        crmag=crmag)

    return opt_model


class RampFitStep (Step):

    """
    This step fits a straight line to the value of counts vs. time to
    determine the mean count rate for each pixel.
    """

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
        maximum_cores = option('none', 'quarter', 'half', 'all', default='none') # max number of processes to create
    """

    # Prior to 04/26/17, the following were also in the spec above:
    #      algorithm = option('OLS', 'GLS', default='OLS') # 'OLS' or 'GLS'
    #      weighting = option('unweighted', 'optimal', default='unweighted') \
    #      # 'unweighted' or 'optimal'
    # As of 04/26/17, the only allowed algorithm is 'ols', and the
    #      only allowed weighting is 'optimal'.

    algorithm = 'ols'      # Only algorithm allowed for Build 7.1
#    algorithm = 'gls'       # 032520

    weighting = 'optimal'  # Only weighting allowed for Build 7.1

    reference_file_types = ['readnoise', 'gain']

    def process(self, input):

        with datamodels.RampModel(input) as input_model:
            max_cores = self.maximum_cores
            readnoise_filename = self.get_reference_file(input_model, 'readnoise')
            gain_filename = self.get_reference_file(input_model, 'gain')

            log.info('Using READNOISE reference file: %s', readnoise_filename)
            readnoise_model = datamodels.ReadnoiseModel(readnoise_filename)
            log.info('Using GAIN reference file: %s', gain_filename)
            gain_model = datamodels.GainModel(gain_filename)

            # Try to retrieve the gain factor from the gain reference file.
            # If found, store it in the science model meta data, so that it's
            # available later in the gain_scale step, which avoids having to
            # load the gain ref file again in that step.
            if gain_model.meta.exposure.gain_factor is not None:
                input_model.meta.exposure.gain_factor = \
                    gain_model.meta.exposure.gain_factor

            log.info('Using algorithm = %s' % self.algorithm)
            log.info('Using weighting = %s' % self.weighting)

            buffsize = ramp_fit.BUFSIZE
            if self.algorithm == "GLS":
                buffsize //= 10

            # Get gain arrays, subrrays if desired.
            frames_per_group = input_model.meta.exposure.nframes
            readnoise_2d, gain_2d = get_ref_subs(
                input_model, readnoise_model, gain_model, frames_per_group)

            # Set int_times depending on model meta data.
            if pipe_utils.is_tso(input_model) and hasattr(input_model, 'int_times'):
                input_model.int_times = input_model.int_times
            else:
                input_model.int_times = None

            image_info, integ_info, opt_info, gls_opt_model = ramp_fit.ramp_fit(
                input_model, buffsize,
                self.save_opt, readnoise_2d, gain_2d, self.algorithm,
                self.weighting, max_cores
            )

            readnoise_model.close()
            gain_model.close()

        # Save the OLS optional fit product, if it exists
        if opt_info is not None:
            opt_model = create_optional_results_model(opt_info)
            self.save_model(opt_model, 'fitopt', output_file=self.opt_name)

        '''
        # GLS removed from code, since it's not implemented right now.
        # Save the GLS optional fit product, if it exists
        if gls_opt_model is not None:
            self.save_model(
                gls_opt_model, 'fitoptgls', output_file=self.opt_name
            )
        '''

        if image_info is not None:
            out_model = create_image_model(input_model, image_info)
            out_model.meta.bunit_data = 'DN/s'
            out_model.meta.bunit_err = 'DN/s'
            out_model.meta.cal_step.ramp_fit = 'COMPLETE'
            if (input_model.meta.exposure.type in ['NRS_IFU', 'MIR_MRS']) or (
                input_model.meta.exposure.type in ['NRS_AUTOWAVE', 'NRS_LAMP'] and
                    input_model.meta.instrument.lamp_mode == 'IFU'):

                out_model = datamodels.IFUImageModel(out_model)

        if integ_info is not None:
            int_model = create_integration_model(input_model, integ_info)
            int_model.meta.bunit_data = 'DN/s'
            int_model.meta.bunit_err = 'DN/s'
            int_model.meta.cal_step.ramp_fit = 'COMPLETE'

        return out_model, int_model
