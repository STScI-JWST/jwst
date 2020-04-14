#!/usr/bin/env python
import logging
from ..stpipe import Pipeline
from .. import datamodels

# step imports
from ..group_scale import group_scale_step
from ..dq_init import dq_init_step
from ..saturation import saturation_step
from ..ipc import ipc_step
from ..superbias import superbias_step
from ..refpix import refpix_step
from ..rscd import rscd_step
from ..firstframe import firstframe_step
from ..lastframe import lastframe_step
from ..linearity import linearity_step
from ..dark_current import dark_current_step
from ..persistence import persistence_step
from ..jump import jump_step
from ..ramp_fitting import ramp_fit_step
from ..gain_scale import gain_scale_step

__all__ = ['Detector1Pipeline']

# Define logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Detector1Pipeline(Pipeline):
    """
    Detector1Pipeline: Apply all calibration steps to raw JWST
    ramps to produce a 2-D slope product. Included steps are:
    group_scale, dq_init, saturation, ipc, superbias, refpix, rscd,
    lastframe, linearity, dark_current, persistence, jump detection,
    ramp_fit, and gain_scale.
    """

    spec = """
        save_calibrated_ramp = boolean(default=False)
    """

    # Define aliases to steps
    step_defs = {'group_scale': group_scale_step.GroupScaleStep,
                 'dq_init': dq_init_step.DQInitStep,
                 'saturation': saturation_step.SaturationStep,
                 'ipc': ipc_step.IPCStep,
                 'superbias': superbias_step.SuperBiasStep,
                 'refpix': refpix_step.RefPixStep,
                 'rscd': rscd_step.RSCD_Step,
                 'firstframe': firstframe_step.FirstFrameStep,
                 'lastframe': lastframe_step.LastFrameStep,
                 'linearity': linearity_step.LinearityStep,
                 'dark_current': dark_current_step.DarkCurrentStep,
                 'persistence': persistence_step.PersistenceStep,
                 'jump': jump_step.JumpStep,
                 'ramp_fit': ramp_fit_step.RampFitStep,
                 'gain_scale': gain_scale_step.GainScaleStep,
                 }

    # start the actual processing
    def process(self, input):

        log.info('Starting calwebb_detector1 ...')

        # open the input as a RampModel
        input = datamodels.RampModel(input)

        # propagate output_dir to steps that might need it
        self.dark_current.output_dir = self.output_dir
        self.ramp_fit.output_dir = self.output_dir

        if input.meta.instrument.name == 'MIRI':

            # process MIRI exposures;
            # the steps are in a different order than NIR
            log.debug('Processing a MIRI exposure')

            input = self.group_scale(input)
            input = self.dq_init(input)
            input1 = input.copy()
            input.close()
            del input
            input1 = self.saturation(input1)
            input2 = input1.copy()
            input1.close()
            del input1
            input2 = self.ipc(input2)
            input3 = input2.copy()
            input2.close()
            del input2
            input3 = self.firstframe(input3)
            input4 = input3.copy()
            input3.close()
            del input3
            input4 = self.lastframe(input4)
            input5 = input4.copy()
            input4.close()
            del input4
            input5 = self.linearity(input5)
            input6 = input5.copy()
            input5.close()
            del input5
            input6 = self.rscd(input6)
            input7 = input6.copy()
            input6.close()
            del input6
            input7 = self.dark_current(input7)
            input8 = input7.copy()
            input7.close()
            del input7
            input8 = self.refpix(input8)

            # skip until MIRI team has figured out an algorithm
            #input = self.persistence(input)

        else:

            # process Near-IR exposures
            log.debug('Processing a Near-IR exposure')

            input = self.group_scale(input)
            input = self.dq_init(input)
            input1 = input.copy()
            input.close()
            del input
            input1 = self.saturation(input1)
            input2 = input1.copy()
            input1.close()
            del input1
            input2 = self.ipc(input2)
            input3 = input2.copy()
            input2.close()
            del input2
            input3 = self.superbias(input3)
            input4 = input3.copy()
            input3.close()
            del input3
            input4 = self.refpix(input4)
            input5 = input4.copy()
            input4.close()
            del input4
            input5 = self.linearity(input5)
            input6 = input5.copy()
            input5.close()
            del input5

            # skip persistence for NIRSpec
            if input6.meta.instrument.name != 'NIRSPEC':
                input6 = self.persistence(input6)
                input7 = input6.copy()
                input6.close()
                del input6
            else:
                input7 = input6.copy()
                input6.close()
                del input6

            input7 = self.dark_current(input7)
            input8 = input7.copy()
            input7.close()
            del input7
        # apply the jump step
        input8 = self.jump(input8)
        input = input8.copy()
        input8.close()
        del input8

        # save the corrected ramp data, if requested
        if self.save_calibrated_ramp:
            self.save_model(input, 'ramp')

        # apply the ramp_fit step
        # This explicit test on self.ramp_fit.skip is a temporary workaround
        # to fix the problem that the ramp_fit step ordinarily returns two
        # objects, but when the step is skipped due to `skip = True` in a
        # cfg file, only the input is returned when the step is invoked.
        if self.ramp_fit.skip:
            input = self.ramp_fit(input)
            ints_model = None
        else:
            input, ints_model = self.ramp_fit(input)

        # apply the gain_scale step to the exposure-level product
        self.gain_scale.suffix = 'gain_scale'
        input = self.gain_scale(input)

        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            self.gain_scale.suffix = 'gain_scaleints'
            ints_model = self.gain_scale(ints_model)
            self.save_model(ints_model, 'rateints')

        # setup output_file for saving
        self.setup_output(input)

        log.info('... ending calwebb_detector1')

        return input

    def setup_output(self, input):
        # Determine the proper file name suffix to use later
        if input.meta.cal_step.ramp_fit == 'COMPLETE':
            self.suffix = 'rate'
        else:
            self.suffix = 'ramp'
