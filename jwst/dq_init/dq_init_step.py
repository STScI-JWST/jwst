#! /usr/bin/env python

from ..stpipe import Step
from .. import datamodels
from . import dq_initialization

class DQInitStep(Step):
    """

    DQInitStep:  Initialize the Data Quality extension from the
    mask reference file.  Also initialize the error extension

    """

    reference_file_types = ['mask']

    def process(self, input):

        with datamodels.open(input) as input_model:

            # Check for consistency between keyword values and data shape
            nints = input_model.data.shape[0]
            ngroups = input_model.data.shape[1]
            nints_kwd = input_model.meta.exposure.nints
            ngroups_kwd = input_model.meta.exposure.ngroups
            if nints != nints_kwd:
                self.log.error("Keyword 'NINTS' value of '{0} does not match data array size of '{1}'".format(nints_kwd,nints))
                raise ValueError("Bad data dimensions")
            if ngroups != ngroups_kwd:
                self.log.error("Keyword 'NGROUPS' value of '{0}' does not match data array size of '{1}'".format(ngroups_kwd,ngroups))
                raise ValueError("Bad data dimensions")

            # Retreive the mask reference file name
            self.mask_filename = self.get_reference_file(input_model, 'mask')
            self.log.info('Using MASK reference file %s', self.mask_filename)

            # Check for a valid reference file
            if self.mask_filename == 'N/A':
                self.log.warning('No MASK reference file found')
                self.log.warning('DQ initialization step will be skipped')
                result = input_model.copy()
                result.meta.cal_step.dq_init = 'SKIPPED'
                return result

            # Load the reference file
            mask_model = datamodels.MaskModel(self.mask_filename)

            # Apply the step
            result = dq_initialization.correct_model(input_model, mask_model)

            # Close the reference file
            mask_model.close()

        return result
