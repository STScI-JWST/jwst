#! /usr/bin/env python

from ..stpipe import Step
from .. import datamodels
from . import wfss_contam


__all__ = ["WfssContamStep"]


class WfssContamStep(Step):
    """
    This Step performs contamination correction of WFSS spectra.
    """

    spec = """
    """

    reference_file_types = ['photom', 'wavelengthrange']

    def process(self, input_model, *args, **kwargs):

        with datamodels.open(input_model) as dm:

            # Get the wavelengthrange ref file
            waverange_ref = self.get_reference_file(dm, 'wavelengthrange')
            self.log.info(f'Using WAVELENGTHRANGE reference file {waverange_ref}')
            waverange_model = datamodels.WavelengthrangeModel(waverange_ref)

            # Get the photom ref file
            photom_ref = self.get_reference_file(dm, 'photom')
            self.log.info(f'Using PHOTOM reference file {photom_ref}')
            photom_model = datamodels.open(photom_ref)

            output_model = wfss_contam.contam_corr(dm, waverange_model, photom_model)

        return output_model
