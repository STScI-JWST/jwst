#!/usr/bin/env python
from ..stpipe import Pipeline
from .. import datamodels
import os

# step imports
from ..dq_init import dq_init_step
from ..flatfield import flat_field_step
from ..guider_cds import guider_cds_step

__version__ = '0.8.0'

# Define logging
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class GuiderPipeline(Pipeline):
    """
    GuiderPipeline: For FGS observations, apply all calibration 
    steps to raw JWST ramps to produce a 3-D slope product. 
    Included steps are: dq_init, guider_cds, and flat_field.
    """

    # Define aliases to steps
    step_defs = {'dq_init': dq_init_step.DQInitStep,
                 'guider_cds': guider_cds_step.GuiderCdsStep,
                 'flat_field': flat_field_step.FlatFieldStep,
                 }

    # Start the processing
    def process(self, input):

        # Set the output product type
        self.suffix = 'cal'

        log.info('Starting calwebb_guider ...')

        # Open the input
        input = datamodels.GuiderRawModel(input) 

        # Apply the steps
        input = self.dq_init(input)  
        input = self.guider_cds(input)
        input = self.flat_field(input)

        log.info('... ending calwebb_guider')

        return input
