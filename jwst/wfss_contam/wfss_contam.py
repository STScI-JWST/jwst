#
#  Top level module for WFSS contamination correction.
#
from os.path import splitext

from jwst import datamodels
from .observations import Observation
from .sens1d import get_photom_data
from ..lib.suffix import replace_suffix

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def contam_corr(input_model, waverange, photom):
    """
    The main WFSS contam correction function

    Parameters
    ----------
    input_model : `~jwst.datamodels.MultiSlitModel`
        Input data model containing 2D spectral cutouts
    waverange : `~jwst.datamodels.WavelengthrangeModel`
        Wavelength range reference file model
    photom : `~jwst.datamodels.NrcWfssPhotomModel` or `~jwst.datamodels.NisWfssPhotomModel`
        Photom (flux cal) reference file model

    Returns
    -------
    output_model : `~jwst.datamodels.MultiSlitModel`
      A copy of the input_model that has been corrected

    """
    output_model = input_model.copy()

    # Get the segmentation map for this grism exposure
    seg_model = datamodels.open(input_model.meta.segmentation_map)

    # Get the direct image from which the segmentation was constructed
    direct_file = input_model.meta.direct_image
    image_names = [direct_file]

    # Load the sensitivity (inverse flux cal) data for this mode
    filter = input_model.meta.instrument.filter
    pupil = input_model.meta.instrument.pupil
    sens_waves, sens_response = get_photom_data(photom, filter, pupil, order=1)

    # Get the grism WCS from the input model
    grism_wcs = input_model.slits[0].meta.wcs

    # Create a simulated grism image containing all of the sources
    # defined in the segmentation map
    obs = Observation(image_names, seg_model, grism_wcs, waverange, filter,
                      sens_waves, sens_response, order=1, max_split=2,
                      max_cpu=1, boundaries=[0, 2047, 0, 2047])

    log.debug("Creating full simulated grism image with all sources")
    obs.disperse_all()
    simul_all = obs.simulated_image

    # Save the full simulated grism image
    simul_model = datamodels.ImageModel(data=simul_all)
    root, _ = splitext(input_model.meta.filename)
    simul_model.meta.filename = replace_suffix(root, 'simul') + '.fits'
    log.info(f"Saving full simulated image as {simul_model.meta.filename}")
    simul_model.save(simul_model.meta.filename)

    # Loop over all slits/sources
    log.debug("Creating contam image for each individual source")
    for slit in output_model.slits:

        # Create simulated spectrum for this source only
        id = slit.source_id
        order = slit.meta.wcsinfo.spectral_order
        if order != 1:
            continue
        obs.disperse_chunk(id)
        this_source = obs.simulated_image

        # Contamination estimate is full simulated image
        # minus this source
        contam = simul_all - this_source

        # Subtract the cutout of the contam image from the slit image
        x1 = slit.xstart - 1
        x2 = x1 + slit.xsize
        y1 = slit.ystart - 1
        y2 = y1 + slit.ysize
        slit.data -= contam[y1:y2, x1:x2]

    # Set the step status to COMPLETE
    output_model.meta.cal_step.wfss_contam = 'COMPLETE'

    return output_model
