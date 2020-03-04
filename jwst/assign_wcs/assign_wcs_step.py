#! /usr/bin/env python
from ..stpipe import Step
from .. import datamodels
from .assign_wcs import load_wcs
from .util import MSAFileError

__all__ = ["AssignWcsStep"]


class AssignWcsStep(Step):
    """
    AssignWcsStep: Create a gWCS object and store it in ``Model.meta``.

    Reference file types:

    camera             Camera model (NIRSPEC)
    collimator         Collimator Model (NIRSPEC)
    disperser          Disperser model (NIRSPEC)
    distortion         Spatial distortion model (FGS, MIRI, NIRCAM, NIRISS)
    filteroffset       Filter offsets (MIRI Imager)
    fore               Transform through the FORE optics (NIRSPEC)
    fpa                Transform in the FPA plane (NIRSPEC)
    ifufore            Transforms from the MSA plane to the plane of the IFU slicer (NIRSPEC)
    ifupost            Transforms from the slicer plane to the MSA plane (NIRSPEC)
    ifuslicer          Metrology of the IFU slicer (NIRSPEC)
    msa                Metrology of the MSA plane (NIRSPEC)
    ote                Transform through the Optical Telescope Element (NIRSPEC)
    specwcs            Wavelength calibration models (MIRI, NIRCAM, NIRISS)
    regions            Stores location of the regions on the detector (MIRI)
    wavelengthrange    Typical wavelength ranges (MIRI, NIRCAM, NIRISS, NIRSPEC)

    Parameters
    ----------
    input : `~jwst.datamodels.ImageModel`, `~jwst.datamodels.IFUImageModel`, `~jwst.datamodels.CubeModel`
        Input exposure.
    """

    spec = """
        slit_y_low = float(default=-.55)  # The lower edge of a slit.
        slit_y_high = float(default=.55)  # The upper edge of a slit.

    """

    reference_file_types = ['distortion', 'filteroffset', 'specwcs', 'regions',
                            'wavelengthrange', 'camera', 'collimator', 'disperser',
                            'fore', 'fpa', 'msa', 'ote', 'ifupost',
                            'ifufore', 'ifuslicer']

    def process(self, input, *args, **kwargs):
        reference_file_names = {}
        with datamodels.open(input) as input_model:
            # If input type is not supported, log warning, set to 'skipped', exit
            if not (isinstance(input_model, datamodels.ImageModel) or
                    isinstance(input_model, datamodels.CubeModel) or
                    isinstance(input_model, datamodels.IFUImageModel)):
                self.log.warning("Input dataset type is not supported.")
                self.log.warning("assign_wcs expects ImageModel, IFUImageModel or CubeModel as input.")
                self.log.warning("Skipping assign_wcs step.")
                result = input_model.copy()
                result.meta.cal_step.assign_wcs = 'SKIPPED'
                return result

            for reftype in self.reference_file_types:
                reffile = self.get_reference_file(input_model, reftype)
                reference_file_names[reftype] = reffile if reffile else ""
            self.log.debug(f'reference files used in assign_wcs: {reference_file_names}')

            # Get the MSA metadata file if needed and add to reffiles
            if input_model.meta.exposure.type == "NRS_MSASPEC":
                msa_metadata_file = input_model.meta.instrument.msa_metadata_file
                if msa_metadata_file is not None and msa_metadata_file.strip() not in ["", "N/A"]:
                    msa_metadata_file = self.make_input_path(msa_metadata_file)
                    reference_file_names['msametafile'] = msa_metadata_file
                else:
                    message = "MSA metadata file (MSAMETFL) is required for NRS_MSASPEC exposures."
                    self.log.error(message)
                    raise MSAFileError(message)
            slit_y_range = [self.slit_y_low, self.slit_y_high]
            result = load_wcs(input_model, reference_file_names, slit_y_range)


        return result
