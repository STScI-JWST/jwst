import os
from collections import defaultdict
import os.path as op
import traceback

from .. import datamodels
from ..assign_wcs.util import NoDataOnDetectorError
from ..lib.exposure_types import is_nrs_ifu_flatlamp, is_nrs_ifu_linelamp
from ..stpipe import Pipeline

# step imports
from ..assign_wcs import assign_wcs_step
from ..background import background_step
from ..barshadow import barshadow_step
from ..cube_build import cube_build_step
from ..extract_1d import extract_1d_step
from ..extract_2d import extract_2d_step
from ..flatfield import flat_field_step
from ..fringe import fringe_step
from ..imprint import imprint_step
from ..master_background import master_background_step
from ..msaflagopen import msaflagopen_step
from ..pathloss import pathloss_step
from ..photom import photom_step
from ..resample import resample_spec_step
from ..srctype import srctype_step
from ..straylight import straylight_step
from ..wavecorr import wavecorr_step

from ..master_background import nirspec_utils

__all__ = ['Spec2Pipeline']

# Classify various exposure types.
NRS_SLIT_TYPES = ['NRS_FIXEDSLIT', 'NRS_BRIGHTOBJ', 'NRS_MSASPEC',
                  'NRS_LAMP', 'NRS_AUTOWAVE', 'NRS_AUTOFLAT']
WFSS_TYPES = ["NIS_WFSS", "NRC_GRISM", "NRC_WFSS"]
GRISM_TYPES = ['NRC_TSGRISM'] + WFSS_TYPES


class Spec2Pipeline(Pipeline):
    """
    Spec2Pipeline: Processes JWST spectroscopic exposures from Level 2a to 2b.
    Accepts a single exposure or an association as input.

    Included steps are:
    assign_wcs, background subtraction, NIRSpec MSA imprint subtraction,
    NIRSpec MSA bad shutter flagging, 2-D subwindow extraction, flat field,
    source type decision, straylight, fringe, pathloss, barshadow,  photom,
    resample_spec, cube_build, and extract_1d.
    """

    spec = """
        save_bsub = boolean(default=False)        # Save background-subracted science
        fail_on_exception = boolean(default=True) # Fail if any product fails.
    """

    # Define aliases to steps
    step_defs = {
        'bkg_subtract': background_step.BackgroundStep,
        'assign_wcs': assign_wcs_step.AssignWcsStep,
        'imprint_subtract': imprint_step.ImprintStep,
        'msa_flagging': msaflagopen_step.MSAFlagOpenStep,
        'extract_2d': extract_2d_step.Extract2dStep,
        'master_background': master_background_step.MasterBackgroundStep,
        'wavecorr': wavecorr_step.WavecorrStep,
        'flat_field': flat_field_step.FlatFieldStep,
        'srctype': srctype_step.SourceTypeStep,
        'straylight': straylight_step.StraylightStep,
        'fringe': fringe_step.FringeStep,
        'pathloss': pathloss_step.PathLossStep,
        'barshadow': barshadow_step.BarShadowStep,
        'photom': photom_step.PhotomStep,
        'resample_spec': resample_spec_step.ResampleSpecStep,
        'cube_build': cube_build_step.CubeBuildStep,
        'extract_1d': extract_1d_step.Extract1dStep
    }

    # Main processing
    def process(self, data):
        """Entrypoint for this pipeline

        Parameters
        ----------
        input: str, Level2 Association, or DataModel
            The exposure or association of exposures to process
        """
        self.log.info('Starting calwebb_spec2 ...')

        # Setup step parameters required by the pipeline.
        self.resample_spec.save_results = self.save_results
        self.resample_spec.suffix = 's2d'
        self.cube_build.output_type = 'multi'
        self.cube_build.save_results = False
        self.extract_1d.save_results = self.save_results

        # Retrieve the input(s)
        asn = self.load_as_level2_asn(data)

        # Each exposure is a product in the association.
        # Process each exposure.
        results = []
        has_exceptions = False
        for product in asn['products']:
            self.log.info('Processing product {}'.format(product['name']))
            self.output_file = product['name']
            try:
                getattr(asn, 'filename')
            except AttributeError:
                asn.filename = "singleton"
            try:
                result = self.process_exposure_product(
                    product,
                    asn['asn_pool'],
                    asn.filename
                )
            except NoDataOnDetectorError as exception:
                # This error merits a special return
                # status if run from the command line.
                # Bump it up now.
                raise exception
            except Exception:
                traceback.print_exc()
                has_exceptions = True
            else:
                if result is not None:
                    results.append(result)

        if has_exceptions and self.fail_on_exception:
            raise RuntimeError(
                'One or more products failed to process. Failing calibration.'
            )

        # We're done
        self.log.info('Ending calwebb_spec2')

        self.output_use_model = True
        self.suffix = False
        return results

    # Process each exposure
    def process_exposure_product(
            self,
            exp_product,
            pool_name=' ',
            asn_file=' '
    ):
        """Process an exposure found in the association product

        Parameters
        ---------
        exp_product: dict
            A Level2b association product.
        """

        # Find all the member types in the product
        members_by_type = defaultdict(list)
        for member in exp_product['members']:
            members_by_type[member['exptype'].lower()].append(member['expname'])

        # Get the science member. Technically there should only be
        # one. We'll just get the first one found.
        science_member = members_by_type['science']
        if len(science_member) != 1:
            self.log.warning(
                'Wrong number of science exposures found in {}'.format(
                    exp_product['name']
                )
            )
            self.log.warning('    Using only first one.')
        science_member = science_member[0]

        self.log.info('Working on input %s ...', science)
        with self.open_model(science) as input:
            exp_type = input.meta.exposure.type
            if isinstance(input, datamodels.CubeModel):
                multi_int = True
            else:
                multi_int = False

            # Suffixes are dependent on whether the science is multi-integration or not.
            if multi_int:
                suffix = 'calints'
                self.extract_1d.suffix = 'x1dints'
            else:
                suffix = 'cal'
                self.extract_1d.suffix = 'x1d'

            # Apply WCS info
            # check the datamodel to see if it's
            # a grism image, if so get the catalog
            # name from the asn and record it to the meta
            if exp_type in WFSS_TYPES:
                try:
                    input.meta.source_catalog = os.path.basename(members_by_type['sourcecat'][0])
                    self.log.info('Using sourcecat file {}'.format(input.meta.source_catalog))
                except IndexError:
                    if input.meta.source_catalog is None:
                        raise IndexError("No source catalog specified in association or datamodel")

            # Decide on what steps can actually be accomplished based on the
            # provided input.
            self._step_verification(exp_type, members_by_type, multi_int)

            # Start processing the individual steps.
            # `assign_wcs` is the critical step. Without it, processing
            # cannot proceed.
            assign_wcs_exception = None
            try:
                input = self.assign_wcs(input)
            except Exception as exception:
                assign_wcs_exception = exception
            if assign_wcs_exception is not None or \
               input.meta.cal_step.assign_wcs != 'COMPLETE':
                message = (
                    'Assign_wcs processing was skipped.'
                    '\nAborting remaining processing for this exposure.'
                    '\nNo output product will be created.'
                )
                if self.assign_wcs.skip:
                    self.log.warning(message)
                    return
                else:
                    self.log.error(message)
                    if assign_wcs_exception is not None:
                        raise assign_wcs_exception
                    else:
                        raise RuntimeError('Cannot determine WCS.')

        # Steps whose order is the same for all types of input.
        calibrated = self.bkg_subtract(calibrated, members_by_type['background'])
        calibrated = self.imprint_subtract(calibrated, members_by_type['imprint'])
        calibrated = self.msa_flagging(calibrated)

        # The order of the next few steps is tricky, depending on mode:
        # WFSS/Grism data need flat_field before extract_2d, but other modes
        # need extract_2d first. Furthermore, NIRSpec MOS and FS need
        # srctype and wavecorr before flat_field.
        if exp_type in GRISM_TYPES:
            input = self._process_grism(input)
            # Apply flat-field correction
        elif exp_type in NRS_SLIT_TYPES:
            input = self._process_nirspec_slits(input)
        else:
            calibrated = self._process_common(calibrated)

        # Setup result metadata for pools, association, and suffix.
        result.meta.asn.pool_name = pool_name
        result.meta.asn.table_name = op.basename(asn_file)
        result.meta.filename = self.make_output_path(suffix=suffix)

        # Produce a resampled product, either via resample_spec for
        # "regular" spectra or cube_build for IFU data. No resampled
        # product is produced for time-series modes.
        if exp_type in ['NRS_FIXEDSLIT', 'NRS_MSASPEC', 'MIR_LRS-FIXEDSLIT'] \
           and not isinstance(calibrated, datamodels.CubeModel):

            # Call the resample_spec step for 2D slit data
            result_extra = self.resample_spec(result)

        elif (exp_type in ['MIR_MRS', 'NRS_IFU']) or is_nrs_ifu_linelamp(result):

            # Call the cube_build step for IFU data;
            # always create a single cube containing multiple
            # wavelength bands
            result_extra = self.cube_build(result)
            if not self.cube_build.skip:
                self.save_model(resampled[0], 's3d')
        else:
            resampled = calibrated

        # Extract a 1D spectrum from the 2D/3D data
        if exp_type in ['MIR_MRS', 'NRS_IFU'] and self.cube_build.skip:
            # Skip extract_1d for IFU modes where no cube was built
            self.extract_1d.skip = True
        x1d_result = self.extract_1d(result_extra)

        resampled.close()
        x1d.close()

        # That's all folks
        self.log.info(
            'Finished processing product {}'.format(exp_product['name'])
        )

        return calibrated

    def _step_verification(self, exp_type, members_by_type, multi_int):
        """Verify whether requested steps can operate on the given data

        Though ideally this would all be controlled through the pipeline
        parameters, the desire to keep the number of config files down has
        pushed the logic into code.

        Once step and pipeline parameters are retrieved from CRDS, this
        logic can be removed.
        """

        # Check for image-to-image background subtraction can be done.
        if not self.bkg_subtract.skip:
            if exp_type in WFSS_TYPES or len(members_by_type['background']) > 0:

                if exp_type in WFSS_TYPES:
                    members_by_type['background'] = []           # will be overwritten by the step

                # Setup for saving
                self.bkg_subtract.suffix = 'bsub'
                if multi_int:
                    self.bkg_subtract.suffix = 'bsubints'

                # Backwards compatibility
                if self.save_bsub:
                    self.bkg_subtract.save_results = True
            else:
                self.log.debug('Science data does not allow direct background subtraction. Skipping "bkg_subtract".')
                self.bkg_subtract.skip = True

        # Check for imprint subtraction
        imprint = members_by_type['imprint']
        if not self.imprint_subtract.skip:
            if len(imprint) > 0 and (exp_type in ['NRS_MSASPEC', 'NRS_IFU'] or \
               is_nrs_ifu_flatlamp(input)):
                if len(imprint) > 1:
                    self.log.warning('Wrong number of imprint members')
                members_by_type['imprint'] = imprint[0]
            else:
                self.log.debug('Science data does not allow imprint processing. Skipping "imprint_subtraction".')
                self.imprint_subtract.skip = True

        # Check for NIRSpec MSA bad shutter flagging.
        if not self.msa_flagging.skip and exp_type not in ['NRS_MSASPEC', 'NRS_IFU', 'NRS_LAMP',
                                                           'NRS_AUTOFLAT', 'NRS_AUTOWAVE']:
            self.log.debug('Science data does not allow MSA flagging. Skipping "msa_flagging".')
            self.msa_flagging.skip = True

        # Check for straylight correction for MIRI MRS.
        if not self.straylight.skip and exp_type != 'MIR_MRS':
            self.log.debug('Science data does not allow stray light correction. Skipping "straylight".')
            self.straylight.skip = True

        # Apply the fringe correction for MIRI MRS
        if not self.fringe.skip and exp_type != 'MIR_MRS':
            self.log.debug('Science data does not allow fringe correction. Skipping "fringe".')
            self.fringe.skip = True

        # Apply pathloss correction to NIRSpec and NIRISS SOSS exposures
        if not self.pathloss.skip and exp_type not in ['NRS_FIXEDSLIT', 'NRS_MSASPEC', 'NRS_IFU', 'NIS_SOSS']:
            self.log.debug('Science data does not allow pathloss correction. Skipping "pathloss".')
            self.pathloss.skip = True

        # Apply barshadow correction to NIRSPEC MSA exposures
        if not self.barshadow.skip and exp_type != 'NRS_MSASPEC':
            self.log.debug('Science data does not allow barshadow correction. Skipping "barshadow".')
            self.barshadow.skip = True

    def _process_grism(self, data):
        """WFSS & Grism processing

        WFSS/Grism data need flat_field before extract_2d.
        """
        calibrated = self.flat_field(data)
        calibrated = self.extract_2d(calibrated)
        calibrated = self.srctype(calibrated)
        calibrated = self.straylight(calibrated)
        calibrated = self.fringe(calibrated)
        calibrated = self.pathloss(calibrated)
        calibrated = self.barshadow(calibrated)
        calibrated = self.photom(calibrated)

        return calibrated

    def _process_nirspec(self, data):
        """Process NIRSpec

        NIRSpec MOS and FS need srctype and wavecorr before flat_field.
        Also have to deal with master background operations.
        """
        calibrated = self.extract_2d(data)
        calibrated = self.srctype(calibrated)

        # Master background requires a different order of processing.
        if not self.master_background.skip:
            calibrated = self._process_nirspec_masterbackground(calibrated)

        # Now continue calibration of the science.
        calibrated = self.wavecorr(calibrated)
        calibrated = self.flat_field(calibrated)
        calibrated = self.pathloss(calibrated)
        calibrated = self.barshadow(calibrated)
        calibrated = self.photom(calibrated)

        return calibrated

    def _process_nirspec_masterbackground(self, data):
        """Prepare and apply the master background subtraction step

        For MOS, and ignoring FS, the calibration process needs to occur
        twice: Once to calibrate background slits and create a master background.
        Then a second time to calibrate science using the master background.

        Parameters
        ----------
        data : MultiSlitData
            The data to apply the master background subtraction

        Returns
        -------
        msb_subtracted : MultiSlitData
            The background subtracted data.

        Notes
        -----
        The algorithm is as follows:

        - Calibrate all slits
          - For each step:
            - Force the source type to be extended source for all slits.
            - Return the correction array used.
        - Create the 1D master background
        - For each slit
          - Expand out the 1D master background to match the 2D wavelength grid of the slit
          - Reverse-calibrate the 2D background, using the correction arrays calculated above.
          - Subtract the background from the input slit data
        """
        # First pass: just do the calibration to determine the correction
        # arrays.
        pre_calibrated, ff_corrections = self.flat_field(
            data, force_extended=True, return_corrections=True
        )
        pre_calibrated = self.pathloss(pre_calibratedforce_extended=True, return_corrections=True)
        pre_calibrated = self.barshadow(pre_calibrated, force_extended=True, return_corrections=True)
        pre_calibrated = self.photom(pre_calibrated, force_extended=True, return_corrections=True)

        # At this point, assume that `pre_calibrated` is a modified `MultiSlitModel` that
        # is also carrying the science calibration information along with it.
        # The next steps may get wrapped into a single master_background_step, but
        # are split out here for design

        # First create the 1D, fully calibrated master background.
        master_background = nirspec_utils.create_background_from_multislit(pre_calibrated)
        if master_background is None:
            return data

        # Now decalibrate the master background for each individual science slit
        # The steps are split out here for design purposes.
        # First step is to map the master background into a MultiSlitModel
        # where the science slits are replaced by the master background
        # Here the broadcasting from 1D to 2D need also occur.
        mb_multislit = nirspec_utils.map_to_science_slits(pre_calibrated, master_background)

        # Now that the master background is pretending to be science,
        # walk backwards through the steps to uncalibrate, using the
        # calibration factors carried in `pre_calibrated`
        # Yes, using kwargs for the steps is invalid, but for design purposes only.
        mb_multislit = self.photom(mb_multislit, inverse=True, factors=pre_calibrated)
        mb_multislit = self.barshadow(mb_multislit, inverse=True, factors=pre_calibrated)
        mb_multislit = self.pathloss(mb_multislit, inverse=True, factors=pre_calibrated)
        mb_multislit = self.flat_field(mb_multislit, inverse=True, factors=pre_calibrated)

        # Now apply the de-calibrated background to the original science
        # At this point, should just be a slit-to-slit subtraction operation.
        calibrated = apply_master_background(calibrated, mb_multislit)

        return calibrated

    def _process_common(self, data):
        """Common spectral processing"""
        calibrated = self.srctype(data)
        calibrated = self.flat_field(calibrated)
        calibrated = self.straylight(calibrated)
        calibrated = self.fringe(calibrated)
        calibrated = self.pathloss(calibrated)
        calibrated = self.barshadow(calibrated)
        calibrated = self.photom(calibrated)

        return calibrated
