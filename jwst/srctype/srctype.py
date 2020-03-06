import logging
from ..lib import pipe_utils

from .. import datamodels

log = logging.getLogger(__name__)


def set_source_type(input_model):
    """
    Set the source_type, based on APT input or default values.

    Parameters
    ----------
    input_model : `~jwst.datamodels.CubeModel`, `~jwst.datamodels.ImageModel`,
                  `~jwst.datamodels.IFUImageModel`, `~jwst.datamodels.MultiSlitModel`,
                  or `~jwst.datamodels.SlitModel`
        The data model to be processed.

    Returns
    -------
    input_model : `~jwst.datamodels.CubeModel`, `~jwst.datamodels.ImageModel`,
                  `~jwst.datamodels.IFUImageModel`, `~jwst.datamodels.MultiSlitModel`,
                  or `~jwst.datamodels.SlitModel`
        The updated model.
    """

    # Get the exposure type of the input model
    exptype = input_model.meta.exposure.type
    if exptype is None:
        log.error('EXP_TYPE value not found in input')
        raise RuntimeError('Step cannot be executed without an EXP_TYPE value')
    else:
        log.info('Input EXP_TYPE is %s' % exptype)

    # For exposure types that have a single source specification, get the
    # user-supplied source type from the selection they provided in the APT
    if exptype in ['MIR_LRS-FIXEDSLIT', 'MIR_LRS-SLITLESS', 'MIR_MRS',
                   'NRC_TSGRISM', 'NIS_SOSS', 'NRS_FIXEDSLIT',
                   'NRS_BRIGHTOBJ', 'NRS_IFU']:

        bkg_target = input_model.meta.observation.bkgdtarg
        patttype = input_model.meta.dither.primary_type
        user_type = input_model.meta.target.source_type

        if bkg_target:

            # If this image is flagged as a BACKGROUND target, set the
            # source type to EXTENDED regardless of any other settings
            src_type = 'EXTENDED'
            log.info('Exposure is a background target; setting SRCTYPE = %s' % src_type)

        elif pipe_utils.is_tso(input_model):
            src_type = 'POINT'
            log.info('Input is a TSO exposure; setting SRCTYPE = %s' % src_type)

        elif (patttype is not None) and (('NOD' in patttype) or ('POINT-SOURCE' in patttype)):

            # Set all nodded exposures to POINT source type
            src_type = 'POINT'
            log.info('Exposure is nodded; setting SRCTYPE = %s' % src_type)

        elif user_type in ['POINT', 'EXTENDED']:

            # Use the value supplied by the user
            src_type = user_type
            log.info('Using input SRCTYPE = %s' % src_type)

        else:

            # Set a default value based on the exposure type
            if exptype == 'MIR_MRS':
                src_type = 'EXTENDED'
            else:
                src_type = 'POINT'

            log.info('Input SRCTYPE is unknown; setting default SRCTYPE = %s' % src_type)

        # Set the source type in the global meta attribute
        input_model.meta.target.source_type = src_type

        # If the input contains one or more slit instances,
        # set the value in each slit too
        if isinstance(input_model, datamodels.SlitModel):
            input_model.source_type = src_type

        elif input_model.meta.exposure.type == 'NRS_FIXEDSLIT':

            # NIRSpec fixed-slit is a special case: Apply the source type
            # determined above to only the primary slit (the one in which
            # the target is located). Set all other slits to the default
            # value, which for NRS_FIXEDSLIT is 'POINT'.
            default_type = 'POINT'
            primary_slit = input_model.meta.instrument.fixed_slit
            log.debug(' primary_slit = {}'.format(primary_slit))
            for slit in input_model.slits:
                if slit.name == primary_slit:
                    slit.source_type = src_type
                else:
                    slit.source_type = default_type
                log.debug(' slit {} = {}'.format(slit.name, slit.source_type))

    # For NIRSpec MSA exposures, read the stellarity value for the
    # source in each extracted slit and set the point/extended value
    # based on the stellarity.
    elif exptype == 'NRS_MSASPEC':

        # Loop over the input slits
        for slit in input_model.slits:
            stellarity = slit.stellarity

            # Eventually the stellarity value will be compared against
            # a threshold value from a reference file. For now, the
            # threshold is hardwired.
            if stellarity < 0.0:
                slit.source_type = 'UNKNOWN'
            elif stellarity > 0.75:
                slit.source_type = 'POINT'
            else:
                slit.source_type = 'EXTENDED'

            log.info('source_id=%g, stellarity=%g, type=%s' %
                     (slit.source_id, stellarity, slit.source_type))

        # Set the source type value in the primary header to
        # a harmless default
        input_model.meta.target.source_type = 'UNKNOWN'

    # Set all TSO exposures to POINT
    elif pipe_utils.is_tso(input_model):
        src_type = 'POINT'
        log.info('Input is a TSO exposure; setting default SRCTYPE = %s' % src_type)
        input_model.meta.target.source_type = src_type

    # Unrecognized exposure type; set to UNKNOWN as default
    else:
        log.warning('EXP_TYPE %s not applicable to this operation' % exptype)
        src_type = 'UNKNOWN'
        log.warning('Setting SRCTYPE = %s' % src_type)
        input_model.meta.target.source_type = src_type

    # We're done
    return input_model
