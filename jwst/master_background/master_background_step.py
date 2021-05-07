from os.path import basename
import numpy as np
from stdatamodels.properties import merge_tree

from ..stpipe import Step
from .. import datamodels
from ..combine_1d.combine1d import combine_1d_spectra
from .expand_to_2d import expand_to_2d

__all__ = ["MasterBackgroundStep"]


class MasterBackgroundStep(Step):
    """
    MasterBackgroundStep:  Compute and subtract master background from spectra
    """

    spec = """
        user_background = string(default=None) # Path to user-supplied master background
        save_background = boolean(default=False) # Save computed master background
        force_subtract = boolean(default=False) # Force subtracting master background
        output_use_model = boolean(default=True)
    """

    def process(self, input):
        """
        Compute and subtract a master background spectrum

        Parameters
        ----------
        input : `~jwst.datamodels.ImageModel`, `~jwst.datamodels.IFUImageModel`, `~jwst.datamodels.ModelContainer`, association
            Input target datamodel(s) to which master background subtraction is
            to be applied

        user_background : None, string, or `~jwst.datamodels.MultiSpecModel`
            Optional user-supplied master background 1D spectrum, path to file
            or opened datamodel

        save_background : bool, optional
            Save computed master background.

        force_subtract : bool, optional
            Optional user-supplied flag that overrides step logic to force subtraction of the
            master background.
            Default is False, in which case the step logic determines if the calspec2 background step
            has already been applied and, if so, the master background step is skipped.
            If set to True, the step logic is bypassed and the master background is subtracted.

        Returns
        -------
        result : `~jwst.datamodels.ImageModel`, `~jwst.datamodels.IFUImageModel`, `~jwst.datamodels.ModelContainer`
            The background-subtracted science datamodel(s)
        """

        with datamodels.open(input) as input_data:
            # Make the input data available to self
            self.input_data = input_data

            # First check if we should even do the subtraction.  If not, bail.
            if not self._do_sub:
                result = input_data.copy()
                self.record_step_status(result, 'master_background', success=False)
                return result

            # Check that data is a supported datamodel. If not, bail.
            if not isinstance(input_data, (
                datamodels.ModelContainer,
                datamodels.MultiSlitModel,
                datamodels.ImageModel,
                datamodels.IFUImageModel,
            )):
                result = input_data.copy()
                self.log.warning(
                    "Input %s of type %s cannot be handled.  Step skipped.",
                    input, type(input)
                )
                self.record_step_status(result, 'master_background', success=False)
                return result

            # If user-supplied master background, subtract it
            if self.user_background:
                if isinstance(input_data, datamodels.ModelContainer):
                    input_data, _ = split_container(input_data)
                    del _
                    result = datamodels.ModelContainer()
                    result.update(input_data)
                    background_2d_collection = datamodels.ModelContainer()
                    background_2d_collection.update(input_data)
                    for model in input_data:
                        background_2d = expand_to_2d(model, self.user_background)
                        result.append(subtract_2d_background(model, background_2d))
                        background_2d_collection.append(background_2d)
                    # Record name of user-supplied master background spectrum
                    for model in result:
                        model.meta.background.master_background_file = basename(self.user_background)
                # Use user-supplied master background and subtract it
                else:
                    background_2d = expand_to_2d(input_data, self.user_background)
                    background_2d_collection = background_2d
                    result = subtract_2d_background(input_data, background_2d)
                    # Record name of user-supplied master background spectrum
                    result.meta.background.master_background_file = basename(self.user_background)
            # Compute master background and subtract it
            else:
                if isinstance(input_data, datamodels.ModelContainer):
                    input_data, background_data = split_container(input_data)
                    asn_id = input_data.meta.asn_table.asn_id

                    for model in background_data:
                        # Check if the background members are nodded x1d extractions.
                        # Or background from dedicated background exposures
                        # Use "bkgdtarg == False" so we don't also get None cases
                        # for simulated data that didn't bother populating this
                        # keyword
                        this_is_ifu_extended = False
                        if ((model.meta.exposure.type == 'MIR_MRS' or model.meta.exposure.type == 'NRS_IFU')
                                and model.meta.target.source_type == 'EXTENDED'):
                            this_is_ifu_extended = True

                        if model.meta.observation.bkgdtarg is False or this_is_ifu_extended:
                            self.log.debug("Copying BACKGROUND column "
                                           "to SURF_BRIGHT")
                            copy_background_to_surf_bright(model)

                    master_background = combine_1d_spectra(
                        background_data,
                        exptime_key='exposure_time',
                    )

                    background_data.close()

                    result = datamodels.ModelContainer()
                    result.update(input_data)
                    background_2d_collection = datamodels.ModelContainer()
                    background_2d_collection.update(input_data)
                    for model in input_data:
                        background_2d = expand_to_2d(model, master_background)
                        result.append(subtract_2d_background(model, background_2d))
                        background_2d_collection.append(background_2d)

                    input_data.close()

                else:
                    result = input_data.copy()
                    input_data.close()
                    self.log.warning(
                        "Input %s of type %s cannot be handled without user-supplied background.  Step skipped.",
                        input, type(input)
                    )
                    self.record_step_status(result, 'master_background', success=False)
                    return result

                # Save the computed background if requested by user
                if self.save_background:
                    self.save_model(master_background, suffix='masterbg1d', asn_id=asn_id)
                    self.save_model(background_2d_collection, suffix='masterbg2d', asn_id=asn_id)

            self.record_step_status(result, 'master_background', success=True)

        return result

    @property
    def _do_sub(self):
        """
        Decide if subtraction is to be done

        Encapsulates logic that checks if background step has already been run
        on the data, or if the user has selected to force_subtract regardless.

        Returns
        -------
        do_sub : bool
            If ``True``, do the subtraction
        """
        do_sub = True
        if not self.force_subtract:
            input_data = self.input_data
            # check if the input data is a model container. If it is then loop over
            # container and see if the background was subtracted in calspec2.
            # If all data was background subtracted, skip master bgk subtraction.
            # If there is a mixture of some being background subtracted, don't
            # subtract and print warning message
            if isinstance(input_data, datamodels.ModelContainer):
                isub = 0
                for indata in input_data:
                    if indata.meta.cal_step.back_sub == 'COMPLETE' or \
                       indata.meta.cal_step.master_background == 'COMPLETE':
                        do_sub = False
                        isub += 1

                if not do_sub and isub == len(input_data):
                    self.log.info(
                        "Not subtracting master background, background was subtracted in calspec2")
                    self.log.info("To force the master background to be subtracted from this data, "
                                  "run again and set force_subtract = True.")

                if not do_sub and isub != len(input_data):
                    self.log.warning("Not subtracting master background.")
                    self.log.warning("Input data contains a mixture of data with and without "
                                     "background subtraction done in calspec2.")
                    self.log.warning("To force the master background to be subtracted from this data, "
                                     "run again and set force_subtract = True.")
            # input data is a single file
            else:
                if input_data.meta.cal_step.back_sub == 'COMPLETE' or \
                   input_data.meta.cal_step.master_background == 'COMPLETE':
                    do_sub = False
                    self.log.info(
                        "Not subtracting master background, background was subtracted in calspec2")
                    self.log.info("To force the master background to be subtracted from this data, "
                                  "run again and set force_subtract = True.")

        return do_sub


def copy_background_to_surf_bright(spectrum):
    """Copy the background column to the surf_bright column in a MultiSpecModel in-place"""
    for spec in spectrum.spec:
        spec.spec_table['SURF_BRIGHT'][:] = spec.spec_table['BACKGROUND'].copy()
        spec.spec_table['SB_ERROR'][:] = spec.spec_table['BKGD_ERROR'].copy()
        # Zero out the background column for safety
        spec.spec_table['BACKGROUND'][:] = 0
        # Set BERROR to dummy val of 0.0, as in extract_1d currently
        spec.spec_table['BKGD_ERROR'][:] = 0.


def split_container(container):
    """Divide a ModelContainer with science and background into one of each
    """
    asn = container.meta.asn_table.instance
    background = datamodels.ModelContainer()
    science = datamodels.ModelContainer()
    for product in asn['products']:
        for member in product['members']:
            if member['exptype'].lower() == 'science':
                science.append(datamodels.open(member['expname']))
            if member['exptype'].lower() == 'background':
                background.append(datamodels.open(member['expname']))

    # Pass along the association table to the output science container
    science.meta.asn_table = {}
    science.meta.pool_name = container.meta.pool_name
    science.meta.table_name = container.meta.table_name
    merge_tree(science.meta.asn_table._instance, asn)
    # Prune the background members from the table
    for p in science.meta.asn_table.instance['products']:
        p['members'] = [m for m in p['members'] if m['exptype'].lower() != 'background']
    return science, background


def subtract_2d_background(source, background):
    """Subtract a 2D background

    Parameters
    ----------
    source : `~jwst.datamodels.DataModel` or `~jwst.datamodels.ModelContainer`
        The input science data.

    background : `~jwst.datamodels.DataModel`
        The input background data.  Must be the same datamodel type as `source`.
        For a `~jwst.datamodels.ModelContainer`, the source and background
        models in the input containers must match one-to-one.

    Returns
    -------
    `~jwst.datamodels.DataModel`
        Background subtracted from source.
    """

    def _subtract_2d_background(model, background):
        result = model.copy()
        # Handle individual NIRSpec FS, NIRSpec MOS
        if isinstance(model, datamodels.MultiSlitModel):
            for slit, slitbg in zip(result.slits, background.slits):
                slit.data -= slitbg.data
                slit.dq = np.bitwise_or(slit.dq, slitbg.dq)

        # Handle MIRI LRS, MIRI MRS and NIRSpec IFU
        elif isinstance(model, (datamodels.ImageModel, datamodels.IFUImageModel)):
            result.data -= background.data
            result.dq = np.bitwise_or(result.dq, background.dq)
        else:
            # Shouldn't get here.
            raise RuntimeError("Input type {} is not supported."
                               .format(type(model)))
        return result

    # Handle containers of many datamodels
    if isinstance(source, datamodels.ModelContainer):
        result = datamodels.ModelContainer()
        result.update(source)
        for model, bg in zip(source, background):
            result.append(_subtract_2d_background(model, bg))

    # Handle single datamodels
    elif isinstance(source, (datamodels.ImageModel, datamodels.IFUImageModel, datamodels.MultiSlitModel)):
        result = _subtract_2d_background(source, background)

    else:
        # Shouldn't get here.
        raise RuntimeError("Input type {} is not supported."
                           .format(type(source)))

    return result
